# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native MTP features and dual-anchor conditioning, inspired by DMTP (Shen, 2026).

Method: https://sylvan820.github.io/D-MTP/ (November 9 residual update).
Independent implementation reusing ASpec's retained DFlash backbone/objective
and native Qwen3.5 predictor. Configuration, scheduling, optimization and
recovery stay in the existing ASpec workflow. This is not a claim of
reproducing the author's training recipe or downstream serving runtime.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MTPPrefix:
    # Row i is the state predicting x[i+2], after consuming x[i+1].
    prediction_hidden: torch.Tensor | None = None
    # One actual MTP proposal per anchor; None means teacher-forced training.
    first_candidate_ids: torch.Tensor | None = None


class MTPAnchorFusion(nn.Module):
    def __init__(self, embedding_size, mtp_hidden_size, eps=1e-6):
        super().__init__()
        from aspec.models.draft.dflash import DFlashRMSNorm

        self.embedding_norm = DFlashRMSNorm(embedding_size, eps=eps)
        self.hidden_norm = DFlashRMSNorm(mtp_hidden_size, eps=eps)
        self.proj = nn.Linear(embedding_size + mtp_hidden_size, embedding_size, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, embeddings, hidden):
        features = torch.cat((self.embedding_norm(embeddings),
                              self.hidden_norm(hidden.detach())), dim=-1)
        return embeddings + self.proj(features)


class FrozenMTPBinding(nn.Module):
    """Checkpoint the same-Target native-weight identity, with no predictor allocation."""

    def __init__(self, native_config):
        super().__init__()
        self.native_config = native_config
        self.register_buffer("fingerprint", torch.zeros(32, dtype=torch.uint8))

    def _current_fingerprint(self, target_model_path):
        from aspec.models.draft.base import load_tensor_from_pretrained
        from aspec.models.draft.mtp import NativeMTPDraftModel
        from aspec.models.mtp_features import native_mtp_fingerprint

        template = FrozenMTPFeatures(self.native_config)
        return native_mtp_fingerprint(
            (name, load_tensor_from_pretrained(target_model_path, NativeMTPDraftModel.native_key(name)))
            for name in template.state_dict()
        )

    def initialize_native_weights(self, target_model_path):
        digest = self._current_fingerprint(target_model_path)
        self.load_state_dict({"fingerprint": torch.tensor(list(digest), dtype=torch.uint8)}, assign=True)

    def validate_native_weights(self, target_model_path):
        actual = bytes(self.fingerprint.cpu().tolist())
        if actual != self._current_fingerprint(target_model_path):
            raise ValueError("Frozen MTP checkpoint fingerprint differs from its Target")


class FrozenMTPFeatures(nn.Module):
    """The native predictor body, borrowing the Draft's frozen Target embedding.

    It owns no embedding or output head, and therefore does not duplicate the
    vocabulary-sized Target bindings in memory/checkpoints. Native predictor
    mathematics are shared with NativeMTPDraftModel, rather than reimplemented.
    """

    def __init__(self, native_config):
        super().__init__()
        from aspec.models.draft.mtp import NativeMTPConfig, NativeMTPDraftModel

        with torch.device("meta"):
            source = NativeMTPDraftModel(NativeMTPConfig(native_config=native_config))
        self.native = source.native
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "fc", "midlayer", "norm"):
            setattr(self, name, getattr(source, name))
        self.rotary_emb = type(source.rotary_emb)(self.native, device="cpu")
        self._rotary_ready = False
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def _apply(self, fn, recurse=True):
        self._rotary_ready = False
        return super()._apply(fn, recurse=recurse)

    def initialize_native_weights(self, target_model_path):
        from aspec.models.draft.base import load_tensor_from_pretrained
        from aspec.models.draft.mtp import NativeMTPDraftModel

        state = {name: load_tensor_from_pretrained(target_model_path,
                                                  NativeMTPDraftModel.native_key(name))
                 for name in self.state_dict()}
        self.load_state_dict(state, strict=True, assign=True)
        self.requires_grad_(False)

    @torch.no_grad()
    def validate_native_weights(self, target_model_path):
        from aspec.models.draft.base import load_tensor_from_pretrained
        from aspec.models.draft.mtp import NativeMTPDraftModel

        for name, actual in self.state_dict().items():
            if hasattr(actual, "full_tensor"):
                actual = actual.full_tensor()
            expected = load_tensor_from_pretrained(target_model_path,
                                                   NativeMTPDraftModel.native_key(name))
            if actual.shape != expected.shape or not torch.equal(
                actual.cpu(), expected.to(dtype=actual.dtype)
            ):
                raise ValueError("Frozen MTP differs from its Target: " + name)

    @torch.no_grad()
    def forward(self, hidden_states, input_embeds, position_ids):
        from aspec.models.draft.mtp import NativeMTPDraftModel

        if not self._rotary_ready:
            # Device moves, to_empty and checkpoint restoration do not carry
            # the native non-persistent frequencies. Regenerate them once.
            self.rotary_emb = type(self.rotary_emb)(self.native, device=hidden_states.device)
            self._rotary_ready = True
        return NativeMTPDraftModel.predict_embeddings(
            self, hidden_states.detach(), input_embeds.detach(), position_ids
        ).detach()


def training_mtp_prefix(draft, input_ids, last_hidden_states, attention_mask=None):
    policy = draft.config.mtp_conditioning
    if policy["mode"] == "token":
        return MTPPrefix()
    if policy.get("feature_source", "draft") == "target":
        raise ValueError("Target-side MTP conditioning requires captured mtp_hidden_states")
    if last_hidden_states is None or last_hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("MTP hidden conditioning requires aligned post-norm Target final states")
    if input_ids.shape[0] != 1:
        raise ValueError("MTP conditioning currently requires microbatch size one")
    length = input_ids.shape[1]
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("MTP attention mask must align with input token IDs")
        length = int(attention_mask.sum())
        if (not attention_mask[:, :length].bool().all()
                or attention_mask[:, length:].bool().any()):
            raise ValueError("MTP conditioning requires right-padded complete documents")
    if length < 2:
        return MTPPrefix(last_hidden_states.new_zeros(
            (1, max(input_ids.shape[1] - 1, 0), draft.config.target_hidden_size)))
    positions = torch.arange(length - 1, device=input_ids.device).unsqueeze(0)
    hidden = draft.mtp_source(last_hidden_states[:, :length - 1],
                              draft.embed_tokens(input_ids[:, 1:length]), positions)
    hidden = torch.nn.functional.pad(hidden, (0, 0, 0, input_ids.shape[1] - length))
    return MTPPrefix(hidden.detach())


def condition_noise_embeddings(draft, embeddings, input_ids, anchors, keep, prefix, block_size):
    """Add T1 while retaining the original A cutoff and native state timing."""
    if not isinstance(prefix, MTPPrefix):
        raise ValueError("MTP conditioning requires an explicit derived prefix")
    if ((anchors < 1) | (anchors + 1 >= input_ids.shape[1]))[keep].any():
        raise ValueError("MTP anchors need a previous Target state and an in-bounds T1")
    if prefix.first_candidate_ids is None:
        ids = input_ids.gather(1, (anchors + 1).clamp(max=input_ids.shape[1] - 1))
    else:
        ids = prefix.first_candidate_ids
        if ids.shape != anchors.shape or ids.dtype != torch.long:
            raise ValueError("MTP proposal IDs must be int64 with the anchor shape")
    ids = torch.where(keep, ids, torch.full_like(ids, draft.mask_token_id))
    first = draft.embed_tokens(ids)
    if draft.config.mtp_conditioning["mode"] == "hidden":
        hidden = prefix.prediction_hidden
        expected = (input_ids.shape[0], input_ids.shape[1] - 1, draft.config.target_hidden_size)
        if hidden is None or hidden.shape != expected:
            raise ValueError(f"MTP prediction hidden must have shape {expected}")
        indices = (anchors - 1).clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        selected = hidden.gather(1, indices)
        # Inactive features must be finite before trainable normalization/projection;
        # masking their loss afterwards cannot prevent NaN parameter gradients.
        selected = torch.where(keep.unsqueeze(-1), selected, 0.0)
        first = draft.mtp_fusion(first, selected)
    elif prefix.prediction_hidden is not None:
        raise ValueError("Token-only conditioning does not consume MTP hidden features")
    blocks = embeddings.reshape(input_ids.shape[0], anchors.shape[1], block_size, -1)
    return torch.cat((blocks[:, :, :1], first.unsqueeze(2), blocks[:, :, 2:]), dim=2).flatten(1, 2)


@torch.no_grad()
def frozen_mtp_predictions(mtp, input_ids, final_hidden_states):
    """Causal document-local predictor pass; no future token is consumed.

    x[a] is the known pending token A. Its prediction state is output row a-1,
    formed from Target h[a-1] and x[a], and predicts the unverified x[a+1].
    """
    if mtp.training or any(p.requires_grad for p in mtp.parameters()):
        raise ValueError("The native MTP feature producer must be frozen and in eval mode")
    if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2
            or final_hidden_states.shape[:2] != input_ids.shape):
        raise ValueError("MTP feature extraction requires one complete unpadded document")
    positions = torch.arange(input_ids.shape[1] - 1, device=input_ids.device).unsqueeze(0)
    return mtp.predict(final_hidden_states[:, :-1].detach(), input_ids[:, 1:], positions).detach()
