# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from ASpec's retained native MTP implementation, itself adapted from
# Speculators 7ec7aba9450b533334e8f7992baf4e93be3c1a57 (Apache-2.0).
# The shared computation uses official Transformers decoder/normalization/RoPE
# and does not import Draft execution backends into the Target environment.
"""Native predictor mathematics shared by trainable MTP and frozen capture."""


def native_mtp_fingerprint(tensors):
    """Bind frozen predictor content without retaining its weights in a Draft."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.digest()


def predict_mtp_embeddings(model, hidden_states, input_embeds, position_ids, layout=None):
    import torch

    embeds = model.pre_fc_norm_embedding(input_embeds)
    hidden = model.pre_fc_norm_hidden(hidden_states)
    projected = model.fc(torch.cat((embeds, hidden), dim=-1))
    rotary_positions = position_ids.unsqueeze(0).expand(3, -1, -1)
    positions = model.rotary_emb(projected, rotary_positions)
    output = model.midlayer(projected, position_embeddings=positions,
                            position_ids=position_ids, attention_mask=None,
                            **({"mtp_layout": layout} if layout is not None else {}))
    return model.norm(output)
