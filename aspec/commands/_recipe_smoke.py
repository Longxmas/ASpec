# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-Draft recipe GPU smoke with checkpoint-state audit, not a benchmark."""

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import ray

from aspec.commands._checkpoint_audit import audit_state as _base_audit_state
from aspec.training.trainer_actor import TrainerActor


def audit_state(trainer):
    result = _base_audit_state(trainer)
    draft = getattr(trainer, "draft_model", None)
    if getattr(getattr(draft, "config", None), "mtp_conditioning", None):
        from aspec.commands._checkpoint_audit import tensor_hash

        result["mtp_conditioning"] = {
            "policy": draft.config.mtp_conditioning,
            "frozen": tensor_hash((n, p) for n, p in draft.named_parameters()
                                  if n.startswith(("mtp_source.", "embed_tokens."))),
            "fusion": tensor_hash(draft.mtp_fusion.named_parameters())
                      if draft.mtp_fusion is not None else None,
        }
        expected = [p for p in draft.parameters() if p.requires_grad]
        if [id(p) for p in trainer.optimizer.model_params] != [id(p) for p in expected]:
            raise AssertionError("MTP conditioning optimizer ownership differs from the Draft")
    if getattr(getattr(draft, "config", None), "ffn_type", "dense") == "ling_moe":
        result["ling_expert_bias"] = [layer.mlp.gate.expert_bias.detach().cpu().tolist()
                                      for layer in draft.layers]
    if hasattr(draft, "eagle4_policy"):
        from aspec.commands._checkpoint_audit import tensor_hash

        result["eagle4"] = {
            "policy": draft.eagle4_policy, "candidate_steps": int(draft.candidate_head_steps),
            "body": tensor_hash((n, p) for n, p in draft.named_parameters()
                                if not n.startswith("efficient_lm_head")),
            "candidate": tensor_hash((n, p) for n, p in draft.named_parameters()
                                     if n.startswith("efficient_lm_head")),
            "buffers": tensor_hash(draft.named_buffers()),
            "trainable": [n for n, p in draft.named_parameters() if p.requires_grad],
        }
        if draft.candidate_stage:
            expected = [p for n, p in draft.named_parameters() if n.startswith("efficient_lm_head")]
            if [id(p) for p in trainer.optimizer.model_params] != [id(p) for p in expected]:
                raise AssertionError("Candidate optimizer owns parameters outside the two retrieval factors")
    if getattr(getattr(draft, "config", None), "dsa_training_stage", None) is not None:
        from aspec.commands._checkpoint_audit import tensor_hash
        from aspec.config.draft_attention import resolve_draft_attention

        result["dsa"] = {
            "policy": resolve_draft_attention(draft.config.to_dict())["dsa"],
            "main": tensor_hash((name, parameter) for name, parameter in draft.named_parameters()
                                if ".dsa_layer.indexer." not in name),
            "indexer": tensor_hash((name, parameter) for name, parameter in draft.named_parameters()
                                   if ".dsa_layer.indexer." in name),
            "buffers": tensor_hash(draft.named_buffers()),
            "trainable": [name for name, parameter in draft.named_parameters() if parameter.requires_grad],
        }
        expected = [parameter for name, parameter in draft.named_parameters()
                    if parameter.requires_grad]
        if [id(parameter) for parameter in trainer.optimizer.model_params] != [id(parameter) for parameter in expected]:
            raise AssertionError("DSA optimizer ownership differs from the current stage")
    return result


class AuditedTrainerActor(TrainerActor):
    def init(self, args, role, mooncake_config=None, with_ref=False):
        result = super().init(args, role, mooncake_config, with_ref)
        self._audit_root = Path(args.output_dir)
        self._audit_suffix = "" if args.world_size == 1 else f"-rank{self._rank}"
        initial = audit_state(self._trainer)
        (self._audit_root / f"recipe-audit-initial{self._audit_suffix}.json").write_text(
            json.dumps(initial, sort_keys=True)
        )
        if getattr(args, "load_path", None) and not args.continual_training:
            import hashlib

            import torch

            from aspec.commands._checkpoint_audit import compare_checkpoint
            from aspec.training.checkpoint import resolve_resume_model_dir

            source = resolve_resume_model_dir(args).parent
            compared = compare_checkpoint(self._trainer, source)
            if any(value["relative_l2"] or value["max_absolute"] for value in compared.values()):
                raise AssertionError(compared)
            rng = torch.load(source / "rng" / f"rank-{self._rank:05d}.pt", weights_only=False)
            hashes = {
                "torch_rng_sha256": hashlib.sha256(rng["torch"].numpy().tobytes()).hexdigest(),
                "cuda_rng_sha256": hashlib.sha256(rng["cuda"].cpu().numpy().tobytes()).hexdigest(),
            }
            if any(initial[key] != value for key, value in hashes.items()):
                raise AssertionError("Restored RNG differs from saved checkpoint")
            (self._audit_root / f"recipe-resume-diff{self._audit_suffix}.json").write_text(
                json.dumps({"state": compared, "rng_exact": True}, sort_keys=True)
            )
        return result

    def train_from_queue(self, step, num_batches):
        result = super().train_from_queue(step, num_batches)
        draft = self._trainer.draft_model
        if getattr(draft.config, "ffn_type", "dense") == "ling_moe":
            snapshot = {"step": self._trainer.global_step,
                        "expert_bias": [layer.mlp.gate.expert_bias.detach().cpu().tolist()
                                        for layer in draft.layers]}
            with (self._audit_root / f"recipe-ling-steps{self._audit_suffix}.jsonl").open("a") as stream:
                stream.write(json.dumps(snapshot, sort_keys=True) + "\n")
        return result

    def save_model(self, step, force_sync=False):
        super().save_model(step, force_sync)
        current = audit_state(self._trainer)
        if current.get("mtp_conditioning"):
            initial = json.loads((self._audit_root / f"recipe-audit-initial{self._audit_suffix}.json").read_text())
            if current["mtp_conditioning"]["frozen"] != initial["mtp_conditioning"]["frozen"]:
                raise AssertionError("MTP conditioning changed its frozen source/embedding")
        if current.get("eagle4", {}).get("policy", {}).get("stage") == "candidate_head":
            initial = json.loads((self._audit_root / f"recipe-audit-initial{self._audit_suffix}.json").read_text())
            if initial["eagle4"]["body"] != current["eagle4"]["body"]:
                raise AssertionError("Candidate training changed frozen Draft weights")
        if current.get("dsa"):
            initial = json.loads((self._audit_root / f"recipe-audit-initial{self._audit_suffix}.json").read_text())
            stage = current["dsa"]["policy"]["training_stage"]
            frozen = "main" if stage == "indexer_warmup" else "indexer" if stage == "dense" else None
            if frozen is not None and initial["dsa"][frozen] != current["dsa"][frozen]:
                raise AssertionError(f"DSA {stage} changed frozen {frozen} weights")
        (self._audit_root / f"recipe-audit-checkpoint-{step:07d}{self._audit_suffix}.json").write_text(
            json.dumps(current, sort_keys=True)
        )
        names = [f"recipe-audit-initial{self._audit_suffix}.json",
                 f"recipe-audit-checkpoint-{step:07d}{self._audit_suffix}.json",
                 f"recipe-resume-diff{self._audit_suffix}.json", "recipe-fault.json"]
        return {"output_dir": str(self._audit_root), "files": {
            name: (self._audit_root / name).read_text() for name in names
            if (self._audit_root / name).is_file()}}



def main(*, trainer_class=AuditedTrainerActor):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--cluster-address", help="Explicit address of a task-owned multi-node cluster")
    args, overrides = parser.parse_known_args()
    if args.gpus < 2:
        parser.error("Select Target plus Draft GPUs")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.cluster_address:
        if args.cluster_address in {"auto", "local"}:
            parser.error("Use an explicit task-owned cluster address")
        os.environ["RAY_ADDRESS"] = args.cluster_address
    else:
        token = root / "ray-auth-token"
        with os.fdopen(os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(secrets.token_hex(32))
        os.environ["RAY_AUTH_TOKEN_PATH"] = str(token)
        os.environ.pop("RAY_AUTH_TOKEN", None)
        os.environ.pop("RAY_ADDRESS", None)
    from aspec import multi_draft, train_entry
    from aspec.ray.train_group import RayTrainGroup

    original_save = RayTrainGroup.save_model

    def save_with_audits(group, step, force_sync=False):
        results = original_save(group, step, force_sync=force_sync)
        root = Path(group.args.output_dir).resolve()
        for result in results:
            if not result or Path(result["output_dir"]).resolve() != root:
                raise ValueError("Missing or misplaced per-rank audit")
            for name, content in result["files"].items():
                if Path(name).name != name or not name.startswith("recipe-"):
                    raise ValueError("Invalid audit filename")
                (root / name).write_text(content)
        return results

    RayTrainGroup.save_model = save_with_audits
    train_entry.TrainerActor = trainer_class
    multi_draft.TrainerActor = trainer_class
    sys.argv = [sys.argv[0], "--config", args.config, *overrides, f"output_dir={root}"]
    (root / "recipe-command.json").write_text(json.dumps({
        "diagnostic_only": True, "argv": sys.argv, "python": sys.executable,
        "total_gpu_budget": args.gpus,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2))
    try:
        # Use the normal launcher, including managed-local's visibility split,
        # before initializing Draft's Ray runtime. The audit budget spans both
        # roles; it is never a request for that many GPUs in Draft's Ray head.
        train_entry.main()
    finally:
        RayTrainGroup.save_model = original_save
        ray.shutdown()


if __name__ == "__main__":
    main()
