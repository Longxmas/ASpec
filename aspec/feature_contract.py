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

"""Canonical target-feature contract and lightweight runtime validation."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from typing import Any, Mapping

FEATURE_CONTRACT_SCHEMA_VERSION = "aspec.target-feature-contract/v0alpha1"
_EXECUTION_METADATA_KEYS = (
    "target_feature_output",
    "transport_data_plane",
    "transport_protocol",
    "feature_affinity",
    "source_node_ip",
    "assigned_draft_node_ip",
    "draft_attention_backend",
    "draft_context_parallel_size",
    "draft_feature_layout",
    "direct_cp_sharded",
    "usp_sharded",
)


class FeatureContractError(ValueError):
    """A planned or observed target-feature layout violates its contract."""


def _contract_id(contract: Mapping[str, Any]) -> str:
    payload = {key: copy.deepcopy(value) for key, value in contract.items() if key != "contract_id"}
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _dtype_name(value: Any) -> str:
    name = str(value).lower().replace("torch.", "")
    aliases = {
        "long": "int64",
        "float": "float32",
        "half": "float16",
    }
    return aliases.get(name, name)


def build_target_feature_contract(
    *,
    drafter_family: str,
    target_hidden_size: int,
    target_num_hidden_layers: int,
    aux_hidden_state_layers: list[int],
    store_last_hidden_states: bool,
    mtp: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload contract without starting a backend.

    Algorithm identity belongs to the Workload objective. Identical payloads
    remain shareable across algorithms; shape, layer, dtype and value-stage
    differences still produce distinct identities.
    """

    if not isinstance(target_hidden_size, int) or target_hidden_size <= 0:
        raise FeatureContractError(
            f"draft target_hidden_size must be a positive integer, got {target_hidden_size!r}"
        )
    if not isinstance(target_num_hidden_layers, int) or target_num_hidden_layers <= 0:
        raise FeatureContractError(
            "draft target_num_hidden_layers must be a positive integer, got "
            f"{target_num_hidden_layers!r}"
        )
    if not aux_hidden_state_layers or not all(
        isinstance(layer_id, int) and layer_id >= 0 for layer_id in aux_hidden_state_layers
    ):
        raise FeatureContractError(
            "aux_hidden_state_layers must contain non-negative integers, got "
            f"{aux_hidden_state_layers!r}"
        )
    if len(aux_hidden_state_layers) != len(set(aux_hidden_state_layers)):
        raise FeatureContractError(
            "aux_hidden_state_layers must be unique and ordered explicitly, got "
            f"{aux_hidden_state_layers!r}"
        )
    if max(aux_hidden_state_layers) >= target_num_hidden_layers:
        raise FeatureContractError(
            "aux_hidden_state_layers must be within the target decoder; "
            f"max layer {max(aux_hidden_state_layers)} >= {target_num_hidden_layers}"
        )

    tensors: dict[str, Any] = {
        "hidden_states": {
            "semantic": "selected_decoder_layer_outputs",
            "dtype": "bfloat16",
            "shape": ["sequence", len(aux_hidden_state_layers) * target_hidden_size],
            "layout": "layers_concatenated_last_dim",
        },
        "input_ids": {
            "semantic": "target_token_ids",
            "dtype": "int64",
            "shape": ["sequence"],
            "layout": "sequence",
        },
    }
    if drafter_family == "mtp":
        if len(aux_hidden_state_layers) != 1 or store_last_hidden_states:
            raise FeatureContractError("Native MTP requests one post-norm feature, without a duplicate teacher tensor")
        tensors["hidden_states"] = {
            "semantic": "target_final_hidden_state", "dtype": "bfloat16",
            "shape": ["sequence", target_hidden_size], "layout": "sequence_hidden",
            "value_stage": "post_norm",
        }
    if mtp is not None:
        if (not isinstance(mtp, Mapping) or not {"native_config", "embedding_key"} <= set(mtp)
                or set(mtp) - {"native_config", "embedding_key", "alignment"}):
            raise FeatureContractError("MTP capture requires native_config and embedding_key")
        native = mtp["native_config"]
        if (not isinstance(native, Mapping) or native.get("model_type") != "qwen3_5_text"
                or native.get("hidden_size") != target_hidden_size
                or native.get("num_hidden_layers") != target_num_hidden_layers
                or not isinstance(mtp["embedding_key"], str) or not mtp["embedding_key"]):
            raise FeatureContractError("MTP capture differs from its Target binding")
        if mtp.get("alignment", "anchor_post_norm_v1") != "anchor_post_norm_v1":
            raise FeatureContractError("Unsupported native MTP feature alignment")
        tensors["mtp_hidden_states"] = {
            "semantic": "native_mtp_prediction_hidden_state", "dtype": "bfloat16",
            "shape": ["sequence", target_hidden_size], "layout": "sequence_hidden",
            "value_stage": "post_norm", "alignment": "anchor_post_norm_v1",
        }
    omissions = []
    target_distribution = "token_labels"
    if store_last_hidden_states:
        tensors["last_hidden_states"] = {
            "semantic": "target_final_hidden_state",
            "dtype": "bfloat16",
            "shape": ["sequence", target_hidden_size],
            "layout": "sequence_hidden",
            "value_stage": "post_norm",
        }
        target_distribution = "consumer_lm_head_from_last_hidden_states"
        omissions.append(
            {
                "tensor": "logits",
                "reason": "computed_by_consumer_lm_head",
            }
        )
    else:
        omissions.append({"tensor": "logits", "reason": "not_required_by_objective"})

    contract: dict[str, Any] = {
        "schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "producer": {"backend": "sglang", "mode": "prefill_only"},
        "consumer": {
            "target_distribution": target_distribution,
        },
        "transport": "mooncake",
        "target_hidden_size": target_hidden_size,
        "target_num_hidden_layers": target_num_hidden_layers,
        "aux_hidden_state_layers": list(aux_hidden_state_layers),
        "tensors": tensors,
        "omissions": omissions,
        **({"mtp": {**copy.deepcopy(dict(mtp)), "alignment": "anchor_post_norm_v1"}} if mtp is not None else {}),
    }
    contract["contract_id"] = _contract_id(contract)
    return contract


def validate_feature_contract(contract: Mapping[str, Any]) -> None:
    """Reject malformed or mutated contracts at the runtime boundary."""

    if contract.get("schema_version") != FEATURE_CONTRACT_SCHEMA_VERSION:
        raise FeatureContractError(
            f"Unsupported target feature contract schema: {contract.get('schema_version')!r}"
        )
    observed_id = contract.get("contract_id")
    expected_id = _contract_id(contract)
    if observed_id != expected_id:
        raise FeatureContractError(
            f"Target feature contract identity mismatch: {observed_id!r} != {expected_id!r}"
        )
    if contract.get("producer") != {"backend": "sglang", "mode": "prefill_only"}:
        raise FeatureContractError(
            f"Unsupported target feature producer contract: {contract.get('producer')!r}"
        )
    if contract.get("transport") != "mooncake":
        raise FeatureContractError(
            f"Unsupported target feature transport: {contract.get('transport')!r}"
        )
    tensors = contract.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise FeatureContractError("Target feature contract must declare tensors")
    if ("mtp" in contract) != ("mtp_hidden_states" in tensors):
        raise FeatureContractError("MTP producer identity and tensor must be declared together")


class FeatureContractRuntime:
    """Validate one participant against a shared contract and expose audit counters."""

    def __init__(self, contract: Mapping[str, Any], *, role: str, participant: str) -> None:
        validate_feature_contract(contract)
        if role not in {"producer", "consumer"}:
            raise FeatureContractError(f"Unknown target feature participant role: {role!r}")
        self.contract = copy.deepcopy(dict(contract))
        self.role = role
        self.participant = participant
        self._binding_validations = 0
        self._metadata_validations = 0
        self._tensor_validations = 0
        self._execution_metadata: dict[str, set[Any]] = {}
        self._lock = threading.Lock()

    @property
    def contract_id(self) -> str:
        return self.contract["contract_id"]

    def validate_producer_binding(
        self,
        *,
        hidden_size: int,
        num_hidden_layers: int | None,
        aux_hidden_state_layers: list[int],
        store_last_hidden_states: bool,
    ) -> None:
        if self.role != "producer":
            raise FeatureContractError("Only a producer may validate a producer binding")
        expected_hidden = self.contract["target_hidden_size"]
        if hidden_size != expected_hidden:
            raise FeatureContractError(
                f"{self.participant}: target hidden size {hidden_size} does not match "
                f"contract {expected_hidden}"
            )
        expected_layers = self.contract["target_num_hidden_layers"]
        if num_hidden_layers is not None and num_hidden_layers != expected_layers:
            raise FeatureContractError(
                f"{self.participant}: target layer count {num_hidden_layers} does not match "
                f"contract {expected_layers}"
            )
        expected_aux = self.contract["aux_hidden_state_layers"]
        if list(aux_hidden_state_layers) != expected_aux:
            raise FeatureContractError(
                f"{self.participant}: auxiliary layers {list(aux_hidden_state_layers)!r} do not "
                f"match contract {expected_aux!r}"
            )
        expected_last = "last_hidden_states" in self.contract["tensors"]
        if bool(store_last_hidden_states) != expected_last:
            raise FeatureContractError(
                f"{self.participant}: store_last_hidden_states={store_last_hidden_states!r} does "
                f"not match contract {expected_last!r}"
            )
        with self._lock:
            self._binding_validations += 1

    def validate_metadata(
        self,
        *,
        tensor_shapes: Mapping[str, Any],
        tensor_dtypes: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        observed_contract_id = metadata.get("aspec_feature_contract_id")
        if observed_contract_id != self.contract_id:
            raise FeatureContractError(
                f"{self.participant}: sample contract identity {observed_contract_id!r} does not "
                f"match {self.contract_id!r}"
            )

        expected_tensors = self.contract["tensors"]
        expected_names = set(expected_tensors)
        if set(tensor_shapes) != expected_names:
            raise FeatureContractError(
                f"{self.participant}: tensor shape keys {sorted(tensor_shapes)!r} do not match "
                f"contract {sorted(expected_names)!r}"
            )
        if set(tensor_dtypes) != expected_names:
            raise FeatureContractError(
                f"{self.participant}: tensor dtype keys {sorted(tensor_dtypes)!r} do not match "
                f"contract {sorted(expected_names)!r}"
            )

        sequence_length = None
        for name, tensor_contract in expected_tensors.items():
            observed_shape = tuple(tensor_shapes[name])
            expected_shape = tensor_contract["shape"]
            if len(observed_shape) != len(expected_shape):
                raise FeatureContractError(
                    f"{self.participant}: {name} rank {len(observed_shape)} does not match "
                    f"contract rank {len(expected_shape)}"
                )
            for observed_dim, expected_dim in zip(observed_shape, expected_shape, strict=True):
                if expected_dim == "sequence":
                    if not isinstance(observed_dim, int) or observed_dim <= 0:
                        raise FeatureContractError(
                            f"{self.participant}: {name} has invalid sequence dimension "
                            f"{observed_dim!r}"
                        )
                    if sequence_length is None:
                        sequence_length = observed_dim
                    elif observed_dim != sequence_length:
                        raise FeatureContractError(
                            f"{self.participant}: {name} sequence dimension {observed_dim} does "
                            f"not match {sequence_length}"
                        )
                elif observed_dim != expected_dim:
                    raise FeatureContractError(
                        f"{self.participant}: {name} shape {observed_shape!r} does not match "
                        f"contract {expected_shape!r}"
                    )
            observed_dtype = _dtype_name(tensor_dtypes[name])
            if observed_dtype != tensor_contract["dtype"]:
                raise FeatureContractError(
                    f"{self.participant}: {name} dtype {observed_dtype!r} does not match "
                    f"contract {tensor_contract['dtype']!r}"
                )
        with self._lock:
            self._metadata_validations += 1
            for key in _EXECUTION_METADATA_KEYS:
                value = metadata.get(key)
                if isinstance(value, (str, int, float, bool)):
                    self._execution_metadata.setdefault(key, set()).add(value)

    def validate_tensors(
        self,
        tensors: Mapping[str, Any],
        *,
        expected_shapes: Mapping[str, Any],
        expected_dtypes: Mapping[str, Any],
    ) -> None:
        expected_names = set(self.contract["tensors"])
        if set(tensors) != expected_names:
            raise FeatureContractError(
                f"{self.participant}: received tensor keys {sorted(tensors)!r} do not match "
                f"contract {sorted(expected_names)!r}"
            )
        for name in expected_names:
            tensor = tensors[name]
            observed_shape = tuple(tensor.shape)
            expected_shape = tuple(expected_shapes[name])
            if observed_shape != expected_shape:
                raise FeatureContractError(
                    f"{self.participant}: received {name} shape {observed_shape!r} does not match "
                    f"transport metadata {expected_shape!r}"
                )
            observed_dtype = _dtype_name(tensor.dtype)
            expected_dtype = _dtype_name(expected_dtypes[name])
            if observed_dtype != expected_dtype:
                raise FeatureContractError(
                    f"{self.participant}: received {name} dtype {observed_dtype!r} does not match "
                    f"transport metadata {expected_dtype!r}"
                )
        with self._lock:
            self._tensor_validations += 1

    def status(self) -> dict[str, Any]:
        with self._lock:
            binding_validations = self._binding_validations
            metadata_validations = self._metadata_validations
            tensor_validations = self._tensor_validations
            execution_metadata = {
                key: sorted(values, key=lambda value: str(value))
                for key, values in sorted(self._execution_metadata.items())
            }
        if self.role == "producer":
            validated = binding_validations > 0 and metadata_validations > 0
        else:
            validated = metadata_validations > 0 and tensor_validations > 0
        return {
            "participant": self.participant,
            "role": self.role,
            "contract_id": self.contract_id,
            "state": "validated" if validated else "pending_observation",
            "binding_validations": binding_validations,
            "metadata_validations": metadata_validations,
            "tensor_validations": tensor_validations,
            "execution_metadata": execution_metadata,
        }


def shared_feature_union(contracts):
    """Union compatible payloads; retain every consumer's explicit layer order."""
    contracts = list(contracts)
    if not contracts:
        raise FeatureContractError("Shared source requires feature contracts")
    first = contracts[0]
    if first["tensors"]["hidden_states"].get("value_stage") == "post_norm":
        for contract in contracts:
            validate_feature_contract(contract)
        if any(contract != first for contract in contracts):
            raise FeatureContractError("Shared source has incompatible post-norm feature contracts")
        return copy.deepcopy(first)
    for contract in contracts:
        validate_feature_contract(contract)
        for key in ("schema_version", "producer", "transport", "target_hidden_size",
                    "target_num_hidden_layers"):
            if contract[key] != first[key]:
                raise FeatureContractError("Shared source has incompatible Target feature contracts: " + key)
        for name, tensor in contract["tensors"].items():
            if name == "hidden_states":
                tensor = {**tensor, "shape": ["sequence", first["target_hidden_size"]]}
                reference = {**first["tensors"][name], "shape": tensor["shape"]}
            else:
                reference = next(c["tensors"][name] for c in contracts if name in c["tensors"])
            if tensor != reference:
                raise FeatureContractError("Shared source has incompatible tensor values: " + name)
    mtp = [c["mtp"] for c in contracts if "mtp" in c]
    if mtp and any(policy != mtp[0] for policy in mtp):
        raise FeatureContractError("Shared source has incompatible native MTP bindings")
    return build_target_feature_contract(
        drafter_family="shared_source", target_hidden_size=first["target_hidden_size"],
        target_num_hidden_layers=first["target_num_hidden_layers"],
        aux_hidden_state_layers=sorted({layer for c in contracts for layer in c["aux_hidden_state_layers"]}),
        store_last_hidden_states=any("last_hidden_states" in c["tensors"] for c in contracts),
        **({"mtp": mtp[0]} if mtp else {}),
    )
