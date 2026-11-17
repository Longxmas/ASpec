# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
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

"""Checkpoint tensor reads shared by independently installed Target and Draft roles.

Moved from the retained TorchSpec-derived Draft base without changing loading semantics.
"""

import glob
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def load_tensor_from_pretrained(model_path: str, key: str) -> torch.Tensor:
    """Read a single tensor by key from an HF checkpoint directory or hub repository id."""
    if not os.path.exists(model_path):
        # model_path is a huggingface repository, so locate its local cache first
        return load_tensor_from_pretrained(snapshot_download(repo_id=model_path), key)

    # check if there is file ending with index.json
    glob_path = os.path.join(model_path, "*.index.json")
    index_json_path = glob.glob(glob_path)

    if len(index_json_path) == 0:
        # No index.json found, look for single model file
        safetensors_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt") as f:
                return f.get_tensor(key)

        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            return state_dict[key]

        raise FileNotFoundError(
            f"No index.json, model.safetensors or pytorch_model.bin found in {model_path}"
        )
    if len(index_json_path) > 1:
        raise FileNotFoundError(f"Multiple index.json files found in {model_path}")
    index_json_path = index_json_path[0]

    with open(index_json_path, "r") as f:
        index_json = json.load(f)
    ckpt_file = index_json["weight_map"][key]

    if ckpt_file.endswith(".safetensors"):
        with safe_open(os.path.join(model_path, ckpt_file), framework="pt") as f:
            return f.get_tensor(key)
    state_dict = torch.load(os.path.join(model_path, ckpt_file))
    return state_dict[key]
