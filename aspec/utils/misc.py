# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import socket
from typing import List

import ray
from transformers import AutoConfig


def get_current_node_ip() -> str:
    """Return the address by which the current process joined Ray.

    A host can have several routable interfaces.  Ray's generic address
    discovery follows the default route, which need not be the interface used
    by the cluster.  Actors that advertise transfer endpoints must use the
    NodeManager address that was already validated by cluster preflight.
    """

    try:
        if ray.is_initialized():
            node_id = ray.get_runtime_context().get_node_id()
            for node in ray.nodes():
                if node.get("Alive", False) and node.get("NodeID") == node_id:
                    return str(node["NodeManagerAddress"]).strip("[]")
    except Exception:
        # Preserve the single-node/pre-Ray fallback used by callers during
        # early initialization.
        pass

    return ray._private.services.get_node_ip_address().strip("[]")


def _is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_free_port(start_port=10000, consecutive=1):
    """Find a consecutive TCP port range, failing when the search is exhausted."""
    if (
        not isinstance(start_port, int) or isinstance(start_port, bool)
        or not isinstance(consecutive, int) or isinstance(consecutive, bool)
        or not 1 <= start_port <= 65535
        or consecutive < 1 or start_port + consecutive - 1 > 65535
    ):
        raise ValueError("invalid TCP port search range")
    for port in range(start_port, 65537 - consecutive):
        if all(_is_port_available(port + i) for i in range(consecutive)):
            return port
    raise RuntimeError(f"no available TCP port range starting at or above {start_port}")


def visible_gpu_ids() -> list[int]:
    """Physical IDs in CUDA order, shared by independent role launchers."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        import torch

        return list(range(torch.cuda.device_count()))
    ids = [int(value) for value in visible.split(",") if value.strip()]
    if any(value < 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct nonnegative GPU IDs")
    return ids


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
    )


def get_default_eagle3_aux_layer_ids(
    model_path: str, trust_remote_code: bool = False
) -> List[int]:
    """Get default auxiliary hidden state layer IDs for EAGLE3.

    Args:
        model_path: Path to the HuggingFace model checkpoint.
        trust_remote_code: Whether Transformers may execute repository code.

    Returns:
        List of 3 layer IDs: [1, num_layers // 2 - 1, num_layers - 4]
    """
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    config = getattr(config, "text_config", config)
    num_layers = config.num_hidden_layers
    return [1, num_layers // 2 - 1, num_layers - 4]
