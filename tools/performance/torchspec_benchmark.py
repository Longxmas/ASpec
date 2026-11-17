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

"""Real-corpus DSpark benchmark: prepare data/features, then time native trainers.

Target answers are never regenerated. The default budget is 512 anchors per
physical row; ASpec also supports an explicit per-source-sample budget.
Both sides keep native sampling; actual supervision is reported
separately so packing capacity is not presented as equal supervised work.
No loss, sampler, backward or optimizer implementation is replaced.
"""

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import runpy
import socket
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_revision(module_file):
    result = subprocess.run(
        ["git", "-C", str(Path(module_file).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def configure_native_environment(namespace):
    """Apply actor defaults before package __init__ can initialize PyTorch."""
    spec = importlib.util.find_spec(namespace)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(namespace)
    path = Path(spec.origin).parent / "utils" / "env.py"
    module = runpy.run_path(str(path))
    os.environ.update(module[f"get_{namespace}_env_vars"]())
    return {
        "source": str(path),
        "sha256": sha256(path),
        # Native forwarding can include credentials: never record all values.
        "cpu_thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def observe_cudnn_attention():
    """Observe the real adapter without replacing its math or kernel choice."""
    from aspec.models.ops import cudnn_flex_attention as adapter

    original = adapter.cudnn_flex_attention
    observation = {"forward_calls": 0, "backward_calls": 0, "qkv_format": None}

    def observed(query, key, value, **kwargs):
        result = original(query, key, value, **kwargs)
        observation["forward_calls"] += 1
        qkv_format = "thd" if query.ndim == 3 else "bshd"
        previous_format = observation["qkv_format"]
        if previous_format not in (None, qkv_format):
            raise RuntimeError(
                "benchmark observed mixed cuDNN attention layouts: "
                f"{previous_format} and {qkv_format}"
            )
        observation["qkv_format"] = qkv_format

        def backward(gradient):
            observation["backward_calls"] += 1
            return gradient

        output = result[0] if isinstance(result, tuple) else result
        if output.requires_grad:
            output.register_hook(backward)
        return result

    adapter.cudnn_flex_attention = observed
    return observation


def summary(values):
    values = sorted(values)
    if not values:
        raise ValueError("No eligible records")
    return {
        "count": len(values), "min": values[0], "max": values[-1],
        "mean": sum(values) / len(values),
        **{f"p{p}": values[round((len(values) - 1) * p / 100)] for p in (50, 90, 99)},
    }


def reservoir(source, count, seed):
    """Uniform bounded-memory selection; source order is not length order."""
    rng = random.Random(seed)
    selected = []
    with Path(source).open() as stream:
        for index, line in enumerate(stream):
            if index < count:
                selected.append((index, line))
            else:
                slot = rng.randrange(index + 1)
                if slot < count:
                    selected[slot] = (index, line)
    rng.shuffle(selected)
    return selected


def source_conversation(row):
    for key in ("conversations", "conversation", "messages"):
        if isinstance(row.get(key), list):
            return row[key]
    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        return [{"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]}]
    return []


def length_stratified(records, count):
    """Select deterministic length quantiles, including both endpoints."""

    if count <= 0 or len(records) < count:
        raise ValueError("Length-stratified selection needs enough records")
    ordered = sorted(records, key=lambda row: (int(row["seq_len"]), row["data_id"]))
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("Length-stratified selection produced duplicate quantiles")
    return [ordered[index] for index in indices]


def prepare_data(args):
    from transformers import AutoTokenizer

    from aspec.data.parse import create_parser
    from aspec.data.template import TEMPLATE_REGISTRY
    from aspec.data.utils import pack_loss_mask, serialize_packed_loss_mask
    from tools.convert_pretokenized_jsonl import convert
    from tools.prepare_quality_dataset import normalize_conversation

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    parser = create_parser(tokenizer, TEMPLATE_REGISTRY.get("qwen3-instruct"))
    lengths, supervision = [], []
    filtered = {"format": 0, "length": 0, "supervision": 0}
    raw_path = args.output / "documents.jsonl"
    with raw_path.open("w") as destination:
        for index, line in reservoir(args.source, args.scan, args.seed):
            row = json.loads(line)
            messages = normalize_conversation(source_conversation(row))
            if not messages or any(
                not isinstance(m.get("content"), str)
                or m.get("role") not in {"system", "user", "assistant"} for m in messages
            ):
                filtered["format"] += 1
                continue
            formatted = parser.format(messages, add_generation_prompt=False)
            # Do not truncate away an answer or transform a long document into
            # an artificial short example. Both lanes use whole conversations.
            ids, mask = parser.parse(formatted, args.length + 1, preformatted=True,
                                     last_turn_only=False)
            n, supervised = ids.numel(), int(mask.sum())
            if not args.min_length <= n <= args.length:
                filtered["length"] += 1
                continue
            if supervised < 32:
                filtered["supervision"] += 1
                continue
            item = {
                "data_id": str(row.get("id") or f"source-row-{index}"),
                "input_ids": ids.tolist(),
                "packed_loss_mask": serialize_packed_loss_mask(pack_loss_mask(mask)),
                "seq_len": n, "loss_tokens": supervised, "source_row_index": index,
            }
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            lengths.append(n)
            supervision.append(supervised)
            if len(lengths) == args.rows:
                break
    if len(lengths) < args.rows:
        raise ValueError(f"Only {len(lengths)} eligible rows; inspect the source/length distribution")
    convert(raw_path, args.output / "documents.parquet")
    manifest = {
        "schema": "aspec.torchspec-benchmark-data/v1",
        "source": str(args.source.resolve()), "source_sha256": sha256(args.source),
        "answer_regeneration": False, "whole_conversations": True,
        "template": "qwen3-instruct", "last_turn_loss_only": False,
        "target": args.target, "seed": args.seed, "scanned_pool": args.scan,
        "max_length": args.length, "min_length": args.min_length,
        "lengths": summary(lengths), "supervised_tokens": summary(supervision),
        "filtered": filtered, "jsonl_sha256": sha256(raw_path),
        "parquet_sha256": sha256(args.output / "documents.parquet"),
        "tool_sha256": sha256(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def feature_target_config(config, maximum_length):
    """Use the agreed 8B YaRN lane, not an 8B override on every long Target."""
    text = getattr(config, "text_config", config)
    if text.model_type == "qwen3" and maximum_length > 32768:
        if maximum_length > 131072:
            raise ValueError("The Qwen3-8B benchmark lane is limited to 128K with YaRN")
        text.rope_parameters = {
            **text.rope_parameters, "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        text.max_position_embeddings = 131072
    elif maximum_length > text.max_position_embeddings:
        raise ValueError("Feature documents exceed the Target native context")
    return config


def feature_backbone(target):
    backbone = getattr(target, "language_model", target)
    if not hasattr(backbone, "layers"):
        backbone = getattr(backbone, "model", backbone)
    if not hasattr(backbone, "layers"):
        raise ValueError("Target does not expose its text decoder layers")
    return backbone


def post_layer_capture(captures, index):
    def capture(module, positional, output):
        value = output[0] if isinstance(output, tuple) else output
        captures[index] = value.detach().cpu()
    return capture


def prepare_features(args):
    import gc

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoConfig, AutoModel

    from aspec.data.utils import unpack_loss_mask

    args.output.mkdir(parents=True, exist_ok=False)
    available = pq.read_table(args.data).to_pylist()
    records = (
        length_stratified(available, args.rows)
        if args.feature_selection == "length-stratified"
        else available[:args.rows]
    )
    if len(records) != args.rows:
        raise ValueError("Not enough prepared documents")
    draft = json.loads(args.draft_config.read_text())
    config = feature_target_config(
        AutoConfig.from_pretrained(args.target), max(r["seq_len"] for r in records)
    )
    target = AutoModel.from_pretrained(args.target, config=config, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda").eval()
    backbone = feature_backbone(target)
    captures = {}
    # SGLang's set_eagle3_layers_to_capture/set_dflash_layers_to_capture maps
    # logical ID i to the boundary before i+1: the output of decoder layer i.
    handles = [backbone.layers[i].register_forward_hook(post_layer_capture(captures, i))
               for i in draft["target_layer_ids"]]
    documents = []
    try:
        for index, row in enumerate(records):
            ids = torch.tensor(row["input_ids"], dtype=torch.long)[None]
            mask = unpack_loss_mask(row["packed_loss_mask"]).float()[None]
            captures.clear()
            with torch.inference_mode():
                last = backbone(input_ids=ids.cuda(), use_cache=False).last_hidden_state.cpu()
            hidden = torch.cat([captures[i] for i in draft["target_layer_ids"]], dim=-1)
            path = args.output / f"document-{index:04d}.pt"
            torch.save({"input_ids": ids, "loss_mask": mask,
                        "hidden_states": hidden, "last_hidden_states": last}, path)
            documents.append({"path": path.name, "data_id": row["data_id"],
                              "seq_len": row["seq_len"], "loss_tokens": row["loss_tokens"],
                              "sha256": sha256(path)})
            print(json.dumps({"features": index + 1, "length": row["seq_len"]}), flush=True)
            del hidden, last
    finally:
        for handle in handles:
            handle.remove()
    captures.clear()
    del backbone, target
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {
        "schema": "aspec.torchspec-benchmark-features/v1",
        "data": str(args.data.resolve()), "data_sha256": sha256(args.data),
        "draft_config": draft, "target": args.target, "target_config": config.to_dict(),
        "documents": documents, "last_hidden_states_prenorm": False,
        "feature_source": "Real frozen Target forward on original corpus, not generated answers",
        "feature_selection": args.feature_selection,
        "auxiliary_capture_stage": "logical decoder layer output (SGLang logical ID + 1 boundary)",
        "tool_sha256": sha256(__file__),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class ResidentDocumentFeed:
    """The iterator interface is shared by both native trainers."""

    def __init__(self, documents, records, collator, *, with_identity, documents_per_batch=1):
        self.documents, self.records, self.collator = documents, records, collator
        if not documents or len(documents) != len(records):
            raise ValueError("Resident documents and identities must be nonempty and aligned")
        self.with_identity = with_identity
        self.documents_per_batch = documents_per_batch
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        items = []
        input_tokens = 0
        for _ in range(self.documents_per_batch):
            index = self.cursor % len(self.documents)
            self.cursor += 1
            input_tokens += int(self.records[index]["seq_len"])
            item = dict(self.documents[index])
            if self.with_identity:
                item["_aspec_work"] = {
                    "work_id": self.records[index]["data_id"],
                    "execution_id": f"resident-{self.cursor}",
                }
            items.append(item)
        batch = self.collator(items)
        batch["_aspec_benchmark_documents"] = len(items)
        batch["_aspec_benchmark_input_tokens"] = input_tokens
        return batch

    def iter_step_batches(self, count):
        for _ in range(count):
            yield next(self)


def natural_packing_schedule(records, *, pool_size, max_seq_length, batches):
    """Use the production row packer with manifest-level assistant-token density.

    This resident-feature control excludes scheduling time. Its manifest has
    raw loss-token counts, not the live controller's block-eligible anchor
    counts; the end-to-end lane uses the unmodified live controller.
    """

    from aspec.controller.packing import pack_work_aware_rows

    if not records or min(pool_size, max_seq_length, batches) < 1:
        raise ValueError("Natural packing needs records and positive capacity")
    if len(records) < pool_size:
        raise ValueError("Natural packing source must contain at least one complete pool")
    lengths = [int(record["seq_len"]) for record in records]
    useful = [int(record["loss_tokens"]) for record in records]
    if any(length <= 0 or length > max_seq_length for length in lengths):
        raise ValueError("Every natural-packing document must fit the physical row")

    pending = list(range(pool_size))
    source_cursor = pool_size
    schedule = []
    for _ in range(batches):
        packed = pack_work_aware_rows(
            [lengths[index] for index in pending],
            [useful[index] for index in pending],
            dp_size=1,
            max_seq_length=max_seq_length,
        )
        if packed is None:
            raise RuntimeError("Natural packing could not form a physical row")
        selected = set(packed.selected_indices)
        schedule.append(tuple(pending[index] for index in packed.rank_rows[0]))
        pending = [index for offset, index in enumerate(pending) if offset not in selected]
        while len(pending) < pool_size:
            pending.append(source_cursor % len(records))
            source_cursor += 1
    return schedule


class ScheduledDocumentFeed:
    """Feed deterministic packed rows without timing scheduler Python work."""

    def __init__(self, documents, records, collator, schedule, *, with_identity):
        self.documents, self.records, self.collator = documents, records, collator
        self.schedule = schedule
        self.with_identity = with_identity
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        indices = self.schedule[self.cursor % len(self.schedule)]
        self.cursor += 1
        items = []
        for position, index in enumerate(indices):
            item = dict(self.documents[index])
            if self.with_identity:
                item["_aspec_work"] = {
                    "work_id": self.records[index]["data_id"],
                    "execution_id": f"resident-{self.cursor}-{position}-{index}",
                }
            items.append(item)
        batch = self.collator(items)
        # The native ASpec trainer consumes ``_aspec_work`` before
        # calling ``_train_step``.  Preserve the benchmark-only cardinality
        # separately so a packed THD row is not mistaken for one document.
        batch["_aspec_benchmark_documents"] = len(items)
        batch["_aspec_benchmark_input_tokens"] = sum(int(self.records[index]["seq_len"]) for index in indices)
        return batch

    def iter_step_batches(self, count):
        for _ in range(count):
            yield next(self)


def observed_batch_work(batch):
    documents = batch.pop("_aspec_benchmark_documents")
    input_tokens = batch.pop("_aspec_benchmark_input_tokens")
    if min(documents, input_tokens) < 1:
        raise ValueError("Resident work counters must be positive")
    return {"documents": documents, "input_tokens": input_tokens,
            "physical_tokens": int(batch["input_ids"].numel())}


def resident_rank_records(records, documents_per_step, world_size, rank):
    """Keep fixed global update sets, with stable strided DP assignment.

    This resident diagnostic does not emulate the live asynchronous dispatcher.
    MBS only splits each rank's fixed stream; it never changes its documents.
    """
    if world_size < 1 or not 0 <= rank < world_size or documents_per_step % world_size:
        raise ValueError("Global document batch must divide over valid DP ranks")
    if len(records) % documents_per_step:
        raise ValueError("Resident source must contain complete global update sets")
    return [record for start in range(0, len(records), documents_per_step)
            for record in records[start:start + documents_per_step][rank::world_size]]


def aggregate_resident_results(rank_payloads):
    """Aggregate measured work only; warmup batches stay in raw rank evidence."""
    payload = dict(rank_payloads[0])
    payload["rank_results"] = rank_payloads
    payload["wall_seconds"] = max(row["wall_seconds"] for row in rank_payloads)
    for key in ("committed_documents", "input_tokens", "physical_tokens",
                "supervised_predictions", "anchor_slots", "query_anchor_slots",
                "configured_window_tokens"):
        payload[key] = sum(row[key] for row in rank_payloads)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        payload[key] = max(row[key] for row in rank_payloads)
    payload["padding_efficiency"] = payload["input_tokens"] / payload["physical_tokens"]
    payload["input_fill_ratio"] = payload["padding_efficiency"]
    payload["configured_window_fill_ratio"] = payload["input_tokens"] / payload["configured_window_tokens"]
    measured_batches = sum(len(row["batch_work"]) - row["warmup"] * row["accumulation_steps"]
                           for row in rank_payloads)
    payload["documents_per_physical_batch"] = payload["committed_documents"] / measured_batches
    for metric, counter in (("input_tokens_per_second", "input_tokens"),
                            ("supervised_predictions_per_second", "supervised_predictions"),
                            ("documents_per_second", "committed_documents")):
        payload[metric] = payload[counter] / payload["wall_seconds"]
    return payload


def run_native(args):
    namespace = args.framework
    # Preserve native defaults without importing package __init__ first.
    native_environment = configure_native_environment(namespace)
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        if namespace != "aspec" or args.layout_mode != "non-packing":
            raise ValueError("Distributed resident controls currently require ASpec non-packing")
        if not args.initial.is_file():
            raise ValueError("Distributed controls require an existing shared initial state")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    cudnn_observation = observe_cudnn_attention() if namespace == "aspec" else None
    if args.result.exists():
        raise FileExistsError(args.result)
    manifest = json.loads((args.features / "manifest.json").read_text())
    records = manifest["documents"]
    natural_packing = args.layout_mode == "natural-packing"
    packed_rows = namespace == "aspec" and args.layout_mode != "non-packing"
    if not natural_packing and len(records) % args.documents_per_step:
        raise ValueError("Resident document count must form complete logical batches")
    if not natural_packing and args.documents_per_step % (world_size * args.micro_batch_size):
        raise ValueError("Logical batch must be divisible by the physical document batch")
    ordering = getattr(args, "document_ordering", "native")
    if natural_packing and ordering != "native":
        raise ValueError("Explicit document ordering applies to non-natural-packing controls")
    length_grouped = ordering == "length-grouped" or (ordering == "native" and (
        namespace == "torchspec" or (namespace == "aspec" and not natural_packing and args.micro_batch_size > 1)
    ))
    if length_grouped:
        length_grouped_order = importlib.import_module(namespace + ".data.utils").length_grouped_order

        # Preserve TorchSpec's default local length grouping for the same
        # document stream on both sides; do not manufacture unsorted padding.
        records = length_grouped_order(records, group_size=32)
    global_document_order = [r["data_id"] for r in records]
    if world_size > 1:
        records = resident_rank_records(records, args.documents_per_step, world_size, rank)
    batches_per_step = (
        args.accumulation_steps
        if natural_packing
        else args.documents_per_step // (world_size * args.micro_batch_size)
    )
    resident = []
    for record in records:
        path = args.features / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Feature identity changed: {path}")
        resident.append({k: v.cuda() for k, v in torch.load(path, mmap=True, weights_only=True).items()})
    cfg_module = importlib.import_module(namespace + ".config.train_config")
    config = {
        **dataclasses.asdict(cfg_module.TrainingConfig()),
        **dataclasses.asdict(cfg_module.DebugConfig()),
    }
    config.update(
        target_model_path=manifest["target"], chat_template=None,
        embedding_key=args.embedding_key,
        attention_backend=args.attention_backend if namespace == "aspec" else "flex_attention",
        draft_context_parallel_size=1, fsdp_strategy="REPLICATE",
        dflash_packing=packed_rows,
        **({"anchor_sampling_scope": args.anchor_sampling_scope} if namespace == "aspec" else {}),
        micro_batch_size=1 if packed_rows else args.micro_batch_size,
        draft_accumulation_steps=batches_per_step,
        dspark_num_anchors=512, dflash_block_size=7, dspark_num_target_layers=5,
        max_seq_length=(
            args.max_sequence_length
            if natural_packing
            else max(r["seq_len"] for r in records) * (args.micro_batch_size if packed_rows else 1)
        ),
        seed=42,
        learning_rate=6e-4, min_lr=0.0, weight_decay=0.0, max_grad_norm=1.0,
        warmup_ratio=0.04, lr_total_steps=1000, last_hidden_states_prenorm=False,
        compile_model=False, load_path=None, resume=None, checkpoint_dir=None,
        enable_perf_metrics=True, num_train_steps=args.warmup + args.steps,
    )
    if namespace == "aspec":
        config["dspark_lm_head_precision"] = args.dspark_lm_head_precision
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.initial.parent.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    else:
        dist.init_process_group("nccl", init_method="file://" + str(args.result.with_suffix(".rdzv").resolve()),
                                rank=0, world_size=1)
    importlib.import_module(namespace + ".utils.distributed").init_gloo_group()
    init_module = importlib.import_module(namespace + ".training.dflash_trainer")
    dspark_module = importlib.import_module(namespace + ".training.dspark_trainer")
    load_original = init_module.fsdp2_load_full_state_dict
    dspark_load_original = getattr(dspark_module, "fsdp2_load_full_state_dict", None)
    initial_load_calls = 0

    def shared_initial(model, state, *positional, **kwargs):
        nonlocal initial_load_calls
        if args.initial.exists():
            state = torch.load(args.initial, mmap=True, weights_only=True)
        else:
            torch.save(state, args.initial)
        initial_load_calls += 1
        return load_original(model, state, *positional, **kwargs)

    init_module.fsdp2_load_full_state_dict = shared_initial
    if dspark_load_original is not None:
        dspark_module.fsdp2_load_full_state_dict = shared_initial
    trainer = dspark_module.DSparkTrainer(argparse.Namespace(**config))
    try:
        trainer.init_model(manifest["draft_config"], manifest["target"])
    finally:
        init_module.fsdp2_load_full_state_dict = load_original
        if dspark_load_original is not None:
            dspark_module.fsdp2_load_full_state_dict = dspark_load_original
    if initial_load_calls != 1:
        raise RuntimeError(f"Expected one native Draft initial-state load, observed {initial_load_calls}")
    collator_module = importlib.import_module(namespace + ".data.utils")
    collator = (collator_module.DFlashPackingCollator(config["max_seq_length"], pad_to_max_length=False)
                if packed_rows else collator_module.DataCollatorWithPadding())
    observed = []
    batch_work = []
    original_step = trainer._train_step

    def observe(*positional, **kwargs):
        batch = positional[0] if positional else kwargs["batch"]
        # The validated feature manifest supplies logical work. Do not launch
        # a GPU reduction + .item() synchronization just to count every batch.
        work = observed_batch_work(batch)
        work["anchor_slots"] = (
            work["documents"] * 512 if args.anchor_sampling_scope == "sample"
            else int(batch["input_ids"].shape[0]) * 512
        )
        if packed_rows and args.anchor_sampling_scope == "sample":
            from aspec.models.ops.anchors import document_anchor_slots

            work["query_anchor_slots"] = document_anchor_slots(
                batch["packed_document_lengths"], 512, 7
            )[2]
        else:
            work["query_anchor_slots"] = int(batch["input_ids"].shape[0]) * 512
        metrics = original_step(*positional, **kwargs)
        work["loss_head_chunks"] = getattr(trainer.dflash, "loss_head_chunks_observed", 1)
        work["loss_head_blocks"] = getattr(trainer.dflash, "loss_head_blocks_observed", None)
        work["backbone_blocks"] = getattr(trainer.dflash, "backbone_blocks_observed", None)
        work["cached_teacher_blocks"] = getattr(
            trainer.dflash, "loss_head_cached_teacher_blocks_observed", None
        )
        work["cached_draft_blocks"] = getattr(
            trainer.dflash, "loss_head_cached_draft_blocks_observed", None
        )
        work["cached_context_tokens"] = getattr(
            trainer.dflash, "loss_head_cached_context_tokens_observed", None
        )
        observed.append(metrics["count_per_position"].detach())
        batch_work.append(work)
        return metrics

    trainer._train_step = observe
    if natural_packing and namespace == "aspec":
        schedule = natural_packing_schedule(
            records,
            pool_size=args.packing_pool_size,
            max_seq_length=args.max_sequence_length,
            batches=(args.warmup + args.steps) * batches_per_step,
        )
        trainer.data_fetcher = ScheduledDocumentFeed(
            resident, records, collator, schedule, with_identity=True
        )
    else:
        trainer.data_fetcher = ResidentDocumentFeed(
            resident, records, collator, with_identity=namespace == "aspec",
            documents_per_batch=args.micro_batch_size)
    steps = []
    started = None
    try:
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                if world_size > 1:
                    dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
            begin = time.perf_counter()
            metrics = trainer.train_from_queue(step=step, num_batches=batches_per_step)
            steps.append({"step": step + 1, "measured": step >= args.warmup,
                          "wall_seconds": time.perf_counter() - begin, "metrics": metrics})
            print(json.dumps({"framework": namespace, "step": step + 1,
                              "wall": steps[-1]["wall_seconds"], "loss": metrics.get("train/avg_loss")}),
                  flush=True)
        final_metrics = trainer.flush_pending_metrics() if hasattr(trainer, "flush_pending_metrics") else {}
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        counts = torch.stack(observed).float().cpu()
        measured_counts = counts[args.warmup * batches_per_step:]
        measured_work = batch_work[args.warmup * batches_per_step:]
        n = sum(row["documents"] for row in measured_work)
        valid_tokens = sum(row["input_tokens"] for row in measured_work)
        physical_tokens = sum(row["physical_tokens"] for row in measured_work)
        configured_window_tokens = (
            len(measured_work) * args.max_sequence_length
            if natural_packing
            else physical_tokens
        )
        finalized = [s["metrics"] for s in steps if s["metrics"]]
        if final_metrics:
            finalized.append(final_metrics)
        if len(finalized) != len(steps) or not all(
            math.isfinite(m.get("train/avg_loss", float("nan")))
            and math.isfinite(m.get("train/grad_norm", float("nan"))) for m in finalized
        ):
            raise ValueError("Missing or nonfinite native loss/gradient")
        observations = [getattr(layer.self_attn, "attention_observation", {})
                        for layer in trainer.draft_model.layers]
        qk_prepare_backends = [
            getattr(layer.self_attn, "qk_prepare_backend", None)
            for layer in trainer.draft_model.layers
        ]
        if namespace == "aspec":
            if args.attention_backend == "fa4" and not all(
                o.get("qkv_format") == "thd" and o.get("backward_observed") for o in observations
            ):
                raise ValueError("THD forward/backward did not execute on every Draft layer")
            if args.attention_backend == "cudnn_flex_attention":
                expected_calls = len(trainer.draft_model.layers) * len(batch_work)
                if any(cudnn_observation[key] != expected_calls for key in ("forward_calls", "backward_calls")):
                    raise ValueError("cuDNN forward/backward did not execute on every Draft layer")
            if args.require_qk_backend and set(qk_prepare_backends) != {args.require_qk_backend}:
                raise ValueError(
                    "Observed Q/K preparation backends do not satisfy "
                    f"--require-qk-backend={args.require_qk_backend}: {qk_prepare_backends}"
                )
        payload = {
            "schema": "aspec.torchspec-draft-control/v1", "framework": namespace,
            "layout": "unpadded token rows (THD)" if packed_rows else "per-document padded batches (BSHD)",
            "document_ordering": (
                "explicit local length grouping" if ordering == "length-grouped" else
                "explicit manifest order" if ordering == "manifest" else
                "native local length grouping" if namespace == "torchspec" else
                "production row packer" if namespace == "aspec" and natural_packing else
                "local length grouping for equal-work control" if namespace == "aspec" and args.micro_batch_size > 1 else
                "feature manifest order"
            ),
            "dp_size": world_size, "dp_rank": rank,
            "global_documents_per_update": args.documents_per_step if not natural_packing else None,
            "global_document_order": global_document_order,
            "packing_multiple_documents": packed_rows and (natural_packing or args.micro_batch_size > 1),
            "original_documents_per_microbatch": (
                None if natural_packing and namespace == "aspec" else args.micro_batch_size
            ),
            "anchor_sampling_scope": args.anchor_sampling_scope if namespace == "aspec" else "sample",
            "anchor_slots_per_physical_sequence": (
                None if packed_rows and args.anchor_sampling_scope == "sample" else 512
            ),
            "anchor_slots_per_source_sample": 512 if args.anchor_sampling_scope == "sample" else None,
            "document_order": [r["data_id"] for r in records], "features_manifest": manifest,
            "wall_seconds": wall, "steps": args.steps, "warmup": args.warmup,
            "layout_mode": args.layout_mode,
            "max_sequence_length": config["max_seq_length"],
            "requested_max_sequence_length": args.max_sequence_length,
            "packing_pool_size": args.packing_pool_size if natural_packing else None,
            "accumulation_steps": batches_per_step,
            "committed_documents": n, "input_tokens": valid_tokens,
            "physical_tokens": physical_tokens,
            "padding_efficiency": valid_tokens / physical_tokens,
            "configured_window_tokens": configured_window_tokens,
            "configured_window_fill_ratio": valid_tokens / configured_window_tokens,
            # Backward-compatible alias retained for earlier evidence readers.
            "input_fill_ratio": valid_tokens / physical_tokens,
            "documents_per_physical_batch": n / len(measured_work),
            "supervised_predictions": float(measured_counts.sum()),
            "anchor_slots": sum(row["anchor_slots"] for row in measured_work),
            "query_anchor_slots": sum(row["query_anchor_slots"] for row in measured_work),
            "input_tokens_per_second": valid_tokens / wall,
            "supervised_predictions_per_second": float(measured_counts.sum()) / wall,
            "documents_per_second": n / wall, "supervision_per_microbatch": counts.tolist(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "native_config": config, "initial_sha256": sha256(args.initial),
            "initial_load_calls": initial_load_calls,
            "attention_observations": observations,
            "qk_prepare_backends": qk_prepare_backends,
            "l1_loss_backend": getattr(trainer.model, "l1_loss_backend", None),
            "l1_kernel_invocations": getattr(trainer.model, "l1_kernel_invocations", None),
            "fused_clip_scaling": getattr(trainer.optimizer, "_fuse_clip_scaling", False),
            "cudnn_observation": cudnn_observation,
            "document_layout": "unpadded_packed_stream" if packed_rows else "padded_documents",
            "tool_sha256": sha256(__file__), "native_source": {},
            "native_environment": native_environment,
            "environment": {"host": socket.gethostname(), "python": sys.executable,
                            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                            "capability": torch.cuda.get_device_capability(),
                            "nvidia_cutlass_dsl": package_version("nvidia-cutlass-dsl"),
                            "quack_kernels": package_version("quack-kernels"),
                            "cpu_threads": torch.get_num_threads(),
                            "framework_root": importlib.import_module(namespace).__file__,
                            "commit": source_revision(importlib.import_module(namespace).__file__)},
            "step_records": steps, "batch_work": batch_work,
            "final_flushed_metrics": final_metrics,
            "scope": "Draft-only native input throughput/capacity; no Target/transport timing. Supervision is reported separately, not an equal-supervision or equal-quality claim.",
        }
        for suffix in ("models.dspark", "models.dflash", "models.draft.dspark", "models.draft.dflash",
                       "training.dspark_trainer", "training.dflash_trainer", "training.optimizer",
                       "training.trainer", "training.fsdp", "data.utils",
                       *(("models.ops.output_blocks", "models.ops.direct_cp") if namespace == "aspec" else ())):
            path = importlib.import_module(namespace + "." + suffix).__file__
            payload["native_source"][path] = sha256(path)
        if namespace == "aspec" and "aspec_cutedsl" in qk_prepare_backends:
            for module_name in (
                "aspec.models.ops.dspark_qk_norm_rope",
                "aspec.kernels.cutedsl._primitives",
                "aspec.kernels.cutedsl.dspark_qk_norm_rope",
                "aspec.kernels.cutedsl.dspark_qk_norm_rope_bwd",
            ):
                path = importlib.import_module(module_name).__file__
                payload["native_source"][path] = sha256(path)
        if namespace == "aspec":
            for suffix in ("models.ops.loss", "models.ops.tiled_l1", "models.ops.gradient_scaling",
                           "models.ops.cudnn_flex_attention"):
                path = Path(importlib.import_module(namespace).__file__).parent / (suffix.replace(".", "/") + ".py")
                if path.is_file():
                    payload["native_source"][str(path)] = sha256(path)
        if world_size > 1:
            # CPU collection occurs after wall/peak capture; no GPU object
            # collective or synthetic per-update barrier enters the window.
            rank_path = args.result.with_name(args.result.stem + f".rank{rank}.json")
            rank_path.write_text(json.dumps(payload, indent=2) + "\n")
            rank_payloads = [None] * world_size
            group = importlib.import_module("aspec.utils.distributed").get_gloo_group()
            dist.all_gather_object(rank_payloads, payload, group=group)
            if rank == 0:
                payload = aggregate_resident_results(rank_payloads)
        if rank == 0:
            args.result.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({k: payload[k] for k in ("framework", "wall_seconds",
                                                "input_tokens_per_second",
                                                "supervised_predictions_per_second")}), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("data", "features", "run"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--scan", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--draft-config", type=Path, default=Path("aspec/config/dspark_draft_config.json"))
    parser.add_argument(
        "--feature-selection",
        choices=("prefix", "length-stratified"),
        default="prefix",
        help="Select the prepared rows used for Target feature materialization.",
    )
    parser.add_argument("--features", type=Path)
    parser.add_argument("--framework", choices=("aspec", "torchspec", "angelspec"))
    parser.add_argument("--dspark-lm-head-precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    parser.add_argument("--attention-backend", choices=("cudnn_flex_attention", "fa4"), default="cudnn_flex_attention",
                        help="ASpec backend; native baselines keep their own FlexAttention backend.")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--documents-per-step", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--document-ordering", choices=("native", "manifest", "length-grouped"),
                        default="native", help="Fix the document stream independently of MBS for batch-size controls.")
    parser.add_argument("--anchor-sampling-scope", choices=("row", "sample"), default="row")
    parser.add_argument(
        "--layout-mode",
        choices=("equal-work", "natural-packing", "non-packing"),
        default="natural-packing",
    )
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--packing-pool-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--require-qk-backend",
        choices=("eager", "aspec_cutedsl"),
        help="Fail unless every ASpec Draft layer observes this Q/K preparation backend.",
    )
    args = parser.parse_args()
    if args.dspark_lm_head_precision == "fp8" and (args.mode != "run" or args.framework != "aspec"):
        parser.error("FP8 LM head requires run --framework aspec")
    required = {
        "data": ("source", "output"), "features": ("data", "output"),
        "run": ("features", "framework", "initial", "result"),
    }
    for name in required[args.mode]:
        if getattr(args, name) is None:
            parser.error(f"{args.mode} requires --{name.replace('_', '-')}")
    if args.rows > args.scan:
        parser.error("--rows must not exceed --scan")
    if min(
        args.rows,
        args.scan,
        args.documents_per_step,
        args.micro_batch_size,
        args.steps,
        args.warmup,
        args.max_sequence_length,
        args.packing_pool_size,
        args.accumulation_steps,
    ) <= 0:
        parser.error("Counts and warmup must be positive")
    if args.mode != "run":
        from aspec.utils.env import get_aspec_env_vars

        for key, value in get_aspec_env_vars().items():
            os.environ.setdefault(key, value)
    {"data": prepare_data, "features": prepare_features, "run": run_native}[args.mode](args)


if __name__ == "__main__":
    main()
