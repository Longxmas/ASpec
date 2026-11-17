# P-EAGLE model and loss core

This is the first prerequisite for [ASpec #16](https://github.com/inclusionAI/ASpec/issues/16).
It provides `aspec.models.peagle.PEagleDraftModel` and `peagle_kl_loss` for one
packed row of Qwen3 features, full vocabulary, full attention, zero attention dropout,
and eager or SDPA execution. It is a model/loss API; the ASpec training CLI does not
select this algorithm yet.

## Reference and selected recipe

The implementation is **adapted** from
[Speculators `3702320434bbdaa853f0846a66e6b6b8f1d97aec`](https://github.com/vllm-project/speculators/tree/3702320434bbdaa853f0846a66e6b6b8f1d97aec/src/speculators/models/peagle)
and its EAGLE first-layer fusion. The selected reference is the
[Qwen3-8B example](https://github.com/vllm-project/speculators/blob/3702320434bbdaa853f0846a66e6b6b8f1d97aec/examples/train/peagle_qwen3_8b_ultrachat_online_5k.sh):
four Draft layers, four prediction depths, no normalization before the first residual,
no pre-FC normalization and no per-feature FC normalization. The example's online
training, optimizer schedule and COD sampling policy are deferred. Speculators
deprecates P-EAGLE at this revision; this API does not depend on its current main branch.
The applicable license is distributed in `licenses/Speculators-LICENSE`.

The pinned TorchSpec EAGLE3 implementation was reviewed. ASpec retains its recurrent
Llama-style EAGLE3 path. Reusing that trainer would freeze token embeddings and feed
recurrent hidden states; its attention also omits Qwen3 Q/K normalization. This core
instead adapts the applicable Speculators first-layer fusion and reuses Transformers'
Qwen3 attention, MLP, norms, rotary embeddings and subsequent decoder layers.

## Inputs and prediction semantics

Before packing, shift **each document separately**, as in the reference's
`models/eagle3/data.py::shift_batch`:

| Aligned index `i` | Value |
| --- | --- |
| `input_ids[:, i]` | Original token `x[i+1]` |
| `hidden_states[:, i]` | Concatenated selected auxiliary Target features `h[i]` |
| `teacher_logits[:, i]` | Frozen Target logits at original index `i+1` |
| `loss_mask[:, i]` | Reference loss mask at original index `i+1` |
| `position_ids[:, i]` | Document-local position of original token `x[i+1]` |

Target logits at original index `j` predict token `x[j+1]`. Thus a slot
`(anchor=a, depth=d)` predicts `x[a+d+2]` and distills
`teacher_logits[:, a+d]`. Loss masks follow this explicit reference alignment;
callers must adapt other mask conventions before calling this API. Globally
shifting an already packed row would create invalid cross-document pairs.

Shapes are `[1,S]` for token/document/position IDs, `[1,S,A*H]` for auxiliary
features, and `[1,S,V]` for Target logits. The caller supplies the ordered `A`
Target feature layers and matching Qwen3 hidden width `H`; `V` is the full
Target vocabulary. This core does not load, normalize or bind a real Target checkpoint.
Features must be finite and in the normalization convention of the selected reference.
Target auxiliary features and loss targets are detached.

`document_ids=-1` marks padding. Every contiguous document segment is isolated, even
if a document label is reused later. Omitted position IDs start at one and reset
at each segment. Supply explicit IDs when retaining a document-local position offset.

Depth zero includes every position and uses the real token and Target auxiliary
features. Greater depths use a **trainable token embedding** at `mask_token_id`
and a shared **trainable `mask_hidden`** vector in auxiliary-feature space.
One FC projection and the Draft weights are shared across depths. All embedding
rows, FC/attention/MLP/norm/head parameters and `mask_hidden` are trainable; no
frozen Target module is owned by the Draft. The mask embedding remains trainable
even when its ID equals `config.pad_token_id`.

All slots pass through each Draft layer once. A slot attends to preceding base
context in its document and to earlier/equal depths in its own anchor chain.
It cannot read future real tokens or another chain's non-base slots. Padding
attends only to itself, keeping attention rows finite.

The deterministic layout includes every in-document anchor/depth pair. This is an
**unsampled numerical core**, not the reference's stochastic COD sampler. Dense
attention requires quadratic memory in the total number of slots; use short inputs
for this API. This choice makes the model/loss semantics independently reviewable.

## Loss and observations

`peagle_kl_loss(output, teacher_logits, loss_mask)` computes forward
`KL(Target || Draft)` in FP32, with the pinned reference's reduction
`sum(selected_KL) / (selected_count + 1e-5)`. It gathers targets at
`output.anchor_pos + output.depth`. Padding and unsupervised positions are
excluded before softmax; empty supervision returns a differentiable zero.

The returned `kl_sum`, `count` and `correct` vectors have one entry per requested
depth. Aggregate sums/counts before calculating means. These describe independent
parallel predictions; they do not measure a jointly accepted decoding prefix.

## Minimal numerical use

Inside the [isolated Linux environment](installation.md), this example creates
synthetic features and randomly initialized small Qwen3 weights:

```python
import torch
from transformers import Qwen3Config
from aspec.models.peagle import PEagleDraftModel, peagle_kl_loss

config = Qwen3Config(
    vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    head_dim=8, attention_dropout=0.0,
)
model = PEagleDraftModel(config, mask_token_id=0, num_depths=4)
output = model(
    input_ids=torch.randint(0, 32, (1, 6)),
    hidden_states=torch.randn(1, 6, 48),
    document_ids=torch.zeros(1, 6, dtype=torch.long),
)
loss, observations = peagle_kl_loss(
    output, torch.randn(1, 6, 32), torch.ones(1, 6),
)
loss.backward()
```

Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/cpu/test_peagle.py`
for independent scalar attention rules, explicit linear-algebra forward/backward
comparisons, target-index alignment, trainable masks, document/future isolation,
empty supervision, BF16 CPU arithmetic and strict state-dict completeness.

## Remaining work

COD sampling and its RNG/resume semantics, the public trainer and configuration,
Target feature/head bindings, exact training resume, asset export/strict asset loading,
proposal consumers, GPU numerical validation and per-depth held-out/acceptance
experiments remain pending. A strict PyTorch state-dict check is not an ASpec export
or exact-resume guarantee. Synthetic checks do not establish Qwen3-8B training
quality, GPU throughput, consumer compatibility or decoding acceptance.
