# Path A: read-only PEFT LoRA artifact inspection (source candidate)

This module inspects **existing local** `adapter_config.json` and
`adapter_model.safetensors` without loading model parameters or importing any
PyTorch/PEFT runtime. It is intended to make the next TRAININGDAY host probe
concrete while leaving host authority and real model eligibility unchanged.

```python
from neurosync.peft_preflight import inspect_peft_lora_artifacts

proposal = inspect_peft_lora_artifacts("/path/to/authorized/adapter")
assert proposal["candidate_only"] and not proposal["trainingday_ready"]
```

The checked footprint includes exact local file SHA-256 digests; bounded
safetensors header and contiguous data byte ranges; no duplicate JSON keys;
supported BF16/F16/F32/F64 two-dimensional LoRA A/B matrices; pairwise dtype,
rank and module target agreement; integer rank, alpha and ordinary LoRA
configuration; and an explicit immutable-looking base revision string.
The tensor-name grammar also forbids empty module path segments. The config
preflight rejects unmodeled PEFT initializers (such as OLoRA/PiSSA), bias and
weight-tying extensions, invalid dropout values, alternate target selectors,
and additional architecture/adapter extensions. These options may be valid
in PEFT itself; their interpretation is **not established by this narrow
inspector**. OLoRA in particular can change base weights at initialization,
so treating its output as an ordinary frozen-base LoRA delta would be unsafe.

## Bounded filesystem observation

The revised inspector requires a **narrow, absolute POSIX directory** and
`O_NOFOLLOW` directory-fd traversal through every ancestor. It opens both
files as regular, single-link file descriptors (rejecting symlinks, hardlink
aliases, FIFOs, devices, and directories) and enforces **1 MiB config** and
**4 GiB adapter** size ceilings. Parsing and SHA-256 hashing operate on the
*same held descriptors*. Before returning it verifies descriptor
size/device/inode/mtime/ctime, reopens each fixed filename, and reopens the
declared directory to reject observable pathname/allowed-root replacement.

On Windows, unsupported POSIX handle primitives cause an explicit HOLD; this
code does not claim equivalent Windows no-follow semantics. The checked
pathnames are valid **only at inspection time**: a later change remains
possible without separate host-owned custody, leases and signed receipts.

Regression tests use real synthetic temporary files and cover a symlinked
ancestor, a hardlinked config, a config swapped for an external symlink after
the initial open, and relocation/replacement of the root during scanning.

**It is intentionally restrictive.** This first version does not support
`adapter_model.bin`/pickle, embedding adapters, DoRA, rank/alpha patterns,
transposed `fan_in_fan_out`, other PEFT algorithms, mixed tensor dtypes,
additional trainable modules, or uncommon safetensors tensor formats. It
rejects unusual layouts instead of guessing that similar tensor names imply
compatible numerical meaning. Supported metadata is checked, not executed.

## What it cannot certify

The adapter config's base identifier and revision are **self-reported**, even
when they look like an immutable Git SHA. The parser does not establish that
actual recipient Qwen base weights, tokenizer, config, target layer layouts,
loader versions, license/rights, or optimizer state match. A successful local
SHA-256 inspection doesn't independently witness custody or prove the files
cannot be modified after inspection. It does not conduct model training,
PEFT import, LoRA merge, GGUF conversion, cold restart or semantic evaluation.

The output is labeled `LOCAL_FILE_INSPECTION_SELF_REPORTED`, and
`trainingday_ready` is **always false**. Its digest records are candidates for
host-authorized proof: an independent host must rehash the exact files and
base/tokenizer artifact tree, validate full adapter/architecture compatibility
with the actual loader, allocate an isolated resource lease, preserve a fresh
recipient artifact, cold-load it without the donor, then score sealed behavioral
and negative-transfer controls. `model-training` admission and authority remain
separate. This routine must never be used to flip TRAININGDAY's HOLD to READY.

## Source references

- [Safetensors official format](https://github.com/huggingface/safetensors/blob/main/README.md#format): 8-byte header length, JSON tensor map, data offsets, no holes or duplicate keys.
- [Hugging Face PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint): `adapter_config.json`, adapter weights, LoRA A/B factor naming, and the need for the original base model.
- [Hugging Face PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora): initializer behavior, bias and targeting extensions, and OLoRA base-weight mutation.

Tests use **synthetic headers and zero bytes only**, never real model weights.
