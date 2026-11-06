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

Tests use **synthetic headers and zero bytes only**, never real model weights.
