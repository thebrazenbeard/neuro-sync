# Real Qwen 0.8B LoRA — measured local feasibility, not Neuro-Sync novelty

This source-only experimental path is independent of StateBus, TrainingDay,
and the staged PR #57/#61. It **does not merge** incompatible source branches,
alter an installed model, or promote user-private training data.

## Host observation, 2026-11-16

On Lappy RTX 3050 4,096 MiB, in a separately installed user-space WSL2
Python 3.12.14 environment (torch 2.9.1+cu128; transformers 5.14.1;
peft 0.19.1; bitsandbytes 0.49.2; accelerate 1.12.0):
- Official Apache-2.0 Qwen/Qwen3.5-0.8B-Base revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` loaded in NF4
  without modification of the local Ollama installation.
- A real LoRA adapter with 159,744 trainable values across 24 tensors
  underwent **four** CUDA optimizer steps; finite gradients and changed
  parameters were observed. Training target loss 4.578054 → 1.430334,
  peak torch allocation 1,769 MiB.
- Adapter safetensors SHA-256:
  `dc8ac35db01c1b8ab44013a2e24cabe455d142b559b4c293466f75c4be0b06bc`.
  The private model adapter and receipts remain outside Git.
- A **separate** Python process loaded the pinned base without the donor
  model object, verified adapter file digests, and attached its adapter.
  Teacher-forced answer-target losses (unmodified base → adapted):
  taught 4.579558 → 1.430875; paraphrase 4.693440 → 1.889898;
  **untaught different satellite** 4.549791 → 1.824705.

**Interpretation:** A real saved and reloaded adapter changes receiver
predictions after the donor process exits. The untaught-target improvement
is an adversarial counterexample to selective recall: B may simply raise
the taught answer's likelihood regardless of which satellite was asked.
This is one seed, one synthetic training association, one paraphrase,
no independently sealed heldout, no host reboot or separately authenticated
donor custody. It is NOT demonstrated generalization, new learning
mechanism, or a StateBus advantage.

## Exact performed scripts and evidence custody

The one-time private experimental scripts are **not the byte-identical
published programs**: this public version replaces host-hardcoded paths
with explicit required env vars and refuses model artifacts inside Git.
Original experiment source SHA-256 (train / receiver):
- `89ee85595486bfb6b385ead0db62104258f96080c34186a28debb2a1bbdf0265`
- `477f6059b4493c32f3009c1c22c98b00458a7522ce4fa7470542f60e68742791`
Original private train receipt SHA-256:
`de7f76166a826af59931c9bf529c616ab5efd817a98287d3bae03ff05f02fbd3`.
Original receiver receipt SHA-256:
`c9b136c3c9e0a977674131cc490da8f83ea92ed879c1d53007dabf93d59f4b16`.

Self-reported hashes provide reproducible byte references but do not
establish external custody or independent serving-process attestation.
## Reproduce only in an isolated, authorized GPU slot

Use Linux user-space Python with pinned CUDA Torch, PEFT, Transformers and
bitsandbytes packages. Verify GPU free headroom before starting. Supply two
absolute private locations, not paths within this repository:

```bash
export NEUROSYNC_HF_HOME=/home/vera/.cache/neuro-sync-repro-hf
export NEUROSYNC_OUTPUT_DIR=/mnt/d/VERA/.scratch/neuro-sync-repro-unique-run
python experiments/real-qwen-transfer/train_gate.py
# Train exits; receiver is an entirely new Python process:
python experiments/real-qwen-transfer/receiver_gate.py
```

Training refuses a preexisting output directory and caps its CUDA allocator
to 85% of device VRAM, rejecting low free headroom. The receiver validates
the saved adapter file digests before and after loading, checks its base
revision, and records unblinded base/adapter losses. Both require explicit
private storage env vars and reject model artifacts under the Git tree.

These scripts are an experiment, not an unattended daemon: no service
restart, permission escalation, paid cloud compute, new runtime
authority, or credential use is performed by the host readback. The
initial pinned checkpoint fetch contacts the public Hugging Face Hub;
no private user conversation or task data is uploaded.

## Test the evidence classifier without GPUs

```bash
python3 -m unittest discover -s experiments/real-qwen-transfer -p 'test_*.py' -v
```

`receipt_audit.classify_self_reported` validates shared model/digest roles,
numeric loss fields and actual adapter delta, and intentionally never
declares independent generalization or Neuro-Sync novelty. Green source CI
does not rerun the GPU experiment. The next real test requires independent
heldouts, donor-absent receiver cold-start, multiple seeds, shuffled-label
and equal-information conventional-adapter/distillation baselines.
