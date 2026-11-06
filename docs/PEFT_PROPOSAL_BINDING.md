# Path A → TRAININGDAY: untrusted PEFT artifact/intent binding

`neurosync.peft_proposal_binding.bind_peft_inspection_to_proposal()` is a
**source-only cross-consistency check** joining three independently supplied
values:

1. The proposed `NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1` inspection from
   Neuro-Sync Draft #38 (actual local file digests and tensor map, but no
   independent physical authority).
2. The raw `NEUROSYNC_TRANSFER_PROPOSAL_V1` intent and corresponding
   `NEUROSYNC_TRAININGDAY_TRANSFER_ENVELOPE_V1` from Draft #37.
3. An explicit `NEUROSYNC_PEFT_ARTIFACT_BINDING_CLAIM_V1` listing **distinct**
   `donor_trained_checkpoint_sha256`, `adapter_weights_sha256`,
   `adapter_config_sha256`, `target_map_sha256` and `rights_record_sha256`.

The adapter is **not** the trained full checkpoint. The existing proposal
uses `input_artifacts.donor_trained` for that full checkpoint and does not
have a separate adapter-weights role. The supplementary claim fills that
provenance gap without changing Draft #37's schema or lying about what was
hashed. It is deliberately not admissible as a replacement for that role.

`target_map_sha256` is SHA-256 of canonical ASCII JSON with sorted keys and
compact separators over `{"rank", "lora_alpha", "target_modules",
"lora_pairs"}`. `target_modules` are sorted; `lora_pairs` are sorted by
`module` and include **rank, dtype, input/output dimensions and target**.
Consequently identical module labels with different tensor shapes are
different proposed target maps. The validator checks exact donor/recipient
base, tokenizer and model-config SHA equality, candidate adapter/config
digests, proposal/intent digest equality, split roles, rights-reference
consistency, $0 external budget and donor-disconnected evaluation controls.

The output is `NEUROSYNC_PEFT_PROPOSAL_BINDING_V1` with
`SELF_REPORTED_CROSS_CONSISTENCY_ONLY`, `physical_launch_authority=HOLD`,
`trainingday_ready=false`. These values cannot change through any successful
input. Even identical verified-bytes hashes would not establish the donor's
learned behavior, generalization or an authenticated recipient load.

## Boundaries and next physical checks

- All four input objects remain caller-controlled and forgeable; their hashes
  are correlation keys, not independently witnessed measurements.
- A real host must independently rehash the adapter weights/config **and**
  full checkpoint, base model, tokenizer and architecture before any effect.
- Loader-specific tensor coordinate mapping, LoRA scale/quantization behavior,
  optimizer lineage, usage rights, protected evaluator custody, host capacity,
  donor-disconnected restart and GGUF serving must be verified elsewhere.
- These source modules must never directly authorize GPU workloads or change
  TRAININGDAY's existing physical HOLD.

The tests use fabricated SHA strings and synthetic tensor maps; no Qwen weights
or user-owned host files were read or modified.
