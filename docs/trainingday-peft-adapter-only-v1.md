# TRAININGDAY Path A: frozen-base PEFT adapter-only candidate

**Status: untrusted source proposal; `trainingday_ready=false`; physical launch
`HOLD`.** This is a *different* method from the changed-full-checkpoint
`exact-base-delta` Path A proposal in Draft PR #37. It does not replace that
method or assert that actual Qwen weights have been inspected or changed.

The [official PEFT checkpoint guide](https://huggingface.co/docs/peft/developer_guides/checkpoint)
states that `adapter_model.safetensors` saves only adapter parameters, not the
base model's parameters. Consequently, the ordinary LoRA training/export path
may legitimately retain identical donor and recipient base-model bytes while
the separate adapter weights change. Requiring a changed **full** donor model
checkpoint in this case would confuse artifact roles and could force a false
training receipt.

`neurosync.peft_adapter_only_proposal.propose_adapter_only_transfer(intent)`
accepts a separately typed `NEUROSYNC_PEFT_ADAPTER_ONLY_INTENT_V1`:

- Unchanged exact declared donor/recipient base weights, tokenizer and model
  config SHA-256. A revision-like hex commit is supplemental, not a byte hash.
- Separate positive-size, content-addressed **adapter weights** and **adapter
  config** roles. Neither is permitted to masquerade as the base, tokenizer or
  base-model config. The five cross-role SHA-256s must be distinct; only the
  deliberately identical donor/recipient references **within the same role**
  may share a digest for the frozen-base method.
- A complete declared target map: rank, alpha, module names, A/B dtypes,
  input/output dimensions and exact target suffix. Its canonical hash matches
  the proposed format documented by the PEFT inspector/binding Drafts #38/#48.
  Consecutive dots and ambiguously overlapping target suffixes fail closed.
- Distinct train, protected holdout and shuffled-control references;
  declared rights provenance; donor-disconnected, cold recipient reload; $0
  external cost and bounded local GPU minutes.
- Optional **untrusted optimizer receipt reference**, which never elevates the
  result above `UNVERIFIED`. This permits proposing an adapter without claiming
  an optimizer receipt already exists.

The output deliberately contains **no** `donor_trained_checkpoint_sha256`
field. If a real merged/full checkpoint is subsequently generated, it belongs
to the separate full-checkpoint lineage method, with its own exact bytes.

## Relation to other Draft PRs

- **#37:** full-checkpoint/delta proposal remains valid for workflows that
  really produce such an artifact. This adapter-only source variant adds a
  separate schema without retroactively reinterpreting #37's digest roles.
- **#38:** the read-only PEFT inspector can supply candidate adapter/config
  hashes and tensor geometry. A source-only synthetic inspection was matched
  against this proposal's hashes and geometry; the result remained HOLD.
- **#48:** its strict binding of a full-checkpoint proposal must **not** be
  reused to equate adapter-only bytes with a fictional full donor checkpoint.
  A future typed join can compare matching hashes and canonical geometry while
  preserving this adapter-only lineage and host authorization boundaries.

## Admission is still unproven

Only an independently authorized host can open and rehash the real Qwen base,
tokenizer, config and PEFT files; verify target modules against actual target
layer dimensions; establish training rights and optimizer provenance; load an
isolated recipient with the adapter; disconnect the donor; cold restart; and
run sealed behavioral/negative-transfer evaluations. A proposed hash map does
none of those things. This code runs no real model, training, conversion or
service action, and grants no authority.

Manual source checks:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_peft_adapter_only_proposal.py' -v
```
