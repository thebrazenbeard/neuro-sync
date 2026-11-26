# Neuro-Sync → TRAININGDAY transfer proposal v1

**Status: source-only, untrusted proposal. No physical launch, admission, or
verified model artifact.** This is a separate preflight input from the source
mechanics observation in [Draft PR #30](https://github.com/thebrazenbeard/neuro-sync/pull/30).

`neurosync.trainingday_proposal.propose_transfer(intent)` validates a strictly
shaped **intention**, returns a canonical SHA-256 of that self-declared intent,
and **always** reports `trainingday_ready=false` and
`physical_launch_authority=HOLD`. The checksum helps detect accidental changes
between agents. It is not a signature, physical file hash readback, rights
grant, operator approval, or independent attestation. A malicious caller can
invent a self-consistent proposal and its digest.

## Two different methods

- `exact-base-delta` (Path A): donor's pre-training base, recipient's initial
  base, tokenizer and configuration *must be declared* at the same exact SHA-256
  for this proposed method. A changed trained donor and a proposed adapter
  configuration SHA must also be declared. This does **not** verify actual PEFT
  tensor shapes or Qwen interoperability. Host-side, byte-level checks remain
  necessary.
- `receiver-native-distillation` (Path B): donor and recipient bases may be
  different, and no adapter transfer is claimed. Proposed training data and
  teacher outputs still need explicit rights verification and a genuine
  recipient-only training run.
- **StateBus (Path C) is excluded:** context sharing does not update weights.

For both methods, the strictly typed proposal requires:

| Field | Source-side check | Still required from independent host |
| --- | --- | --- |
| Lineage | Lowercase SHA-256 strings; donor changed; exact-base A equality | Read actual weight, tokenizer, config, adapter bytes and checkpoint lineage |
| Input artifact inventory | Named donor-trained, recipient-base, train, holdout and shuffle-control references; positive byte-size declarations; digest cross-references | Read and hash exact files; establish custody and immutability |
| Train/eval split | Train, sealed holdout and shuffle controls have distinct declared manifests | Confirm no examples, answers or rubrics leak across splits |
| Data rights | Source record digest, usage-scope declaration and holder label | Independently verify rights holder, license, consent and intended uses |
| Evaluation | Proposes untouched baseline, shuffled donor, donor disconnection and cold recipient reload | Execute, witness and score protected recipient-only results |
| Budget | Explicit $0 external cost and bounded GPU minutes | Observe actual host lease, resources, run cost and service state |

Unknown/extra fields, including `host_authorized`, are rejected so a caller
cannot smuggle an approval through this schema. Existing descriptive fields
such as `evaluator_custody="independent-host-proposed"` remain **plans**, never
evidence that an independent evaluator exists.

## Required separation of authority

1. **PROPOSE:** Neuro-Sync emits `UNTRUSTED_TRANSFER_PROPOSAL` with a digest of
   its proposed manifest. This stage cannot grant itself rights or trust.
2. **HOST VALIDATE:** A separately authorized executor opens and hashes actual
   inputs, verifies the rights record and sealed evaluation custody, checks
   runtime resources and a bounded workload lease. This implementation does
   not perform any of these actions.
3. **EXECUTE:** The host creates a **new recipient artifact** in isolation,
   preserving optimizer and adapter provenance when applicable.
4. **VERIFY:** Reload the exact recipient artifact without donor access,
   independently score heldout results, compare untouched and shuffled arms,
   and bind the serving process to loaded model bytes.
5. **COMMIT:** A host-owned, durable receipt records exact tests and revisions;
   it cannot be authored or self-certified by the training model.

Until a separate TRAININGDAY authority validates all required physical
evidence, keep `model-training` Draft #351's preflight on HOLD. An untrusted
proposal, a successful toy transfer, or a green source CI run must not be
converted into a launch approval or behavioral claim. No current Qwen model,
checkpoint, GGUF, or host process was trained or modified by this contract.

Run source tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_trainingday_proposal.py' -v
```
