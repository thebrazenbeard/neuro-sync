# Zero-weight synchronization: bounded operational pilot

Run from the repository root with Python 3.10+ and **no extra dependencies**:

```sh
python experiments/zero-weight-sync/run.py demo
python -m unittest discover -s experiments/zero-weight-sync -p 'test_*.py' -v
```

This is an actual SQLite StateBus **PROPOSE → VALIDATE → COMMIT** demonstration,
not neural training. Frozen **mock** computational adapters A and B have different
rendering styles, but read the **same `(target_id, session_id)` governed namespace**.
The scripted A adapter first discovers a host capability and actually executes
a synthetic, task-independent probe. A separate host check validates the
observation before A's proposal is committed. With A absent or misidentified,
the proposal fails and B stays at baseline. The approved state contains only
`{ref, policy, valid_until, handler_sha256, donor_artifact_sha256}` for a host-owned capability
that canonicalizes Python source. It carries **neither a task nor its answer**.
The untouched B initially cannot resolve the operation. After a grant, a **fresh
Python receiver process** independently opens the SQLite ledger and the same
unchanged B artifact; it invokes the host tool through the opaque ref. A host-run
checker tests the actual result's AST equivalence and canonical form. Separate
subprocesses yield different displays for A and B, not identical behaviors.

Controls include no state, rejected decoy A proposal, absent/forged A,
stale/expired projected state,
revoke-after-validate (commit denied), session/target isolation, idempotent
replay, a post-commit in-process handler-rebinding negative, eight post-commit
inputs, forged `verified: true` model claims, and
malformed/oversized task rejection, receiver clock-backdating rejection, and
before/after mock adapter artifact SHA256 identity. A source commit OID, scoped
canonical manifest SHA256 (experiment + StateBus schema/ledger source), **A**
artifact SHA256 and deterministic host checker receipt digest bind the
**test fixture**; grant also checks the scoped working files match the exact
current Git HEAD commit (forged/nonexistent OIDs and dirty files are rejected);
they are **not** signed custody or authenticated external
authority. Host callbacks and tools run in the same local Python trust domain.

## Claim ceiling and honest comparisons

- **Supports:** scripted A-observation-gated *external capability discovery*, local
  SQLite persistence, receiver-only process restart, task-blind reference
  transfer, and tool-result checks using unchanged mock adapter artifacts.
  A does not autonomously discover capabilities; the tool registry is already
  installed on the host. This is a test of reference transfer, not learning.
- **Does not support:** weight transfer, model learning, semantic generalization,
  independent trust principals, security against a host DB administrator, a
  real Qwen3.5-4B restart, or neural behavioral identity.
- This is **not compute-matched** against the repository's Trek exact-weight
  transplant or distillation protocols. Those use different mechanisms and
  learners. Comparable end-to-end budgets, fixed heldouts and a real-model
  trial are required before claims of superiority.
- State expiry is enforced by this **receiver fixture**; the core ledger
  currently commits deadline-valid events but retains historical projections.
  Public receive-mode uses the host clock, never a caller-supplied override.
  A fixture-level handler source pin rejects simple registry rebinding *before*
  invocation, but does not prove runtime integrity. This formatter is pure and
  effect-free; **arbitrary effectful capabilities are NOT authorized** by this
  demonstration. Production needs authenticated policy, executable attestation,
  independent host broker, revocation and expiry checking at use time.

Receiver task and checker submissions use stdin, not exposed process argv.
All tasks, refs and model artifacts are synthetic. No Star Trek scripts,
secrets, user data, model weights, network calls or paid compute are needed.
