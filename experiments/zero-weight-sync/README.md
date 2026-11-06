# Zero-weight synchronization: bounded operational pilot

Run from the repository root with Python 3.10+ and **no extra dependencies**:

```sh
python experiments/zero-weight-sync/run.py demo
python experiments/zero-weight-sync/causal_trial.py demo --seed 317
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
revoke-after-validate (commit denied), post-commit use-grant revocation,
per-session grant isolation, replay after revocation remaining denied, session/target isolation, idempotent
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

### Wave 2 — randomized causal and ordinary-file counterfactual

`causal_trial.py` is an **additional scripted mock** experiment, not a replacement
for the original `run.py` pilot or a live language-model test. It generates 13
randomly named affine tool handlers and an independently selected hidden target
for each seed. Scripted A alone sees a task-independent three-point probe
signature through a restricted `DonorView` with **no target reference**, actually
evaluates candidates, and discovers the matching opaque reference. The synthetic
host oracle separately checks A's observation against the hidden
target before it can be committed to a scoped SQLite StateBus event. **Neither
the target coefficients nor probe answers nor held-out tasks are placed in the
StateBus payload.** After A disconnects, a fresh receiver-only Python process
uses frozen B artifact, available tool registry, and approved state to produce
outputs for independently drawn, unobserved integer tasks. B does not train.

Critically, an identically scoped ordinary JSON-file reference works too;
an operator who already knows the reference can supply it without A. Both are
intentional counterfactuals, not test failures. Expected example (`seed=317`):

| Condition | Held-out correct / 8 | What it establishes |
| --- | ---: | --- |
| B, no state | 0/8 | This *scripted* receiver cannot select a tool without a reference |
| B, valid StateBus ref | 8/8 | Host-checked scripted A discovery plus external ref enables host tool use |
| B, same ref in ordinary authorized JSON | 8/8 | StateBus is **not** needed for functional inference in this fixture |
| B, post-commit revoked use-time grant | 0/8 | New **local host-owned broker sidecar** enforces current permission at use |

These counts are **source-fixture expectations**, not claims for real models;
run the exact committed Git revision and record actual stdout and CI receipts.
Additional negative tests cover absent/forged A, wrong discovery/ref/corpus,
expired state, session isolation, malformed tasks/registry, stale local grants,
spoofed outputs, type-confused validator/scorer claims, and an operator-direct
no-A baseline. A **scoped causal source manifest** binds the new evaluator and
exact checked-in test-policy bytes together with the original StateBus core;
both must match `git show HEAD:<path>` before any local A-observation commit.
The event binds a **host-generated fixture receipt digest**; that receipt is
not independently witnessed or held in separate immutable custody.
Source and mock artifact SHA256 identities are checked; **there is no independent
cryptographic witness or independently protected test policy**. The checked-in
test can be replaced by anyone authorized to change the repository, so this is
recoverable byte provenance, not a signed external validation standard.

The original `run.py` fixture also now checks a separate, scoped, operator-owned
`.use-grant-*.json` file **on every tool invocation**. Revoking/deleting it
denies a committed capability, and replaying the already-committed proposal
cannot reactivate the grant. The additional causal receiver checks a similarly
separate operator-owned `broker.json` at every attempted invocation; deleting
or revoking it denies both StateBus and ordinary-file reference use. Because
the brokers and SQLite files reside in one local trust domain, these are
**control-flow regressions**, not real principal authentication, protected
storage, durable witness custody or production ACL. A SQLite commit and a
separate broker grant can fail independently; use fails closed if the latter
was not durably recorded.
Plain-file security could be strengthened with the same broker; do not count
its absence as a StateBus-specific superiority claim. Actual model-to-model
advantage remains **UNKNOWN** until differently implemented inference engines
are tested under matched runtime, task, authority and cost conditions.

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
