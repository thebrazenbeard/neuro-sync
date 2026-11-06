# Exact-plan host effect gate (source prototype)

The opt-in `HostOwnedEffectBroker.grant_plan(...)` / `execute_plan(...)` path
binds a *host-issued approval* to a particular canonical JSON plan, state
key, projected state snapshot and host-installed plan-aware handler descriptor.
The existing `grant(...)` / `execute(...)` key-only path retains its historical
behavior. A plan grant **cannot** be consumed using legacy `execute`, and a
legacy key-only grant **cannot** be consumed using `execute_plan`.

## Calling contract

The host constructs `HostBoundContext` and installs an
`ApprovedPlanEffect(handler, handler_artifact_sha256, source_git_oid)` in
`approved_plan_handlers`, by key. The plan handler accepts **two** arguments:
`handler(detached_projected_state, detached_approved_plan)`. These arguments
are independent JSON copies; changing either cannot mutate the caller's plan
or the object returned by the ledger's `project` call. Handler *behavior* is
not attested by this interface. In particular, it could ignore the plan or
consult different external state, and would have to be audited separately.

Host code then invokes:

```python
broker.grant_plan("approval-17", "capabilities.synthetic",
                  plan={"steps": [{"handle": "cap-one"}]},
                  expires_at_epoch=200.0)
result, receipt = broker.execute_plan(
    "approval-17", "attempt-17", "capabilities.synthetic",
    plan={"steps": [{"handle": "cap-one"}]},
)
```

The broker, **not** a model message, computes SHA-256 over bounded canonical
JSON: UTF-8 with sorted object keys, compact separators, non-ASCII retained,
NaN and Infinity rejected, and input size bounded by `MAX_READ_VALUE_BYTES`.
Both the requested plan and plan-gated state snapshots are preflighted with
limits on nesting depth, visited nodes and individual scalar length before
canonical serialization allocates the full JSON representation.
The `plan_grants` table stores the resulting `plan_sha256`, the SHA-256 of projected state
at approval time, and the asserted handler artifact/source identifiers. A
caller-supplied hash is neither accepted nor used. `execute_plan` checks the
exact plan hash and handler identifiers before recording an attempt, then
checks the grant again and compares a fresh detached state snapshot against
the stored approval-state digest before invoking the two-argument handler.

Plan grants are **single-use**. The first valid invocation atomically claims
both its invocation ID and the plan approval, before any handler call. A
second invocation ID on that approval is denied, including after the first
handler raises, the state changes, or the process crashes after claim.
Legacy key-only grants remain reusable by design. A rejected wrong plan
*before claim* does not consume the approval or invocation ID.

The handler is invoked under the broker's SQLite writer transaction; another
broker-owned revocation cannot commit *during* that handler call. Attempts
have a durable `claimed` row before the effect, so replay of the invocation
ID after a crash or ordinary handler exception is rejected. A rejected plan
before claim creates no attempt; its invocation ID can be reused with a
subsequently valid plan. Once claimed, changed state, expiry, revocation or
handler failure consumes the approval and invocation ID. Receipts for plan attempts use
`version=2` and include the executed `plan_sha256`; legacy receipts remain v1.

Existing SQLite legacy grant and attempt rows survive idempotent additive
migration. Plan approvals live in a **separate `plan_grants` table**, invisible
to older broker binaries that query only `grants`. Historic attempts remain
retrievable. New code disallows duplicate grant IDs across the two tables
within a scope. No grant ID is resurrected after revocation via the new API.

## Explicit limits and evidence classification

- **SOURCE/TEST:** This is an in-process, single-authority SQLite broker,
  qualified only by source-level tests. The hashes and receipts are
  same-trust host evidence, not independent proof of external effects.
- **NOT OS/REMOTE AUTH:** Same-UID users with direct database, ledger or
  handler access can bypass the gate. `HostBoundContext` is host-issued
  metadata, not authenticated user identity in this prototype.
- **NOT FULL TOCTOU ELIMINATION:** Grant and effect-side state copies are
  compared; independent `StateLedger` writers are not transactionally locked
  by this separate broker database, and a handler may consult ungoverned
  external state. The hash attests only to the copy actually passed.
- **NOT EXACTLY ONCE:** A process can crash after durable claim but before or
  after a nontransactional side effect; recovery must treat `claimed` as
  ambiguous and refuse automatic replay. The broker affords at-most-one
  handler *attempt* per invocation ID, not exactly-once external delivery.
- **NOT HANDLER ATTESTATION:** `handler_artifact_sha256` and `source_git_oid`
  are supplied by the host. An incorrect or deliberately falsified descriptor
  is not detected. A changed descriptor is denied when using an old grant.
- **HOST CLOCK TRUST:** Expiry relies on the host-supplied clock. If it rolls
  backward, an expired but *unconsumed* grant may become valid again. Hosts
  must supply and protect a trustworthy time source; no persistent high-water
  clock floor is implemented here.
- **OLD-BINARY CAVEAT:** Older code cannot consume a *new* plan approval via
  its legacy grant lookup. An independently authorized old binary may still
  create entirely new key-only legacy grants, since this is not an OS-level
  service-version pin or exclusive control of the SQLite file.
- **NOT SCIENTIFIC GENERALIZATION EVIDENCE:** No training, model improvement,
  causal transfer, human review or independent production qualification is
  established by these tests.

Do not expose grant creation to an unauthenticated model/remote endpoint, or
allow models to supply their own supposedly approved handler descriptor.
