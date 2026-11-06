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
The `clock_guarded_plan_grants` table stores `plan_sha256`, the SHA-256 of
projected state at approval time, and asserted handler artifact/source IDs. A
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
migration. Active plan approvals now live in a third,
**`clock_guarded_plan_grants`** table, invisible both to key-only brokers and
to earlier plan-aware brokers that read `plan_grants`. Historic `plan_grants`
rows are retained **as inert history** in the new API; pending old approvals
are not silently migrated or executed. A host must issue a fresh v2 approval
with a new grant ID. Reuses are denied across all three tables in the same
scope, and revocation covers each table without trusting the clock. This is a
deliberate fail-closed compatibility change, not a transparent schema upgrade.

The v2 sidecar `<authority-db>.clock.sqlite3` records the greatest *observed*
host epoch, independently committed from the main authority transaction.
Main `plan_clock_binding` binds it by UUID; the old in-main `plan_clock_state`
is obsolete metadata. Plan grant, claim, effect and receipt-time reads use
main→sidecar lock ordering, including expired **pre-claim denials**. Once a
request observes epoch 250, another request at epoch 150 is denied against
the same intact paired files, even after restart. A missing/mismatched READY
sidecar denies plans; existing revocations remain available. Identical times
are allowed; future observations raise the floor. This is not independent
evidence of genuine UTC progression or anti-rollback protection against a
valid historical same-UUID sidecar snapshot. Both files require paired
backup/custody. See
[`docs/CLOCK_AUTHORITY.md`](CLOCK_AUTHORITY.md).

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
- **HOST CLOCK TRUST:** Expiry still relies on the host-supplied clock. The
  separately committed sidecar high-water rejects detected rollback, but
  cannot infer time that passed without observation or defend against a valid
  older same-UUID sidecar/paired snapshot restore or malicious host/DB edits.
  A forward jump may block subsequent valid work until the host clock catches
  up or a separately authorized recovery occurs. Hosts must still supply and
  protect a trustworthy clock; the broker does not independently attest time.
- **OLD-BINARY CAVEAT:** Key-only and earlier plan-aware binaries cannot
  consume *new v2* approvals because they do not query the new table. An
  earlier plan-aware binary can still execute **historical v1 approvals**
  against the untouched v1 table, or create new v1 approvals. An older host
  may also issue new key-only grants. This is not OS-level version pinning.
- **NOT SCIENTIFIC GENERALIZATION EVIDENCE:** No training, model improvement,
  causal transfer, human review or independent production qualification is
  established by these tests.

Do not expose grant creation to an unauthenticated model/remote endpoint, or
allow models to supply their own supposedly approved handler descriptor.
