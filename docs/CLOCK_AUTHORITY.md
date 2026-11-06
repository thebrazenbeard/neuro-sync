# V2 observed-clock floor: source-level authorization mitigation

This is a **host-owned SQLite mechanism** for `grant_plan` and
`execute_plan`. It is not a secure time service, cryptographic clock proof,
remote principal authentication, or evidence of neural-model capability.

## Bounded claim

The broker pairs a main authority SQLite file with a separate sibling
`<authority-db>.clock.sqlite3` containing `clock_state.last_observed_epoch`.
The main file retains a `plan_clock_binding` UUID/status and the sidecar
holds the matching UUID. **Both files are part of one authority** and need
joint custody. The old in-main `plan_clock_state` table is not authoritative
for this implementation.

Under an existing main `BEGIN IMMEDIATE` writer lock, each plan clock read
acquires the **separate sidecar SQLite connection and writer lock**, compares
the freshly read host epoch with its stored maximum and commits the sidecar
**independently** of the main transaction:

- If the host epoch is **less** than the maximum, the plan operation fails
  closed with `ReadAccessDenied`; no plan handler executes.
- If equal, plan work may continue under ordinary grant, expiry, state and
  single-use conditions.
- If greater, the new value is durably committed even if the main grant,
  claim, effect, receipt or SQL transaction later fails. An observed expiry
  at `t=250` cannot be revived by a subsequent call at `t=150` against the
  *same, untampered pair of files*, including after broker restart.

Every plan phase first checkpoints before fallible callbacks and rechecks
inside its substantive writer transaction; receipt timestamps also pass
through this durable sidecar clock, not an unrecorded raw host-clock read.
No code path nests sidecar→main writer locks; the required lock order is
**main→sidecar**, with separate connections and commits. This adds sidecar
writer transactions per plan operation; latency impact has not been measured.

Startup binds a fresh empty authority using a PENDING→READY main marker and
matching sidecar UUID. A READY sidecar is opened `mode=rw` (no automatic
re-creation). Missing, malformed or mismatched files deny plan operations;
existing v2 grants without their binding cannot bootstrap a fresh floor.
The persisted epoch must be a finite number: a NULL floor is allowed only
before any v2 approval, and a NULL, infinity, negative infinity or text
epoch in an authority with v2 history fails closed. These checks catch
malformed data, not a deliberately restored valid older epoch.
`revoke` deliberately does **not** depend on clock validity or sidecar
availability; an owner can revoke even when the clock has regressed.
The legacy key-only `grant`/`execute` path is not upgraded or made safer by
this mechanism.

## Fail-closed source migration

New plan approvals use `clock_guarded_plan_grants`, which is **invisible** to
the prior version that only knows `plan_grants`. Previously approved v1
records are retained but **inert** in the v2 `execute_plan` path. There is no
automatic migration, ID reuse, or restoration: a host must independently
approve a **new** plan grant with a new ID. This does not delete historical
grant/attempt evidence. Revocation can still mark old v1 approvals.

An operator running the older v1 source can still execute *old* v1 grants;
the source cannot prevent downgrading a trusted host process or bypassing the
DB. Production use requires separately verified control of executable
version, database custody, caller identity and clock source.

## Deliberate limitations

> **Unobserved time is unknowable.** If wall time advances beyond a grant's
> expiry and then rolls back before any authorized process records that
> advanced time, this floor cannot detect it. Neither the clock nor the
> SQLite file is independently attested.

> **Forward jump may poison availability.** A mistakenly large observed
> clock value causes subsequent ordinary clock values to fail closed. No
> automatic reset or downgrade fallback is provided. **Restoring a valid
> older sidecar with the same UUID can erase high-water without detection
> and revive an old v2 grant**, even when the main DB remains unchanged.
> Restoring both files to an earlier paired snapshot is also a bypass.
> UUID pairing detects mismatches, *not* same-UUID historical replacement.
> Back up/restore the main DB and sidecar as a coordinated pair; an
> independently protected, nonrollbackable witness is required to defend
> against malicious/accidental historical restores. The tests explicitly
> demonstrate this non-guarantee. Recovery needs separately authorized host
> investigation and an auditable corrective plan.

> **Nontransactional boundaries remain.** Independent writes to the state
> ledger, changes to handler code behind host-asserted digests, crashes
> after durable invocation claims, and direct same-UID DB mutation remain
> outside this prototype's guarantees. Source tests do not qualify the
> host or prove exactly-once external effects. A sidecar and main DB restored
> from inconsistent snapshots may be unusable or, if UUIDs match, unsafe.

## Reproduce source-level checks (no model calls)

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_clock_floor.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_plan_gate.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_effect_gate.py' -v
```

Record exact source head and test results. A PASS on these fixtures is not a
real clock qualification or independent OS-host attestation.
