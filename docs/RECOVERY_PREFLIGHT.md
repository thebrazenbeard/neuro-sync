# Paired-authority recovery preflight (read-only diagnostic)

Neuro-Sync plan authority now spans a main SQLite database and its companion
`<main>.clock.sqlite3`. A pair can look structurally intact even after a
historical restore that revives expired approvals or consumed one-use plans.
This preflight **does not solve or prove resistance to those rollbacks**.

Run from the repository root, without activating the broker:

```sh
PYTHONPATH=src python3 -m neurosync.bus.recovery_preflight /path/to/authority.sqlite3
```

It first rejects WAL-mode database headers and pre-existing SQLite journal,
`-wal` or `-shm` artifacts **before opening SQLite**, because SQLite can
otherwise create WAL/SHM helper files even for a `mode=ro` connection.
Under a quiesced rollback-journal pair, it opens **both files read-only**, runs
a bounded SQLite quick check, verifies
required main/sidecar table columns, the main `READY` binding against the
sidecar UUID, and validates that the
saved floor is finite (or is NULL only before any v2 grant). The JSON report
has `authority_effect: "NO_INTENTIONAL_AUTHORITY_DB_WRITE"` and one of two classifications:

- `BLOCKED`: missing, corrupt, mismatched, incomplete or invalid pair.
- `STRUCTURALLY_COMPATIBLE_UNATTESTED`: no structural problem detected in
  two non-atomic reads. **Not permission to recover, authorize, execute, or
  claim historical freshness.**

Exit code is 0 for the latter diagnostic and 2 for `BLOCKED`. An unrecognized
file can cause an error with diagnostic details, but neither file is repaired,
created, migrated, or rewritten by this tool.

The pre-open header and journal checks are **not atomic** with concurrent
external writers. A process changing journaling mode or performing writes
concurrently can invalidate the no-new-files observation. Do not run this
as proof of zero filesystem change on a live active database. It requires
quiesced file custody to make that claim.
The diagnostic checks journaling state again after closing connections,
but this does not remove races or undo files a concurrent writer may cause.

Schema inspection checks **required column presence**, not every constraint,
index, runtime handler, ledger, or version binding. Even a structurally
compatible pair can still be unusable or historically restored. The result
must not be treated as a full broker-installation qualification.

> **Known adversarial falsifiers:** Restoring both files from an earlier
> snapshot can revive an expired grant. Restoring only an older **main**
> database can replay an already consumed plan while the clock sidecar stays
> current. Neither attack requires malformed SQLite data. A clock-only
> witness is insufficient; independently durable **authorization-event
> history** (grant issuance, consumption, revocation, authority generation
> and observed time) is required for broader recovery protections.

The snapshots are not held under a shared writer lock during inspection;
concurrent writers may change them between reads. Use no real restore or host
side effects without separate authority and a controlled recovery protocol.

Run source checks:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_recovery_preflight.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_recovery_falsifiers.py' -v
```

The second test suite intentionally reproduces known security limitations.
Its PASS means the falsifier reproduced; it is **not a security PASS**.
