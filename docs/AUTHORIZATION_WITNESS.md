# Authorization witness — v3 research contract (SOURCE ONLY)

**Status:** Unconnected Python reference, not an installed or independent
security boundary. The production `HostOwnedEffectBroker` and its v2 SQLite
authority are unchanged. No migration, schema revision, provider registration,
rollout, runtime deployment, or claim of protection has occurred.

## Problem and intended invariant

In the source-level rollback attack, restoring an older copy of the main
SQLite authority can erase the local record that a grant has been used.
An attacker can make old approval state appear current. A second SQLite file
alongside the main database is not independent custody if both are restored.

The v3 target requires an **independently retained, authenticated, linearizable,
monotonic authorization event stream**. Every grant issue, irrevocable
execution claim, revocation, and trusted clock observation must be recorded
outside the main authority's rollback domain. The main DB must persist the
corresponding witness sequence and SHA-256 event digest. Each subsequent
mutation must compare the exact persisted pair against the witness's current
tip, under the main writer's exclusivity, before performing any effect.
Mismatch, absence, rollback or outage means **deny**, not fallback to v2.

An external host implementation would additionally authenticate the authority
UUID, generation, principal, target, session, grant, state key, planned state,
plan digest, handler artifact and source identity. It must bind main authority
sequence/hash as well as witness sequence/hash in the authority transaction,
not merely trust a caller-supplied cursor. This independent binding is
**not implemented** in the in-process reference.

## Prototype in `authorization_witness.py`

| Source element | Current source-only guarantee | Production proof absent |
| --- | --- | --- |
| `WitnessGrant` | Immutable, validated authority/generation/principal/target/session/grant/state/plan/handler/expiry binding | Authenticated origin, actual handler artifact attestation |
| `WitnessCursor` | Exact witness sequence + SHA-256 event digest compare | Anti-rollback custody outside the DB/process/host |
| `issue` | Reject reused grant ID, bad generation, expiry, stale tip | Host-specific authorized grant issuance |
| `reserve_claim` | Append consumed claim **before** calling effect handler; later claims denied | Durable append before non-reversible effect, crash recovery |
| `revoke` | Reject revoked claim and repeated revocation | Revocation propagation after partition |
| `observe_time` | Reject locally decreasing or non-finite observations | Trusted clock, freshness, offline expiry |
| `require_witness` | Reject `None`, invalid cursor or noncallable method contract | Verified provider identity/capabilities |
| SHA-256 event chain | Exported canonical JSON contains full binding + predecessor digest; hashes independently reconstructable | Witness authentication, signing, anchoring |

The `InProcessAuthorizationWitness` has one process-local `RLock`, so
simultaneous calls to the **same live object** serialize; competing claims
with the same expected tip yield one successful append. Its constructor is
public and regenerates the same genesis for identical authority/generation.
Thus a process restart, object reconstruction, separate worker, or attacker
with state mutation can fork/rollback this reference; it does **not** satisfy
the required independent custody. Copy, deepcopy, pickle and in-object reset
are deliberately rejected, but these safeguards do not survive a new object
or protect against direct Python memory mutation.

### Event ordering

1. Acquire exclusive main authority writer.
2. Read main authority's current generation, identity, sequence/hash and
   persisted witness cursor; authenticate an independent witness provider.
3. Compare main's tip and binding to witness's tip and binding. Any mismatch
   or unavailable provider denies progress. No auto-reconciliation.
4. For issue, revocation or trusted clock observation: append the event to
   the independent witness with compare-and-append, then commit matching main
   state/cursor. A failed main commit after witness append intentionally halts
   subsequent operations until authorized recovery; witness is never rewound.
5. For execution: append `CLAIM` to witness **before** invoking the handler;
   persist its cursor with main authority transaction. A subsequent crash may
   leave the side effect outcome unknown; the same grant must never be retried
   automatically. Design an operator-controlled unknown-outcome recovery
   protocol before production deployment.

The reference tests simulate this contract, including a restored in-memory
SQLite authority cursor, but do not integrate with `effect_gate.py`. They
cannot establish atomicity across an external witness, DB commit, and handler.

## What hostile tests demonstrate

- Source test: normal issue followed by claim consumes a grant once.
- Source test: restoring the main SQLite cursor to its prior snapshot cannot
  pass the unchanged live witness's current-tip comparison.
- Source test: a witness append followed by failed main DB commit leaves a
  deliberate **fail-closed mismatch**; there is no automatic repair.
- Source test: stale expected cursor, altered plan/state/handler/identity,
  invalid expiry or clock, duplicate grant ID, revoked claim, and concurrent
  competing claims reject.
- Source test: integer timestamps above `2**53` reject rather than silently
  rounding a literal expiry boundary into a different binary64 value.
- Source test: equal int/float expiry values produce identical canonical
  events and digests; event exports support independent hash/predecessor
  reconstruction and reveal naive payload tampering.
- Source test: the same process can create a **new witness object with matching
  genesis**, demonstrating the absence of independent persistence and the
  remaining split-brain/fork hazard.

These are **TEST** evidence for a deliberately small in-process source
reference, not a runtime probe or a security qualification of a network,
host, multi-process, or independently stored provider.
An adversary controlling the complete event log can rewrite its entire
unauthenticated chain: reproducible SHA-256 hashes are **not** an anti-rollback
or anti-fork trust anchor.

Run from the repository root:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_authorization_witness.py' -v
```

## Production blockers — fail closed

1. **Custody:** no independent durable append-only witness, anti-rollback
   protected identity, trust root, authenticated transport, or quorum.
2. **Atomicity:** no linearizable cross-process compare-and-append, main writer
   integration, authority sequence/hash binding, two-phase crash handling, or
   effect outcome reconciliation.
3. **Clock:** no trusted monotonic time source or remote freshness attestation;
   an arbitrary caller can report stale wall time. Local monotonic comparison
   alone does not prevent freeze/rewind under restart.
4. **Split brain:** same authority/generation can be reconstructed in a
   different process with identical genesis. A cloned or restored external
   witness would re-enable replay unless custody prevents fork and restart.
5. **Downgrade:** the v2 broker remains available; nothing in this source module
   prevents callers from bypassing the optional v3 interface. Actual deployment
   must refuse every effect unless qualified witness enforcement is mandatory
   at a single authorized effect entry point.
6. **Handler mutation:** a source hash field is a supplied assertion, not
   evidence of the loaded code bytes, executable image or running process.
7. **Recovery:** missing/pending/claimed/unknown-outcome states need explicit
   operator procedure and durable causal receipts before real-world execution.

**Verdict:** Mechanics for a v3 witness are testable in source. A truly
non-rollbackable authorization witness and its host integration remain
**UNPROVEN AND UNIMPLEMENTED**; no production acceptance is warranted.
