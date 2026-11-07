# Witness-composed broker — explicitly unsafe for production

**Status:** SOURCE/TEST ONLY. Not installed, not a durable trust boundary,
not a replacement for the existing `HostOwnedEffectBroker` runtime, and not a
solution to whole-authority snapshot rollback. This is an experimental
composition of the v3 witness source reference with a separate, test-only
`SourceOnlyWitnessedPlanBroker`. It deliberately requires
`explicit_source_only=True` at construction.

## Observable source behavior

The caller configures a nonempty immutable `required_keys` policy outside the
SQLite database, and a source-only authority identity/generation and witness
object. Legacy `grant`, `grant_plan`, `execute`, and `execute_plan` calls for
these protected keys are rejected even when old v2 rows exist. New v3
approvals live in `v3_witness_plan_grants`, a table that old v1/v2 broker
queries cannot see. The v3 code exposes separate `grant_witnessed_plan`,
`revoke_witnessed_plan` and `execute_witnessed_plan` test paths.

Every source-level v3 mutation obtains the main SQLite writer, compares the
main witness identity/generation/sequence/event digest with the witness tip,
then performs an irreversible source witness event before committing matching
main state. A failed SQLite commit after `ISSUE`, `REVOKE` or `CLAIM` leaves
the witness ahead; subsequent v3 actions halt rather than reset/replay.
Each v3 operation also commits an independent witness `CLOCK` event **before**
its grant validity decision, including a subsequently denied/expired attempt.
The source model therefore rejects a stale host-clock observation after the
ordinary SQLite clock sidecar alone is restored, so long as the same live
witness retains the previous epoch. This still does not attest clock truth.
The exact witnessed epoch, not a fresh unrecorded `_plan_now()` sample, is
used for each authorization check. Execution takes separate witnessed clock
samples for claim and pre-handler phases. Receipt timestamps use the latter
sample as a bound rather than independently attested wall-clock completion.

`CLAIM` is appended **before** the handler is invoked, so a failed main
commit cannot create a retryable effect. An attempt may remain ambiguous
after process crash or callback failure; there is no exactly-once delivery.
The implementation rechecks the tip after external projection callbacks and
immediately before handler invocation, blocking a deterministic external
revocation change in that window.

## Key unresolved qualifications

1. **Provider authenticity:** `require_witness` is structural validation,
   not a trust-root check. A forged object can return arbitrary cursors and
   accept fabricated claims. The in-process reference itself is trivially
   reconstructed with its same genesis and forgets prior grants/revocations.
2. **Policy authority:** an attacker or older runtime able to instantiate the
   unguarded v2 broker can execute preexisting v2 grants even for keys this
   subclass marks required. The host must enforce mandatory policy outside
   rollbackable SQLite and prevent access to old code/handlers. This module
   does not establish such host enforcement. After rollback before the v3
   binding exists, no SQLite-only marker can prove prior v3 deployment.
3. **Time:** the sidecar high-water is still host-owned and rollbackable;
   there is no authenticated monotonic epoch or external freshness source.
4. **Concurrent revoke:** CLAIM is the authorization linearization point.
   A later revoke prevents **new claims**, but is not guaranteed to cancel
   an in-flight claimed effect. A just-in-time witness tip check can catch
   witnessed advances before the check but has an unavoidable check-to-call
   gap without a qualified lease or lock spanning handler start. Source
   tests do not establish revocation-before-handler atomicity.
5. **Crash recovery:** witness-ahead/main-behind becomes an intentional HALT.
   No operator-supervised unknown-outcome reconciliation, independent event
   receipt escrow, or idempotent handler replay protocol has been qualified.
6. **Host effects:** no target host, external witness, authenticated transport,
   process identity, handler binary, or durable storage was touched or
   validated. Test callbacks are local fixtures only.

**Security verdict:** PASS as a bounded in-process composition test, **FAIL
for production protection** against historical rollback. Do not enable this
test-only class as an effect authority. A real v3 implementation requires a
host-qualified external witness plus fail-closed authority policy enforced at
the single host entrypoint, not only in this subclass.

## Offline test

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_witness_effect_gate.py' -v
```
