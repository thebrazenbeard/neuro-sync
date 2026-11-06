# Governance differential: what actually makes the difference?

Source-level, local mechanical test with **no LLM, outside verifier, live
network authority, or trained neural result**. All three arms receive the same
synthetic capability state and use the **identical**
`HostOwnedEffectBroker` code, context and host-approved handler.

| Control or intervention | StateLedger + broker | Ungoverned JSON + broker | Governed JSON test adapter + broker |
| --- | --- | --- | --- |
| Authorized use, durable invocation replay, revocation, expiry and scope | Same broker | Same broker | Same broker |
| Authorized nonstale state update | Accepted | Accepted | Accepted |
| Producer write rejected by host test callback | Rejected | No producer gate installed | Rejected |
| Stale revision refused | Rejected | No revision CAS installed | Rejected |

The basic file arm is **deliberately ungoverned**. A third, single-process
*governed JSON* test adapter implements equivalent callback authorization and
revision comparison, and is expected to match StateLedger for those two
interventions. It does not add crash safety, atomic interprocess writes,
process authentication or durable revision persistence. The experiment tests
whether explicit controls affect state before the *same* broker consumes it.
It does **not** prove that SQLite, a StateBus transport, or the Neuro-Sync
protocol are intrinsically safer or more accurate than a file system.
The experiment also cannot isolate same-UID callers from directly reading or
changing any of the stores. Test-only provenance hashes and callbacks are **not**
cryptographic receipt validation or real independent third-party review.

Run from the repository root:

```bash
python3 experiments/governance-differential/study.py
python3 -m unittest discover -s experiments/governance-differential -p 'test_*.py' -v
```

`study.py` runs all three arms in fresh temporary directories, checks
equivalent broker behavior and then tries three separately reset state
interventions: a valid approved update, an unapproved producer write and a
stale key-revision write. It raises on
any unexpected outcome and prints a JSON receipt labeled
`LOCAL_TEST_SOURCE_ONLY`; the process exit status does **not** attest any
remote runtime or model behavior.

The broker's foreign-principal isolation probe is performed **before** the
rightful principal first uses its grant. The rightful principal must then use
that same grant successfully. Legacy grants in this test are **multi-use**;
only invocation IDs have replay protection. The earlier probe order was less
direct, but reuse of the grant alone did not create a replay false positive.
An explicit negative-control test temporarily removes principal scoping and
confirms the study detects unauthorized foreign execution. This remains a
same-process synthetic test, not authenticated principal isolation across
operating-system identities.
