# Actual broker, scripted receiver plan

The real-receiver pilot's **X** arm tests a cooperative, in-memory `ToolGrant`.
It does **not** exercise the SQLite-backed `HostOwnedEffectBroker`. This
independent local fixture supplies the missing **source-level seam** without
changing the seven-arm model evaluation or claiming that a neural model has
emitted a successful plan.

```bash
python3 experiments/real-model-sync/broker_probe.py
python3 -m unittest discover -s experiments/real-model-sync -p 'test_broker_probe.py' -v
```

The fixture:

1. Builds a deterministic synthetic catalog with the same 13 opaque-handle
   operations as `pilot.py` and parses a literal typed JSON plan through
   `pilot.parse_plan`.
2. Commits that catalog to a **real SQLite StateLedger** using shape-valid
   *test-sentinel* evidence and same-process fixture validation callbacks.
3. Configures an actual `HostOwnedEffectBroker` with an `ApprovedPlanEffect`
   handler accepting **the broker-checked plan argument**, not a plan captured
   by a Python closure. `grant_plan` binds canonical plan SHA-256, committed
   state digest, and host-asserted handler artifact/source; `execute_plan`
   produces one effect and a durable version-2 plan-bound `ok` receipt.
4. Revokes an otherwise eligible grant **after** parsing the plan and verifies
   a denied use performs no handler effect. A grant expiring at the exact
   decision time is also denied; a reopened broker rejects replay and a
   differently scoped context cannot inherit the grant.
5. Attempts a **different one of the 13 handles under the same state key**,
   a second invocation on a consumed single-use plan grant, and a changed
   host-asserted handler artifact. Each must be rejected before the handler.
6. Commits a second real `StateLedger` revision *after* approval and confirms
   that execution is denied with a consumed attempt claim and **zero effects**.

The tests inspect the actual `authority.sqlite3` `attempts` table: the
authorized invocation is `ok`, and the state-changed post-claim invocation is
`denied`; pre-claim mismatched/revoked/expired requests have **no durable attempt
receipt**. A claimed/crashed invocation remains ambiguous and fail-closed.
The report's `wire_plan_text_sha256` hashes the literal JSON string (including
spaces); `allowed.receipt_plan_sha256` hashes the broker's **canonical compact,
sorted** JSON representation. They are not expected to match. A separate test
independently recomputes the canonical digest and compares it to the receipt.

### Claim boundary and next seam

- `LOCAL_SCRIPTED_BROKER_FIXTURE_TEST`, **not** a local/remote language-model
  inference, training, independent verifier, authenticated principal, or
  host-security qualification.
- The broker serializes use-time grant changes among cooperating connections
  using its SQLite authority. A same-UID caller that can use the raw database
  or handler directly can bypass the checks. In-process role names do not
  create independent identity or provenance.
- The new opt-in plan API proves **source-level exact-content equality** for
  the selected plan and projected state at use time. It does **not** prove
  that an independent authority approved the model's choices, that the model
  produced the plan, or that the host-asserted handler artifact/source actually
  corresponds to executed code. A dishonest same-UID host could reuse the
  purported digest with a different callable.
- Legacy `grant`/`execute` remains intentionally **key-only and reusable**;
  protected model-selected plans must use the opt-in plan API. A caller that
  routes through the legacy surface is not plan-bound by this fixture.
- Host wall-clock rollback can undermine expiry if time is not provided by a
  suitably monotonic authority; single-use plans and revocation are separate
  protections and do not establish trusted time.
- Reopening a broker object against the same database is a persistence check
  **within one running process**, not a crash/restart or recovery qualification.
- Only pure arithmetic effects are used. SQLite does not guarantee rollback
  or exactly-once delivery of external irreversible effects.

This negative boundary is deliberate: a passing local revoked-use test must
not upgrade the existing pilot's `X` results into a proven production broker.
