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
3. Configures an actual `HostOwnedEffectBroker` with an approved pure
   arithmetic handler; an authorized plan produces one effect and a durable
   `ok` invocation receipt.
4. Revokes an otherwise eligible grant **after** parsing the plan and verifies
   a denied use performs no handler effect. A grant expiring at the exact
   decision time is also denied; a reopened broker rejects replay and a
   differently scoped context cannot inherit the grant.

The tests inspect the actual `authority.sqlite3` `attempts` table: only the
  authorized invocation is present. A pre-claim denial has **no durable attempt
receipt**; a consumed or crashed invocation is governed by the broker's
different fail-closed ambiguity rules, not proven by this scenario.

### Claim boundary and next seam

- `LOCAL_SCRIPTED_BROKER_FIXTURE_TEST`, **not** a local/remote language-model
  inference, training, independent verifier, authenticated principal, or
  host-security qualification.
- The broker serializes use-time grant changes among cooperating connections
  using its SQLite authority. A same-UID caller that can use the raw database
  or handler directly can bypass the checks. In-process role names do not
  create independent identity or provenance.
- The handler captures the already parsed plan in a local closure; its
  `HostOwnedEffectBroker` receipt binds **state and result hashes, but not the
  submitted plan bytes/hash**. A future protected model-generated invocation
  needs a host-verified plan digest/decision binding in its authority protocol.
- Grants apply to the **state key**, not to one particular handle/plan among
  the 13 operations. A host that changes the approved handler for that key can
  execute a different arithmetic plan under the same still-live grant. This
  does not show an unauthorized caller bypassing a trusted host, but it means
  the fixture does not qualify plan-specific authorization.
- Reopening a broker object against the same database is a persistence check
  **within one running process**, not a crash/restart or recovery qualification.
- Only pure arithmetic effects are used. SQLite does not guarantee rollback
  or exactly-once delivery of external irreversible effects.

This negative boundary is deliberate: a passing local revoked-use test must
not upgrade the existing pilot's `X` results into a proven production broker.
