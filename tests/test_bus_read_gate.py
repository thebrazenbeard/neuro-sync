"""Source-level tests for the opt-in host-bound StateBus read/use facade.

These fake host callbacks are NOT authentication or independent witnesses.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from neurosync.bus import (
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger, StateOp,
    ValidationReceipt,
)

from neurosync.bus.read_gate import (
    AccessCheck, HostBoundContext, HostBoundReadGate, ReadAccessDenied,
)


class SpyLedger:
    def __init__(self, values=None, on_project=None):
        self.values = values if values is not None else {"capabilities.alpha": {"v": [1]}}
        self.on_project = on_project
        self.calls = []

    def project(self, target, session):
        self.calls.append((target, session))
        if self.on_project is not None:
            self.on_project()
        return self.values


class HostBoundReadGateTests(unittest.TestCase):
    def setUp(self):
        self.bound = HostBoundContext("operator-A", "node-B", "session-A")
        self.ledger = SpyLedger()

    def gate(self, policy, handlers=None):
        return HostBoundReadGate(self.ledger, context=self.bound,
                                 authorize=policy, approved_handlers=handlers)

    def test_read_returns_only_bound_session_key_and_copy(self):
        checked = []
        def authorize(check: AccessCheck):
            checked.append(check)
            return check.context.principal_id == "operator-A"

        value = self.gate(authorize).read("capabilities.alpha")
        self.assertEqual(value, {"v": [1]})
        self.assertEqual(self.ledger.calls, [("node-B", "session-A")])
        self.assertEqual([c.operation for c in checked], ["read-pre", "read-post"])
        value["v"].append(99)
        self.assertEqual(self.ledger.values["capabilities.alpha"], {"v": [1]})

    def test_pre_read_denial_never_opens_projection(self):
        with self.assertRaises(ReadAccessDenied):
            self.gate(lambda c: False).read("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [])

    def test_truthy_nonboolean_not_authorization(self):
        for forged in ("verified", 1, {"allowed": True}, [True]):
            with self.subTest(forged=forged), self.assertRaises(ReadAccessDenied):
                self.gate(lambda c, result=forged: result).read("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [])

    def test_forged_principal_or_target_binding_denied_before_projection(self):
        for forged in (
            HostBoundContext("intruder", "node-B", "session-A"),
            HostBoundContext("operator-A", "other-node", "session-A"),
            HostBoundContext("operator-A", "node-B", "other-session"),
        ):
            with self.subTest(forged=forged):
                gate = HostBoundReadGate(
                    self.ledger, context=forged,
                    authorize=lambda check: check.context == self.bound,
                )
                with self.assertRaises(ReadAccessDenied):
                    gate.read("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [])

    def test_policy_cannot_mutate_authoritative_projection(self):
        def mutating_policy(check):
            if check.operation == "read-post":
                check.value["v"].append("policy-mutated-copy")
            return True
        returned = self.gate(mutating_policy).read("capabilities.alpha")
        self.assertEqual(returned, {"v": [1]})
        self.assertEqual(self.ledger.values["capabilities.alpha"], {"v": [1]})

    def test_post_policy_projection_mutation_cannot_change_authorized_snapshot(self):
        """The delivered bytes must be the same bounded value approved post-read.

        Even a source-only host callback can indirectly mutate the ledger's
        projection during authorization.  That must not change the value
        returned after the post-check or supplied to an approved handler.
        """
        checks = []

        def authorize(check):
            if check.operation == "read-post":
                checks.append(check.value)
                self.ledger.values["capabilities.alpha"]["v"].append(99)
            return True

        delivered = []
        gate = self.gate(authorize, {
            "capabilities.alpha": lambda value: delivered.append(value) or value["v"],
        })
        self.assertEqual(gate.read("capabilities.alpha"), {"v": [1]})
        self.assertEqual(checks, [{"v": [1]}])
        self.assertEqual(self.ledger.values["capabilities.alpha"], {"v": [1, 99]})
        # Fresh gate/ledger keeps the use-path test independent of the first
        # read while exercising the same read-post authorization window.
        self.ledger.values = {"capabilities.alpha": {"v": [1]}}
        self.assertEqual(gate.use("capabilities.alpha"), [1])
        self.assertEqual(delivered, [{"v": [1]}])

    def test_policy_exception_fails_closed_without_projection(self):
        def broken(_check):
            raise RuntimeError("private grant store inaccessible")
        with self.assertRaises(ReadAccessDenied):
            self.gate(broken).read("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [])

    def test_mid_read_revocation_blocks_return(self):
        rights = {"allowed": True}
        self.ledger.on_project = lambda: rights.update(allowed=False)
        gate = self.gate(lambda _: rights["allowed"])
        with self.assertRaises(ReadAccessDenied):
            gate.read("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [("node-B", "session-A")])

    def test_post_read_revocation_blocks_next_read(self):
        rights = {"allowed": True}
        gate = self.gate(lambda _: rights["allowed"])
        self.assertEqual(gate.read("capabilities.alpha"), {"v": [1]})
        rights["allowed"] = False
        with self.assertRaises(ReadAccessDenied):
            gate.read("capabilities.alpha")
        self.assertEqual(len(self.ledger.calls), 1)

    def test_use_requires_separate_current_use_grant(self):
        rights = {"read": True, "use": False}
        effects = []
        checks = []
        def authorize(check):
            checks.append(check.operation)
            return rights["use"] if check.operation in {"use-pre", "use"} else rights["read"]
        def host_handler(value):
            effects.append(value)
            return value["v"][0]
        gate = self.gate(authorize, {"capabilities.alpha": host_handler})
        with self.assertRaises(ReadAccessDenied):
            gate.use("capabilities.alpha")
        self.assertEqual(effects, [])
        self.assertEqual(checks, ["use-pre"])
        rights["use"] = True
        self.assertEqual(gate.use("capabilities.alpha"), 1)
        self.assertEqual(effects, [{"v": [1]}])
        self.assertEqual(checks[-4:], ["use-pre", "read-pre", "read-post", "use"])

    def test_revoked_after_read_before_use_blocks_effect(self):
        rights = {"allowed": True}
        effects = []
        def authorize(check):
            if check.operation == "read-post":
                rights["allowed"] = False
                return True
            return rights["allowed"]
        with self.assertRaises(ReadAccessDenied):
            self.gate(authorize, {"capabilities.alpha":
                                  lambda value: effects.append(value)}).use("capabilities.alpha")
        self.assertEqual(effects, [])

    def test_caller_cannot_inject_handler_at_invocation(self):
        effects = []
        gate = self.gate(lambda _: True,
                         {"capabilities.alpha": lambda value: value["v"][0]})
        with self.assertRaises(TypeError):
            gate.use("capabilities.alpha", host_handler=lambda _: effects.append("injected"))
        self.assertEqual(effects, [])
        self.assertEqual(self.ledger.calls, [])

    def test_missing_handler_denied_before_projection(self):
        with self.assertRaises(ReadAccessDenied):
            self.gate(lambda _: True).use("capabilities.alpha")
        self.assertEqual(self.ledger.calls, [])

    def test_denied_use_cannot_enumerate_host_handler_registry(self):
        """Unknown and installed handler names are indistinguishable pre-grant."""
        gate = self.gate(lambda _: False,
                         {"capabilities.alpha": lambda value: value["v"][0]})
        errors = []
        for key in ("capabilities.alpha", "capabilities.unknown"):
            with self.assertRaises(ReadAccessDenied) as caught:
                gate.use(key)
            errors.append(str(caught.exception))
        self.assertEqual(errors, ["host policy denied", "host policy denied"])
        self.assertEqual(self.ledger.calls, [])

    def test_host_handler_registry_is_snapshot_at_construction(self):
        called = []
        def approved(value):
            called.append("approved")
            return value["v"][0]
        handlers = {"capabilities.alpha": approved}
        gate = self.gate(lambda _: True, handlers)
        handlers["capabilities.alpha"] = lambda _: called.append("replaced")
        handlers["capabilities.other"] = lambda _: called.append("new")
        self.assertEqual(gate.use("capabilities.alpha"), 1)
        self.assertEqual(called, ["approved"])
        with self.assertRaises(ReadAccessDenied):
            gate.use("capabilities.other")

    def test_constructor_rejects_untrusted_handler_registry_shapes(self):
        for bad in ("some.handler", {"capabilities.alpha": "code"},
                    {"capabilities/invalid": lambda _: True}):
            with self.subTest(bad=str(bad)):
                with self.assertRaises((TypeError, ReadAccessDenied)):
                    self.gate(lambda _: True, bad)

    def test_invalid_or_missing_key_never_leaks_other_values(self):
        gate = self.gate(lambda _: True)
        for key in ("", "a/b", "__secret", "x" * 129, 45):
            with self.subTest(key=key), self.assertRaises(ReadAccessDenied):
                gate.read(key)
        self.assertEqual(self.ledger.calls, [])
        with self.assertRaises(ReadAccessDenied):
            gate.read("capabilities.other")

    def test_oversized_and_invalid_json_projections_denied(self):
        self.ledger.values = {"capabilities.alpha": "X" * 8200}
        with self.assertRaises(ReadAccessDenied):
            self.gate(lambda _: True).read("capabilities.alpha")
        self.ledger.values = {"capabilities.alpha": {"invalid": float("nan")}}
        with self.assertRaises(ReadAccessDenied):
            self.gate(lambda _: True).read("capabilities.alpha")

    def test_constructor_requires_scoped_host_binding_and_policy(self):
        for kwargs in (
            {"context": self.bound, "authorize": None},
            {"context": object(), "authorize": lambda _: True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                HostBoundReadGate(self.ledger, **kwargs)
        for name in ("principal_id", "target_id", "session_id"):
            fields = {"principal_id": "owner", "target_id": "node-B", "session_id": "sess-A"}
            fields[name] = "not valid/spaces"
            with self.subTest(name=name), self.assertRaises(ValueError):
                HostBoundContext(**fields)
        class ImposterStr(str):
            pass
        with self.assertRaises(ValueError):
            HostBoundContext(ImposterStr("operator-A"), "node-B", "session-A")
        with self.assertRaises(ReadAccessDenied):
            self.gate(lambda _: True).read(ImposterStr("capabilities.alpha"))

    def test_raw_ledger_remains_outside_facade_protection(self):
        """A direct DB/API holder can bypass this seam: NOT production ACL."""
        gate = self.gate(lambda _: False)
        with self.assertRaises(ReadAccessDenied):
            gate.read("capabilities.alpha")
        self.assertEqual(self.ledger.project("node-B", "session-A"),
                         {"capabilities.alpha": {"v": [1]}})

    def test_real_sqlite_ledger_integration_but_not_external_authentication(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = StateLedger(Path(tmp) / "bus.sqlite3", clock=lambda: 1500)
            evidence = EvidenceBinding(
                artifact_sha256="a" * 64,
                source_git_commit_oid="sha1:" + "b" * 40,
                source_tree_sha256="c" * 64,
                host_test_receipt_sha256="d" * 64,
                evidence_class=EvidenceClass.TEST,
            )
            item = StateDelta.create(
                event_id="evt-real", source_id="node-A", target_id="node-B",
                session_id="session-A", sequence=1, expected_revision=0,
                operation=StateOp.SET,
                payload={"key": "capabilities.alpha", "value": {"ref": "cap-synthetic"}},
                issued_at_epoch=1400, expires_at_epoch=2000,
                evidence=evidence, provenance_refs=("fixture-a",),
                capability_refs=("synthetic-capability",),
            )
            validation = ValidationReceipt(
                version=1, validation_id="validated-real",
                proposal_hash=item.content_hash, evidence_sha256=evidence.sha256(),
                verifier_id="fixture", verifier_role="independent-verifier",
                decision="approve", evaluated_at_epoch=1500,
            )
            ledger.propose(item, authorize_proposer=lambda _: True)
            ledger.validate(item.event_id, validation, verify_validator=lambda *_: True)
            ledger.commit(item.event_id, verify_validator=lambda *_: True,
                          authorize_commit=lambda *_: True)
            rights = {"operator-A": True}
            gate = HostBoundReadGate(
                ledger, context=self.bound,
                authorize=lambda check: rights.get(check.context.principal_id, False),
            )
            self.assertEqual(gate.read("capabilities.alpha"), {"ref": "cap-synthetic"})
            rights["operator-A"] = False
            with self.assertRaises(ReadAccessDenied):
                gate.read("capabilities.alpha")
            # Deliberate counterexample: direct path holder still bypasses gate.
            self.assertEqual(ledger.project("node-B", "session-A"),
                             {"capabilities.alpha": {"ref": "cap-synthetic"}})


if __name__ == "__main__":
    unittest.main()
