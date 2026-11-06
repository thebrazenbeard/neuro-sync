"""Reproducible local StateLedger-vs-file *mechanics* comparison.

All three arms use the identical host-owned effect broker. One file arm is
intentionally an *ungoverned* JSON projection; the other adds limited
same-process policy checks. No arm authenticates a real remote principal.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from neurosync.bus import (  # noqa: E402
    AuthorizationError, ConflictError, EvidenceBinding, EvidenceClass,
    StateDelta, StateLedger, StateOp, ValidationReceipt,
)
from neurosync.bus.effect_gate import ApprovedEffect, HostOwnedEffectBroker  # noqa: E402
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied  # noqa: E402


KEY = "capabilities.synthetic"
TARGET = "receiver"
SESSION = "trial"
PRINCIPAL = "host"
INITIAL = {"ref": "fixture-capability", "inputs": [2, 3]}
UNAPPROVED = {"ref": "unapproved", "inputs": [50, 50]}
STALE = {"ref": "stale", "inputs": [40, 60]}
APPROVED = {"ref": "approved", "inputs": [5, 7]}


class PlainFileProjection:
    """A scoped JSON snapshot with *no* write authorization or revision CAS."""

    def __init__(self, path: Path):
        self.path = path

    def set(self, target: str, session: str, value: Any) -> None:
        # A deliberately ungoverned baseline; not a recommended writer.
        self.path.write_text(json.dumps({
            "target": target, "session": session, "state": {KEY: value},
        }, sort_keys=True), encoding="utf-8")

    def project(self, target: str, session: str) -> dict[str, Any]:
        item = json.loads(self.path.read_text(encoding="utf-8"))
        if (item["target"], item["session"]) != (target, session):
            return {}
        return item["state"]


class GovernedFileProjection:
    """Single-writer test adapter giving a JSON file the *same* policy rules.

    This minimal comparison adapter is NOT a concurrent, crash-safe or
    separately authenticated file service. It deliberately keeps the
    revision in process and cannot claim persistence after restart.
    """

    def __init__(self, path: Path):
        self._file = PlainFileProjection(path)
        self._file.set(TARGET, SESSION, INITIAL)
        self._revision = 1

    def project(self, target: str, session: str) -> dict[str, Any]:
        return self._file.project(target, session)

    def publish(self, value: Any, *, authorized: bool,
                expected_revision: int) -> None:
        if authorized is not True:
            raise AuthorizationError("producer not authorized")
        if expected_revision != self._revision:
            raise ConflictError("revision CAS mismatch")
        self._file.set(TARGET, SESSION, value)
        self._revision += 1


def _evidence() -> EvidenceBinding:
    # Test-only shape-valid sentinels. These are NOT independent artifact,
    # commit-tree, host-test or rights proofs.
    return EvidenceBinding(
        artifact_sha256="a" * 64, source_git_commit_oid="sha1:" + "b" * 40,
        source_tree_sha256="c" * 64, host_test_receipt_sha256="d" * 64,
        evidence_class=EvidenceClass.TEST,
    )


def _delta(*, event: str, sequence: int, expected_revision: int,
           value: Any) -> StateDelta:
    return StateDelta.create(
        event_id=event, source_id="source", target_id=TARGET,
        session_id=SESSION, sequence=sequence,
        expected_revision=expected_revision, operation=StateOp.SET,
        payload={"key": KEY, "value": value},
        issued_at_epoch=900, expires_at_epoch=2000,
        evidence=_evidence(), provenance_refs=("synthetic-test",),
        capability_refs=("synthetic-cap",),
    )


def _validation(item: StateDelta) -> ValidationReceipt:
    # This asserts a test callback, not an external verifier's endorsement.
    return ValidationReceipt(
        version=1, validation_id="validate-" + item.event_id,
        proposal_hash=item.content_hash,
        evidence_sha256=item.evidence.sha256(), verifier_id="fixture-verifier",
        # Schema requires the literal role token. This does NOT make the
        # in-process always-True fixture callback independent.
        verifier_role="independent-verifier", decision="approve",
        evaluated_at_epoch=1000,
    )


def _commit(ledger: StateLedger, item: StateDelta) -> None:
    ledger.propose(item, authorize_proposer=lambda _x: True)
    ledger.validate(item.event_id, _validation(item),
                    verify_validator=lambda _x, _y: True)
    ledger.commit(item.event_id, verify_validator=lambda _x, _y: True,
                  authorize_commit=lambda _x, _y: True)


def _state(root: Path, arm: str) -> StateLedger | PlainFileProjection | GovernedFileProjection:
    if arm == "statebus":
        ledger = StateLedger(root / "state.sqlite3", clock=lambda: 1000)
        _commit(ledger, _delta(event="initial", sequence=1,
                               expected_revision=0, value=INITIAL))
        return ledger
    if arm == "file":
        file = PlainFileProjection(root / "snapshot.json")
        file.set(TARGET, SESSION, INITIAL)
        return file
    if arm == "governed_file":
        return GovernedFileProjection(root / "guarded.json")
    raise ValueError(arm)


def _broker_controls(root: Path, arm: str) -> dict[str, Any]:
    state = _state(root, arm)
    now = [1000.0]
    effects: list[Any] = []
    def handler(data):
        effects.append(data)
        return {"sum": sum(data["inputs"])}
    approved = {KEY: ApprovedEffect(
        handler, "a" * 64, "sha1:" + "b" * 40,
    )}  # Both fields are test sentinels, never verified production bindings.
    context = HostBoundContext(PRINCIPAL, TARGET, SESSION)
    path = root / "broker.sqlite3"

    def broker(ctx=context):
        return HostOwnedEffectBroker(
            path, state, context=ctx, approved_handlers=approved,
            clock=lambda: now[0],
        )

    owner = broker()
    owner.grant("live", KEY, expires_at_epoch=2000)

    # Probe foreign-principal isolation BEFORE the first use consumes the
    # grant. Otherwise even a broker lacking principal isolation could appear
    # to block the foreign caller solely because the grant was already spent.
    foreign = broker(HostBoundContext("other-principal", TARGET, SESSION))
    try:
        foreign.execute("live", "foreign", KEY)
        scope_blocked = False
    except ReadAccessDenied:
        scope_blocked = True

    # The rightful owner must still be able to use the grant. This also
    # shows the foreign denial was not caused by spending the grant first.
    result, receipt = owner.execute("live", "first", KEY)
    allowed = result == {"sum": 5} and receipt["status"] == "ok"

    try:
        broker().execute("live", "first", KEY)
        replay_blocked = False
    except ReadAccessDenied:
        replay_blocked = True

    owner.grant("revoked", KEY, expires_at_epoch=2000)
    owner.revoke("revoked")
    try:
        owner.execute("revoked", "new-revoked", KEY)
        revoked_blocked = False
    except ReadAccessDenied:
        revoked_blocked = True

    owner.grant("expires", KEY, expires_at_epoch=1001)
    now[0] = 1001.0
    try:
        owner.execute("expires", "new-expired", KEY)
        expired_blocked = False
    except ReadAccessDenied:
        expired_blocked = True
    now[0] = 1000.0

    return {
        "allowed": allowed,
        "replay_blocked_after_reopen": replay_blocked,
        "revoked_blocked": revoked_blocked,
        "expired_blocked": expired_blocked,
        "foreign_context_blocked": scope_blocked,
        "owner_succeeded_after_foreign_probe": allowed,
        "handler_effect_count": len(effects),
        "receipt_status": receipt["status"],
        "receipt_binding": receipt["artifact_binding_kind"],
        "consumed_record_survived_reopen": broker().attempt("first")["status"] == "ok",
    }


def _mutation(root: Path, arm: str, kind: str) -> dict[str, Any]:
    state = _state(root, arm)
    replacement = {"unapproved": UNAPPROVED, "stale": STALE,
                   "approved": APPROVED}[kind]
    if arm == "file":
        assert isinstance(state, PlainFileProjection)
        state.set(TARGET, SESSION, replacement)
        blocked = False
    elif arm == "governed_file":
        assert isinstance(state, GovernedFileProjection)
        try:
            state.publish(replacement, authorized=kind != "unapproved",
                          expected_revision=0 if kind == "stale" else 1)
            blocked = False
        except (AuthorizationError, ConflictError):
            blocked = True
    else:
        assert isinstance(state, StateLedger)
        next_item = _delta(event="next", sequence=2,
                           expected_revision=0 if kind == "stale" else 1,
                           value=replacement)
        try:
            if kind == "unapproved":
                state.propose(next_item, authorize_proposer=lambda _x: False)
            elif kind in ("stale", "approved"):
                _commit(state, next_item)
            else:
                raise ValueError(kind)
            blocked = False
        except (AuthorizationError, ConflictError):
            blocked = True
    observed = state.project(TARGET, SESSION)[KEY]
    return {
        "write_blocked": blocked,
        "baseline_state_preserved": observed == INITIAL,
        "replacement_visible": observed == replacement,
    }


def run() -> dict[str, Any]:
    """Run bounded tests against real local SQLite and JSON file adapters."""
    with tempfile.TemporaryDirectory(prefix="neuro-sync-governance-") as tmp:
        root = Path(tmp)
        policy = {}
        changes = {"approved": {}, "unapproved": {}, "stale": {}}
        for arm in ("statebus", "file", "governed_file"):
            gate_dir = root / (arm + "-gate")
            gate_dir.mkdir()
            policy[arm] = _broker_controls(gate_dir, arm)
            for intervention in changes:
                subdir = root / (arm + "-" + intervention)
                subdir.mkdir()
                changes[intervention][arm] = _mutation(subdir, arm, intervention)
    parity = policy["statebus"] == policy["file"] == policy["governed_file"]
    report = {
        "schema_version": 1,
        "evidence_class": "LOCAL_TEST_SOURCE_ONLY",
        "test_rights_and_receipts": "synthetic_sentinels_not_attested",
        "model_called": False,
        "effect_broker_code_shared": True,
        "comparison": "statebus_vs_uncontrolled_file_vs_single_writer_governed_file",
        "broker_controls": policy,
        "broker_policy_parity": parity,
        "data_mutation_interventions": changes,
        "limitations": [
            "No authenticated remote principal, third-party verifier, or signed anchor",
            "Same-UID process can bypass broker/database and alter both stores",
            "File baseline lacks producer validation and revision CAS by design",
            "The governed-file test adapter matches both checks but lacks cross-process atomicity",
            "No LLM behavior, model-weight transfer, or live network integration tested",
        ],
    }
    # Self-check fail closed so CLI 'success' actually requires expected controls.
    expected_gate = {
        "allowed": True, "replay_blocked_after_reopen": True,
        "revoked_blocked": True, "expired_blocked": True,
        "foreign_context_blocked": True, "handler_effect_count": 1,
        "owner_succeeded_after_foreign_probe": True,
        "receipt_status": "ok", "receipt_binding": "host_asserted_unverified",
        "consumed_record_survived_reopen": True,
    }
    if not (parity and policy["statebus"] == expected_gate):
        raise AssertionError("broker control regression")
    for name in changes:
        blocked = name != "approved"
        for governed_arm in ("statebus", "governed_file"):
            if changes[name][governed_arm] != {
                "write_blocked": blocked, "baseline_state_preserved": blocked,
                "replacement_visible": not blocked,
            }:
                raise AssertionError(f"governance regression in {governed_arm}")
        if changes[name]["file"] != {
            "write_blocked": False, "baseline_state_preserved": False,
            "replacement_visible": True,
        }:
            raise AssertionError("ungoverned file baseline changed")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
