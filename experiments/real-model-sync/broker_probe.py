"""Scripted real-receiver plan through the *actual* host-owned effect broker.

This is a local source-mechanics fixture, not a neural model call. Unlike the
pilot's in-memory X arm, it commits synthetic capability state to StateLedger,
parses the same typed plan envelope, and invokes HostOwnedEffectBroker for the
permission decision immediately before a pure arithmetic handler.

There is one in-process host fixture; identities and receipt provenance are
host-asserted. Same-UID direct database or handler access remains a bypass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pilot  # noqa: E402 — reuse real receiver's plan parser/tool oracle
from neurosync.bus import (  # noqa: E402
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger, StateOp,
    ValidationReceipt,
)
from neurosync.bus.effect_gate import (  # noqa: E402
    ApprovedEffect, HostOwnedEffectBroker,
)
from neurosync.bus.read_gate import HostBoundContext, ReadAccessDenied  # noqa: E402


TARGET = "local-receiver"
SESSION = "scripted-broker-probe"
PRINCIPAL = "host-scripted-fixture"
KEY = pilot.KEY
NOW = 1000.0


def _synthetic_case() -> dict[str, Any]:
    """All 13 ordinary pilot operations, deterministic opaque handle mapping."""
    handles = [f"cap-{number:016x}" for number in range(1, len(pilot.OPS) + 1)]
    registry = dict(zip(handles, pilot.OPS))
    return {"start": 2, "goal": 4, "handles": handles,
            "registry": registry, "scripted_handle": handles[0]}


def _commit_state(ledger: StateLedger, value: dict[str, Any]) -> None:
    # The schema-valid sentinels and in-process callbacks below are NOT
    # real artifact attestation or a second independent rights principal.
    evidence = EvidenceBinding(
        artifact_sha256="a" * 64, source_git_commit_oid="sha1:" + "b" * 40,
        source_tree_sha256="c" * 64, host_test_receipt_sha256="d" * 64,
        evidence_class=EvidenceClass.TEST,
    )
    delta = StateDelta.create(
        event_id="broker-scripted-001", source_id="synthetic-fixture",
        target_id=TARGET, session_id=SESSION, sequence=1,
        expected_revision=0, operation=StateOp.SET,
        payload={"key": KEY, "value": value},
        issued_at_epoch=900, expires_at_epoch=2000,
        evidence=evidence, provenance_refs=("synthetic-fixture",),
        capability_refs=("local-arithmetic",),
    )
    receipt = ValidationReceipt(
        version=1, validation_id="validate-scripted-broker-001",
        proposal_hash=delta.content_hash, evidence_sha256=evidence.sha256(),
        verifier_id="same-process-fixture", verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=1000,
    )
    ledger.propose(delta, authorize_proposer=lambda x: x == delta)
    ledger.validate(delta.event_id, receipt,
                    verify_validator=lambda x, r: x == delta and r == receipt)
    ledger.commit(delta.event_id,
                  verify_validator=lambda x, r: x == delta and r == receipt,
                  authorize_commit=lambda x, r: x == delta and r == receipt)


def run_probe(root: Path, *, handler_log: list[dict[str, Any]] | None = None) -> dict:
    """Create isolated SQLite stores and exercise a plan with real broker calls.

    ``root`` must be a new, absent directory so receipts from an older run
    cannot be mistaken for the current invocation. No model/HTTP call occurs.
    """
    root.mkdir(parents=True, exist_ok=False)
    case = _synthetic_case()
    plan_text = json.dumps({"steps": [
        {"handle": case["scripted_handle"], "input": "initial"}],
        "abstain": False})
    selected, abstain = pilot.parse_plan(plan_text, case["handles"])
    if abstain or len(selected) != 1:
        raise AssertionError("scripted plan did not parse as an effect request")
    # Bind the exact validated plan envelope, including the chosen opaque
    # handle and dependency order, to the grant and durable broker receipt.
    # This is host-asserted intent, not independent callback attestation.
    selected_plan_sha256 = hashlib.sha256(plan_text.encode("utf-8")).hexdigest()

    ledger = StateLedger(root / "state.sqlite3", clock=lambda: NOW)
    _commit_state(ledger, {"operations": case["registry"]})
    effects: list[dict[str, Any]] = [] if handler_log is None else handler_log
    clock = [NOW]

    def arithmetic_handler(committed_state: Any) -> dict[str, Any]:
        if type(committed_state) is not dict or set(committed_state) != {"operations"}:
            raise ValueError("wrong committed-state shape")
        mapping = committed_state["operations"]
        value = case["start"]
        for handle in selected:
            operation = mapping[handle]
            value = pilot.apply_tool(operation, value)
        outcome = {"result": value, "steps": len(selected)}
        effects.append(outcome)
        return outcome

    approved = {KEY: ApprovedEffect(
        arithmetic_handler, "a" * 64, "sha1:" + "b" * 40,
    )}
    context = HostBoundContext(PRINCIPAL, TARGET, SESSION)

    def broker(current_context: HostBoundContext = context) -> HostOwnedEffectBroker:
        return HostOwnedEffectBroker(
            root / "authority.sqlite3", ledger, context=current_context,
            approved_handlers=approved, clock=lambda: clock[0],
        )

    owner = broker()
    owner.grant("allowed", KEY, expires_at_epoch=2000,
                plan_sha256=selected_plan_sha256)
    allowed_result, allowed_receipt = owner.execute(
        "allowed", "allowed-use", KEY, plan_sha256=selected_plan_sha256)
    if allowed_result != {"result": case["goal"], "steps": 1}:
        raise AssertionError("executed plan differs from scripted expected effect")

    # The typed plan exists *before* both decisions. A revoked grant must not
    # invoke arithmetic_handler, and never creates an effect attempt claim.
    owner.grant("revoked", KEY, expires_at_epoch=2000,
                plan_sha256=selected_plan_sha256)
    owner.revoke("revoked")
    before_revoked = len(effects)
    try:
        owner.execute("revoked", "revoked-use", KEY,
                      plan_sha256=selected_plan_sha256)
    except ReadAccessDenied as exc:
        revoked_denial = type(exc).__name__
    else:
        raise AssertionError("revoked plan unexpectedly executed")
    revoked_effects = len(effects) - before_revoked
    if revoked_effects:
        raise AssertionError("handler ran after revocation")

    owner.grant("expires", KEY, expires_at_epoch=1001,
                plan_sha256=selected_plan_sha256)
    clock[0] = 1001.0
    before_expiry = len(effects)
    try:
        owner.execute("expires", "expired-use", KEY,
                      plan_sha256=selected_plan_sha256)
    except ReadAccessDenied as exc:
        expiry_denial = type(exc).__name__
    else:
        raise AssertionError("expired plan unexpectedly executed")
    expired_effects = len(effects) - before_expiry
    if expired_effects:
        raise AssertionError("handler ran after expiry")
    clock[0] = NOW

    # Reopen the same durable authority and reissue the exact invocation ID.
    try:
        broker().execute("allowed", "allowed-use", KEY,
                         plan_sha256=selected_plan_sha256)
    except ReadAccessDenied as exc:
        replay_denial = type(exc).__name__
    else:
        raise AssertionError("replayed plan unexpectedly executed")
    if len(effects) != 1:
        raise AssertionError("unexpected additional handler effect")

    foreign = broker(HostBoundContext("other-principal", TARGET, SESSION))
    try:
        foreign.execute("allowed", "foreign-use", KEY,
                        plan_sha256=selected_plan_sha256)
    except ReadAccessDenied as exc:
        foreign_denial = type(exc).__name__
    else:
        raise AssertionError("cross-principal grant unexpectedly inherited")
    if len(effects) != 1:
        raise AssertionError("foreign context invoked handler")

    reloaded = broker()
    report = {
        "schema_version": 1,
        "evidence_class": "LOCAL_SCRIPTED_BROKER_FIXTURE_TEST",
        "model_called": False,
        "plan_source": "literal_scripted_typed_json_not_neural",
        "plan_receipt_binding": "host_asserted_selected_plan_digest_and_handler_descriptor",
        "grant_scope": "state_key_selected_plan_digest_and_host_asserted_handler_descriptor",
        "actual_broker": "HostOwnedEffectBroker",
        "state_source": "committed_StateLedger",
        "state_evidence": "schema_valid_sentinels_not_independently_attested",
        "scripted_plan_sha256": selected_plan_sha256,
        "parsed_steps": len(selected),
        "allowed": {"handler_result": allowed_result,
                    "receipt_status": allowed_receipt["status"],
                    "receipt_binding": allowed_receipt["artifact_binding_kind"],
                    "plan_sha256": allowed_receipt["plan_sha256"],
                    "intent_binding_kind": allowed_receipt["intent_binding_kind"],
                    "durable_attempt_status": reloaded.attempt("allowed-use")["status"]},
        "revoked_after_plan": {
            "error": revoked_denial, "new_effects": revoked_effects,
            "durable_attempt": reloaded.attempt("revoked-use")},
        "expired_after_plan": {
            "error": expiry_denial, "new_effects": expired_effects,
            "durable_attempt": reloaded.attempt("expired-use")},
        "replayed_after_reopen": {"error": replay_denial,
                                   "effects_total": len(effects)},
        "foreign_context": {"error": foreign_denial,
                            "durable_attempt": foreign.attempt("foreign-use")},
        "handler_effects": effects,
        "limits": [
            "one scripted plan, not model-produced inference",
            "same-UID caller with direct broker-handler or database access bypasses controls",
            "single-process fixture role names do not authenticate independent principals",
            "SQLite protects invocation claims, not exactly-once external effects",
            "revoked or expired before first claim yields no durable attempt receipt",
            "selected-plan SHA-256 and handler descriptors are asserted by the same host fixture",
            "same-UID host can alter callback logic while reusing an asserted handler SHA-256",
            "broker object reopen occurs in one process, not a crash or process restart",
        ],
    }
    if report["revoked_after_plan"]["new_effects"] != 0 or \
            report["expired_after_plan"]["new_effects"] != 0:
        raise AssertionError("forbidden handler effect")
    return report


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ns-scripted-broker-") as directory:
        report = run_probe(Path(directory) / "fixture")
        print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
