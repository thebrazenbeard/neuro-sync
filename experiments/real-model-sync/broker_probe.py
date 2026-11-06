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
    ApprovedPlanEffect, HostOwnedEffectBroker,
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


def _commit_state(ledger: StateLedger, value: dict[str, Any], *, revision: int = 0) -> None:
    # The schema-valid sentinels and in-process callbacks below are NOT
    # real artifact attestation or a second independent rights principal.
    evidence = EvidenceBinding(
        artifact_sha256="a" * 64, source_git_commit_oid="sha1:" + "b" * 40,
        source_tree_sha256="c" * 64, host_test_receipt_sha256="d" * 64,
        evidence_class=EvidenceClass.TEST,
    )
    delta = StateDelta.create(
        event_id=f"broker-scripted-{revision + 1:03d}", source_id="synthetic-fixture",
        target_id=TARGET, session_id=SESSION, sequence=revision + 1,
        expected_revision=revision, operation=StateOp.SET,
        payload={"key": KEY, "value": value},
        issued_at_epoch=900, expires_at_epoch=2000,
        evidence=evidence, provenance_refs=("synthetic-fixture",),
        capability_refs=("local-arithmetic",),
    )
    receipt = ValidationReceipt(
        version=1, validation_id=f"validate-scripted-broker-{revision + 1:03d}",
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
    plan = json.loads(plan_text)
    selected, abstain = pilot.parse_plan(plan_text, case["handles"])
    if abstain or len(selected) != 1:
        raise AssertionError("scripted plan did not parse as an effect request")

    ledger = StateLedger(root / "state.sqlite3", clock=lambda: NOW)
    _commit_state(ledger, {"operations": case["registry"]})
    effects: list[dict[str, Any]] = [] if handler_log is None else handler_log
    clock = [NOW]
    clock_observations = [clock[0]]

    def arithmetic_handler(committed_state: Any, approved_plan: Any) -> dict[str, Any]:
        if type(committed_state) is not dict or set(committed_state) != {"operations"}:
            raise ValueError("wrong committed-state shape")
        # A plan-aware handler consumes the checked plan passed by the broker,
        # not a previously captured handle held outside the authority path.
        approved_handles, is_abstention = pilot.parse_plan(
            json.dumps(approved_plan, allow_nan=False), case["handles"])
        if is_abstention or len(approved_handles) != 1:
            raise ValueError("approved plan is not one executable tool step")
        mapping = committed_state["operations"]
        value = case["start"]
        for handle in approved_handles:
            operation = mapping[handle]
            value = pilot.apply_tool(operation, value)
        outcome = {"result": value, "steps": len(approved_handles)}
        effects.append(outcome)
        return outcome

    approved = {KEY: ApprovedPlanEffect(
        arithmetic_handler, "a" * 64, "sha1:" + "b" * 40,
    )}
    context = HostBoundContext(PRINCIPAL, TARGET, SESSION)

    def broker(current_context: HostBoundContext = context, *,
               override: ApprovedPlanEffect | None = None) -> HostOwnedEffectBroker:
        return HostOwnedEffectBroker(
            root / "authority.sqlite3", ledger, context=current_context,
            approved_handlers={},
            approved_plan_handlers={KEY: override or approved[KEY]},
            clock=lambda: clock[0],
        )

    owner = broker()
    owner.grant_plan("allowed", KEY, plan=plan, expires_at_epoch=2000)
    allowed_result, allowed_receipt = owner.execute_plan(
        "allowed", "allowed-use", KEY, plan=plan)
    if allowed_result != {"result": case["goal"], "steps": 1}:
        raise AssertionError("executed plan differs from scripted expected effect")

    # The typed plan exists *before* both decisions. A revoked grant must not
    # invoke arithmetic_handler, and never creates an effect attempt claim.
    owner.grant_plan("revoked", KEY, plan=plan, expires_at_epoch=2000)
    owner.revoke("revoked")
    before_revoked = len(effects)
    try:
        owner.execute_plan("revoked", "revoked-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        revoked_denial = type(exc).__name__
    else:
        raise AssertionError("revoked plan unexpectedly executed")
    revoked_effects = len(effects) - before_revoked
    if revoked_effects:
        raise AssertionError("handler ran after revocation")

    owner.grant_plan("expires", KEY, plan=plan, expires_at_epoch=1001)
    clock[0] = 1001.0
    clock_observations.append(clock[0])
    before_expiry = len(effects)
    try:
        owner.execute_plan("expires", "expired-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        expiry_denial = type(exc).__name__
    else:
        raise AssertionError("expired plan unexpectedly executed")
    expired_effects = len(effects) - before_expiry
    if expired_effects:
        raise AssertionError("handler ran after expiry")
    # Do not rewind the host clock for later negative cases. A plan authority
    # with a durable observed-time floor must reject that synthetic rollback,
    # which would test the clock instead of the intended grant behavior.

    # Reopen the same durable authority and reissue the exact invocation ID.
    try:
        broker().execute_plan("allowed", "allowed-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        replay_denial = type(exc).__name__
    else:
        raise AssertionError("replayed plan unexpectedly executed")
    if len(effects) != 1:
        raise AssertionError("unexpected additional handler effect")

    foreign = broker(HostBoundContext("other-principal", TARGET, SESSION))
    try:
        foreign.execute_plan("allowed", "foreign-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        foreign_denial = type(exc).__name__
    else:
        raise AssertionError("cross-principal grant unexpectedly inherited")
    if len(effects) != 1:
        raise AssertionError("foreign context invoked handler")

    # The same state key must not license another of the 13 typed handles.
    other_plan = {"steps": [{"handle": case["handles"][1],
                             "input": "initial"}], "abstain": False}
    owner.grant_plan("exact-only", KEY, plan=plan, expires_at_epoch=2000)
    before_mismatch = len(effects)
    try:
        owner.execute_plan("exact-only", "other-handle", KEY, plan=other_plan)
    except ReadAccessDenied as exc:
        mismatch_denial = type(exc).__name__
    else:
        raise AssertionError("an unapproved handle executed under the plan grant")
    mismatch_effects = len(effects) - before_mismatch
    if mismatch_effects:
        raise AssertionError("mismatched plan reached host handler")

    # A plan-specific grant is one-use: another invocation ID is not a retry.
    try:
        broker().execute_plan("allowed", "second-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        repeat_denial = type(exc).__name__
    else:
        raise AssertionError("one-use plan grant unexpectedly allowed another effect")

    # A changed host-asserted handler artifact is rejected for the same plan.
    def changed_handler(_state: Any, _plan: Any) -> dict[str, Any]:
        raise AssertionError("changed handler should never be called")
    altered_broker = broker(override=ApprovedPlanEffect(
        changed_handler, "c" * 64, "sha1:" + "b" * 40))
    before_change = len(effects)
    try:
        altered_broker.execute_plan("exact-only", "changed-handler", KEY, plan=plan)
    except ReadAccessDenied as exc:
        handler_denial = type(exc).__name__
    else:
        raise AssertionError("changed artifact unexpectedly inherited plan authorization")
    changed_handler_effects = len(effects) - before_change
    if changed_handler_effects:
        raise AssertionError("changed artifact reached the handler")

    # Revisions are part of the grant's exact approved input. This is a real
    # second StateLedger commit, not mutation of an in-memory mock projection.
    owner.grant_plan("stale-state", KEY, plan=plan, expires_at_epoch=2000)
    changed_state = {"operations": case["registry"], "revision_marker": 2}
    _commit_state(ledger, changed_state, revision=1)
    before_stale = len(effects)
    try:
        owner.execute_plan("stale-state", "stale-use", KEY, plan=plan)
    except ReadAccessDenied as exc:
        stale_denial = type(exc).__name__
    else:
        raise AssertionError("changed state unexpectedly passed bound plan grant")
    stale_effects = len(effects) - before_stale
    if stale_effects:
        raise AssertionError("stale state reached the handler")

    reloaded = broker()
    clock_observations.append(clock[0])
    report = {
        "schema_version": 2,
        "evidence_class": "LOCAL_SCRIPTED_BROKER_FIXTURE_TEST",
        "model_called": False,
        "plan_source": "literal_scripted_typed_json_not_neural",
        "plan_receipt_binding": "broker_verified_canonical_plan_sha256_host_asserted",
        "grant_scope": "exact_plan_state_and_handler_one_use",
        "actual_broker": "HostOwnedEffectBroker",
        "state_source": "committed_StateLedger",
        "state_evidence": "schema_valid_sentinels_not_independently_attested",
        "synthetic_plan_clock_epochs": clock_observations,
        # The literal JSON contains spaces; the receipt hashes sorted compact
        # JSON and must be compared to a separately computed canonical digest.
        "wire_plan_text_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
        "parsed_steps": len(selected),
        "allowed": {"handler_result": allowed_result,
                    "receipt_status": allowed_receipt["status"],
                    "receipt_version": allowed_receipt["version"],
                    "receipt_plan_sha256": allowed_receipt["plan_sha256"],
                    "receipt_binding": allowed_receipt["artifact_binding_kind"],
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
        "unapproved_same_key_plan": {
            "error": mismatch_denial, "new_effects": mismatch_effects,
            "durable_attempt": reloaded.attempt("other-handle")},
        "plan_grant_second_use": {
            "error": repeat_denial, "durable_attempt": reloaded.attempt("second-use")},
        "changed_handler_artifact": {
            "error": handler_denial, "new_effects": changed_handler_effects,
            "durable_attempt": reloaded.attempt("changed-handler")},
        "changed_committed_state": {
            "error": stale_denial, "new_effects": stale_effects,
            "durable_attempt_status": reloaded.attempt("stale-use")["status"]},
        "handler_effects": effects,
        "limits": [
            "one scripted plan, not model-produced inference",
            "same-UID caller with direct broker-handler or database access bypasses controls",
            "single-process fixture role names do not authenticate independent principals",
            "SQLite protects invocation claims, not exactly-once external effects",
            "revoked or expired before first claim yields no durable attempt receipt",
            "plan handler artifact and provenance are host asserted, not independently attested",
            "a same-UID host can substitute a different callable while falsely reusing an artifact digest",
            "a trusted host must independently authorize each plan; exact binding is not semantic correctness",
            "wall-clock expiry can be undermined by clock rollback absent a monotonic authority",
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
