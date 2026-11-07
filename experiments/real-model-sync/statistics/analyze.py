"""Strict scenario-paired analysis of synthetic real-receiver pilot receipts.

This is a LOCAL, untrusted-receipt analysis, not independent evaluation or
weight/process attestation.  Tools and steps are not independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path
import re
from typing import Any


ARMS = ("N", "R", "V", "P", "H", "D", "X")
TASK_COMPARATORS = ("P", "H", "N", "R", "D")
IDENTIFIER = re.compile(r"case-[0-9]{2}-[a-f0-9]{8}\Z")
FULL_ARM_SET = frozenset(ARMS)
VERSION_V0 = "neuro-sync-local-receiver-v0"
VERSION_V1 = "neuro-sync-local-receiver-v1"
EMPTY_STATE_SHA256 = hashlib.sha256(b"{}").hexdigest()


def _int(value: Any) -> bool:
    return type(value) is int


def _ensure(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def _sha(value: Any) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value))


def _validate_v1(data: dict, cases: list[str], grouped: dict[str, dict[str, dict]]) -> dict:
    """Cross-check v1's source-reported oracle strata and request bindings.

    The receipt is untrusted: consistency with these labels and hashes cannot
    independently establish the correctness of the sealed suite or model host.
    """
    labels = data.get("case_oracle_labels")
    _ensure(type(labels) is list and len(labels) == len(cases),
            "missing v1 case oracle labels")
    by_case = {}
    for index, item in enumerate(labels):
        _ensure(type(item) is dict and set(item) == {
            "case", "expected_possible", "oracle_minimum_steps"},
            "invalid v1 oracle label fields")
        step = item["oracle_minimum_steps"]
        _ensure(item["case"] == cases[index] and
                type(item["expected_possible"]) is bool and
                (step is None or (type(step) is int and step in (1, 2))) and
                item["expected_possible"] == (step is not None),
                "v1 oracle labels contradict case order or reachability")
        by_case[item["case"]] = item
    if len(cases) >= 2:
        _ensure([by_case[c]["oracle_minimum_steps"] for c in cases[:2]] == [1, 2],
                "v1 first two feasibility cases must be min-one and min-two")
    if data["evaluation_scope"] == "FULL_LOCAL_PROCEDURE":
        expected = {1: len(cases) // 3, None: len(cases) // 6}
        expected[2] = len(cases) - expected[1] - expected[None]
        _ensure(all(sum(by_case[c]["oracle_minimum_steps"] == step for c in cases) == n
                    for step, n in expected.items()), "v1 oracle stratum count mismatch")

    reference = data.get("prepared_reference_state_sha256_by_case")
    # Older v1 pilots prepare all P/H/V fixtures even on a V-only exploratory
    # smoke. Newer lazy-arm pilots correctly avoid creating unrequested P/H
    # fixtures and report zero comparator parity records. Support both shapes,
    # but never waive P/H/V digest checks when all three arms are selected.
    parity_arms_present = {"P", "H", "V"}.issubset(data["arms_evaluated"])
    has_reference = type(reference) is list and len(reference) == len(cases)
    no_partial_reference = (type(reference) is list and len(reference) == 0 and
                            not parity_arms_present and
                            data["evaluation_scope"] == "EXPLORATORY_INCOMPLETE")
    _ensure(has_reference or no_partial_reference,
            "missing v1 prepared reference state digests")
    if no_partial_reference:
        # Internal iteration records are not persisted or passed off as parity
        # evidence; the per-call request digest checks still run for every row.
        reference = [{"case": case, "state_sha256_by_arm": {}} for case in cases]
    for index, item in enumerate(reference):
        _ensure(type(item) is dict and set(item) == {
            "case", "state_sha256_by_arm"}, "bad v1 prepared digest record")
        digests = item["state_sha256_by_arm"]
        _ensure(item["case"] == cases[index] and type(digests) is dict and
                set(digests) == ({"P", "H", "V"} if has_reference else set()) and
                all(_sha(v) for v in digests.values()) and
                (not has_reference or len(set(digests.values())) == 1),
                "prepared P/H/V reference digests differ or are malformed")
        for arm, row in grouped[item["case"]].items():
            request_digest = row.get("request_shared_state_sha256")
            _ensure(_sha(request_digest), "missing per-call request-state digest")
            if arm in digests:
                _ensure(request_digest == digests[arm],
                        "per-call request digest differs from prepared P/H/V reference")
            if arm == "N":
                _ensure(request_digest == EMPTY_STATE_SHA256,
                        "no-state control request contained external state")
            if row["invalid_reply"]:
                _ensure("abstained" not in row, "invalid reply cannot claim valid abstention")
                continue
            _ensure(type(row.get("abstained")) is bool, "valid v1 reply missing abstained flag")
            if row["abstained"]:
                _ensure(row["effects"] == 0 and not row["gate_tested"] and
                        row["denied_attempts"] == 0,
                        "abstention cannot claim execution or gate testing")
                if arm != "X":
                    _ensure(row["success"] is
                            (not by_case[item["case"]]["expected_possible"]),
                            "v1 abstention score contradicts oracle reachability")
                else:
                    _ensure(not row["success"],
                            "revoked governance cannot succeed from abstention")
            elif arm == "X":
                _ensure(row["gate_tested"] and row["effects"] == 0 and
                        row["denied_attempts"] == 1 and row["success"] and
                        row["governance_denial_success"],
                        "nonabstaining revoked control must attempt blocked tool effect")
            else:
                _ensure(row["gate_tested"] and row["effects"] in (1, 2) and
                        row["denied_attempts"] == 0,
                        "nonabstaining task plan must execute one or two tools")
            if arm != "X" and row["success"]:
                _ensure(row["abstained"] is (not by_case[item["case"]]["expected_possible"]),
                        "successful v1 row claims wrong solution category")
                if row["abstained"] is False:
                    _ensure(row["gate_tested"] and row["effects"] in (1, 2) and
                            row["effects"] >= by_case[item["case"]]["oracle_minimum_steps"]
                            and row.get("oracle_agree") is True,
                            "successful possible solution lacks minimum oracle tool effects")

    published = data.get("scoring_by_arm")
    arms = [a for a in data["arms_evaluated"] if a != "X"]
    _ensure(type(published) is dict and set(published) == set(arms),
            "missing or invalid v1 scoring by arm")
    strata = {}
    for arm in arms:
        possible = [grouped[c][arm] for c in cases if by_case[c]["expected_possible"]]
        impossible = [grouped[c][arm] for c in cases if not by_case[c]["expected_possible"]]
        expected = {
            "possible_cases": len(possible),
            "possible_plans_solved": sum(r["success"] for r in possible),
            "impossible_cases": len(impossible),
            "impossible_correct_abstentions": sum(
                r["success"] and r["abstained"] is True for r in impossible),
        }
        _ensure(type(published[arm]) is dict and set(published[arm]) == set(expected) and
                all(_int(published[arm][k]) and published[arm][k] == v
                    for k, v in expected.items()), "v1 stratum aggregate differs from rows")
        strata[arm] = {
            **expected,
            "possible_plan_success_rate": (round(expected["possible_plans_solved"] /
                                                  len(possible), 6) if possible else None),
            "impossible_correct_abstention_rate": (
                round(expected["impossible_correct_abstentions"] / len(impossible), 6)
                if impossible else None),
        }
    return {"labels": by_case, "task_scores": strata}


def _validate(data: dict) -> tuple[list[str], dict[str, dict[str, dict]], bool]:
    """Reject malformed, unbalanced or internally contradictory receipts."""
    _ensure(type(data) is dict, "receipt must be an object")
    _ensure(data.get("experiment") in (VERSION_V0, VERSION_V1),
            "unexpected experiment version")
    _ensure(data.get("unit") == "scenario-arm, NOT individual tool calls",
            "experimental unit not declared")
    scope = data.get("evaluation_scope")
    _ensure(scope in ("EXPLORATORY_INCOMPLETE", "FULL_LOCAL_PROCEDURE"),
            "missing or invalid evaluation scope")
    evidence = data.get("evidence_class")
    _ensure(evidence in ("SCRIPTED_FIXTURE_TEST", "LOCAL_HTTP_UNATTESTED"),
            "unknown evidence class")
    model = data.get("model_evidence")
    _ensure(type(model) is dict and type(model.get("real_endpoint")) is bool,
            "model evidence declaration required")
    _ensure((evidence == "LOCAL_HTTP_UNATTESTED") == model["real_endpoint"],
            "model metadata disagrees with evidence class")
    n = data.get("scenarios_evaluated")
    _ensure(_int(n) and 1 <= n <= 32, "invalid scenario count")
    arms = data.get("arms_evaluated")
    _ensure(type(arms) is list and arms and all(type(a) is str and a in ARMS for a in arms)
            and len(arms) == len(set(arms)), "invalid evaluated arms")
    eligible_full = n >= 12 and set(arms) == FULL_ARM_SET
    _ensure(scope != "FULL_LOCAL_PROCEDURE" or eligible_full,
            "full-procedure claim contradicts incomplete controls")
    # V1 may evaluate the first 12 cases of a larger sealed 12..32-case suite.
    # The receipt cannot upgrade that truncated selection to full merely
    # because there happen to be >=12 balanced rows.
    _ensure(data["experiment"] == VERSION_V1 or
            scope != "EXPLORATORY_INCOMPLETE" or not eligible_full,
            "exploratory label contradicts full procedure")
    full = eligible_full and scope == "FULL_LOCAL_PROCEDURE"

    rows = data.get("results")
    _ensure(type(rows) is list and len(rows) == n * len(arms),
            "missing, extra or imbalanced result rows")
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        _ensure(type(row) is dict, "result row must be object")
        case, arm = row.get("case"), row.get("arm")
        _ensure(type(case) is str and IDENTIFIER.fullmatch(case) is not None,
                "invalid case ID")
        _ensure(type(arm) is str and arm in arms, "invalid arm label")
        _ensure(arm not in grouped.setdefault(case, {}), "duplicated case/arm pair")
        for key in ("success", "invalid_reply", "gate_tested", "governance_denial_success"):
            _ensure(type(row.get(key)) is bool, f"invalid binary result {key}")
        for key in ("effects", "denied_attempts"):
            _ensure(_int(row.get(key)) and row[key] >= 0, f"invalid nonnegative {key}")
        if row["invalid_reply"]:
            _ensure(not row["success"] and row["effects"] == 0 and
                    row["denied_attempts"] == 0 and not row["gate_tested"],
                    "failed model reply cannot execute or count as success")
        if arm == "X":
            _ensure(row["success"] == row["governance_denial_success"],
                    "X scores permission denial, not task success")
            if row["success"]:
                _ensure(row["gate_tested"] and row["denied_attempts"] >= 1
                        and row["effects"] == 0,
                        "revoked success without an attempted blocked effect")
        else:
            _ensure(not row["governance_denial_success"],
                    "task arm cannot claim revoked governance success")
        grouped[case][arm] = row
    _ensure(len(grouped) == n, "scenario count inconsistent with rows")
    _ensure(all(set(per_case) == set(arms) for per_case in grouped.values()),
            "arms must match on every scenario")
    published = data.get("success_by_arm")
    _ensure(type(published) is dict and set(published) == set(arms),
            "invalid published success counts")
    for arm in arms:
        observed = sum(grouped[case][arm]["success"] for case in grouped)
        _ensure(_int(published[arm]) and published[arm] == observed,
                "published arm count differs from result rows")
    if data["experiment"] == VERSION_V1:
        _validate_v1(data, sorted(grouped), grouped)
    return sorted(grouped), grouped, full


def _exact_p(discordant_a: int, discordant_b: int) -> float:
    """Exact two-sided matched-binary discordance test, conditional on ties."""
    m = discordant_a + discordant_b
    if not m:
        return 1.0
    tail = sum(comb(m, k) for k in range(min(discordant_a, discordant_b) + 1))
    return min(1.0, 2.0 * tail / (2 ** m))


def analyze(receipt: dict) -> dict:
    cases, rows, full = _validate(receipt)
    v1 = receipt["experiment"] == VERSION_V1
    stratified = _validate_v1(receipt, cases, rows) if v1 else None
    arms = receipt["arms_evaluated"]
    count = len(cases)
    metrics = {}
    for arm in arms:
        items = [rows[case][arm] for case in cases]
        success = sum(item["success"] for item in items)
        invalid = sum(item["invalid_reply"] for item in items)
        metrics[arm] = {
            "scenarios": count, "successes": success,
            "observed_success_rate": round(success / count, 6),
            "invalid_replies": invalid, "valid_replies": count - invalid,
            "effects": sum(item["effects"] for item in items),
            "denied_attempts": sum(item["denied_attempts"] for item in items),
            "score_kind": "governance_denial_not_accuracy" if arm == "X"
                          else ("mixed_possible_plan_and_impossible_abstention_rubric"
                                if v1 else "synthetic_task_accuracy"),
        }
        if v1 and arm != "X":
            metrics[arm]["stratified_task_scores"] = stratified["task_scores"][arm]
    paired = []
    if "V" in arms:
        for other in TASK_COMPARATORS:
            if other not in arms:
                continue
            # In v1 impossible-case success is correct abstention, a different
            # skill from planning. Do not pool it into a tool-use comparison.
            compared = ([c for c in cases if stratified["labels"][c]["expected_possible"]]
                        if v1 else cases)
            better = sum(rows[c]["V"]["success"] and not rows[c][other]["success"]
                         for c in compared)
            worse = sum(not rows[c]["V"]["success"] and rows[c][other]["success"]
                        for c in compared)
            paired.append({
                "arm_a": "V", "arm_b": other, "scenarios": len(compared),
                "a_only_success": better, "b_only_success": worse,
                "discordants": better + worse,
                "paired_difference": round((better - worse) / len(compared), 6)
                                     if compared else None,
                "exact_two_sided_p": _exact_p(better, worse),
                "p_value_scope": "uncorrected_exploratory_not_inferential",
                "comparison_stratum": "possible_tool_plans_only" if v1 else "legacy_v0_all_cases",
            })
    # Multiplicity matters even for descriptive exploratory comparisons. Holm
    # controls the five (or fewer) comparisons as one family under its model
    # assumptions, but does NOT repair selection, non-independence or custody.
    cumulative = 0.0
    m = len(paired)
    for rank, index in enumerate(sorted(range(m),
                                       key=lambda k: paired[k]["exact_two_sided_p"])):
        adjusted = min(1.0, (m - rank) * paired[index]["exact_two_sided_p"])
        cumulative = max(cumulative, adjusted)
        paired[index]["holm_familywise_adjusted_p"] = cumulative
    evidence = receipt["evidence_class"]
    qualification = "EXPLORATORY_ONLY"
    if full:
        qualification = ("SOURCE_ONLY_FULL_LOCAL_FIXTURE" if evidence == "SCRIPTED_FIXTURE_TEST"
                         else "FULL_LOCAL_HTTP_UNATTESTED")
    incomplete_responses = any(metric["invalid_replies"] > 0 for metric in metrics.values())
    return {
        "schema_version": 2 if v1 else 1,
        "receipt_experiment_version": receipt["experiment"],
        "analysis_protocol": "V1_STRATIFIED" if v1 else "LEGACY_V0_UNSTRATIFIED",
        "receipt_evidence_class": evidence,
        "qualification": qualification,
        "n_independent_scenario_units_at_most": count,
        "response_completeness": ("INCOMPLETE_MODEL_RESPONSES" if incomplete_responses
                                  else "NO_INVALID_REPLIES_RECORDED"),
        "arms": metrics, "paired_comparisons": paired,
        "statebus_specific_accuracy_advantage_established": False,
        "neural_transfer_established": False,
        "limitations": [
            "Receipt is self-reported JSON without independently authenticated custody",
            "One scenario, not individual tool calls or correlated arm effects, is the sampling unit",
            "Exact p and Holm-adjusted p are exploratory descriptions; no independent preregistration, replication or model attestation",
            "V1 oracle strata and request-state SHA-256 values are source-reported; self-consistency cannot independently verify suite or serving process" if v1 else
            "Legacy v0 lacks reachability-stratum and prepared/request digest fields and cannot be upgraded to v1 evidence",
            "V/P/H share identical model-facing state; a V win over N cannot attribute benefit to StateBus",
            "X primary is postplan effect-denial, incomparable with task-accuracy arms",
            "Exploratory subsets and very small n cannot qualify model performance",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="local pilot receipt.json; read only")
    args = parser.parse_args(argv)
    report = analyze(json.loads(args.receipt.read_text(encoding="utf8")))
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
