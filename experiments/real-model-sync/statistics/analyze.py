"""Strict scenario-paired analysis of synthetic real-receiver pilot receipts.

This is a LOCAL, untrusted-receipt analysis, not independent evaluation or
weight/process attestation.  Tools and steps are not independent samples.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import re
from typing import Any


ARMS = ("N", "R", "V", "P", "H", "D", "X")
TASK_COMPARATORS = ("P", "H", "N", "R", "D")
IDENTIFIER = re.compile(r"case-[0-9]{2}-[a-f0-9]{8}\Z")
FULL_ARM_SET = frozenset(ARMS)
VERSION = "neuro-sync-local-receiver-v0"


def _int(value: Any) -> bool:
    return type(value) is int


def _ensure(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def _validate(data: dict) -> tuple[list[str], dict[str, dict[str, dict]], bool]:
    """Reject malformed, unbalanced or internally contradictory receipts."""
    _ensure(type(data) is dict, "receipt must be an object")
    _ensure(data.get("experiment") == VERSION, "unexpected experiment version")
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
    full = n >= 12 and set(arms) == FULL_ARM_SET
    _ensure(scope != "FULL_LOCAL_PROCEDURE" or full,
            "full-procedure claim contradicts incomplete controls")
    _ensure(scope != "EXPLORATORY_INCOMPLETE" or not full,
            "exploratory label contradicts full procedure")

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
                          else "synthetic_task_accuracy",
        }
    paired = []
    if "V" in arms:
        for other in TASK_COMPARATORS:
            if other not in arms:
                continue
            better = sum(rows[c]["V"]["success"] and not rows[c][other]["success"]
                         for c in cases)
            worse = sum(not rows[c]["V"]["success"] and rows[c][other]["success"]
                        for c in cases)
            paired.append({
                "arm_a": "V", "arm_b": other, "scenarios": count,
                "a_only_success": better, "b_only_success": worse,
                "discordants": better + worse,
                "paired_difference": round((better - worse) / count, 6),
                "exact_two_sided_p": _exact_p(better, worse),
                "p_value_scope": "uncorrected_exploratory_not_inferential",
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
        "schema_version": 1,
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
