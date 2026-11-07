"""Offline, fail-closed receipt audit for a preregistered real-receiver pilot.

This does not invoke a model, attest a server, or establish neural transfer.
The donor in pilot.py is scripted; all results are at most local-HTTP observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PILOT = HERE / "pilot.py"
FEASIBILITY_ARMS = ("V",)
CAUSAL_ARMS = ("N", "P", "V")
CASE_COUNT = 12
FEASIBILITY_CASES = 2
TIMEOUT_SECONDS = 25
MAX_TOKENS = 64
PROMPT_PROFILE = "compact"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha(value: object) -> bool:
    return (type(value) is str and len(value) == 64 and
            all(ch in "0123456789abcdef" for ch in value))


def _configured(receipt: dict, arms: tuple[str, ...], count: int) -> list[str]:
    """Check comparability and the *source-reported* local adapter evidence."""
    faults: list[str] = []
    if receipt.get("evidence_class") != "LOCAL_HTTP_UNATTESTED":
        faults.append("not_local_http_evidence")
    if receipt.get("evaluation_scope") != "EXPLORATORY_INCOMPLETE":
        faults.append("unexpected_scope")
    if receipt.get("arms_evaluated") != list(arms):
        faults.append("wrong_arms")
    if receipt.get("scenarios_evaluated") != count:
        faults.append("wrong_case_count")
    metadata = receipt.get("model_evidence")
    if not isinstance(metadata, dict) or metadata.get("real_endpoint") is not True:
        faults.append("no_real_endpoint_adapter_report")
        return faults
    config = metadata.get("request_config", {})
    if (not isinstance(config, dict) or
            config.get("timeout_seconds") != TIMEOUT_SECONDS or
            config.get("max_tokens") != MAX_TOKENS or
            config.get("prompt_profile") != PROMPT_PROFILE):
        faults.append("unmatched_generation_budget")
    if metadata.get("source_file_sha256") != _hash_file(PILOT):
        faults.append("pilot_source_bytes_differ")
    if not _is_sha(metadata.get("suite_self_commitment_sha256")):
        faults.append("missing_suite_digest")
    model_id = metadata.get("model_id")
    backend = metadata.get("backend")
    if (type(model_id) is not str or not model_id.strip() or len(model_id) > 200 or
            any(ord(ch) < 32 for ch in model_id) or type(backend) is not str or
            backend not in ("ollama", "openai")):
        faults.append("missing_model_runtime_id")
    endpoint = metadata.get("endpoint")
    if type(endpoint) is not str or len(endpoint) > 200:
        faults.append("invalid_local_endpoint_claim")
    else:
        try:
            parsed = urlsplit(endpoint)
            expected_path = ("/api/chat" if backend == "ollama" else
                             "/v1/chat/completions" if backend == "openai" else None)
            if (parsed.scheme != "http" or
                    parsed.hostname not in ("localhost", "127.0.0.1", "::1") or
                    parsed.port is None or parsed.username or parsed.password or
                    parsed.query or parsed.fragment or parsed.path != expected_path):
                faults.append("invalid_local_endpoint_claim")
        except ValueError:
            faults.append("invalid_local_endpoint_claim")
    if metadata.get("serving_process_file_binding_verified") is not False:
        faults.append("unsupported_process_binding_claim")
    before = metadata.get("supplied_model_file_sha256_before")
    after = metadata.get("supplied_model_file_sha256_after")
    if before is not None or after is not None:
        if not (_is_sha(before) and _is_sha(after) and before == after):
            faults.append("supplied_model_file_changed_or_unverifiable")
    return faults


def _records(receipt: dict, arms: tuple[str, ...], count: int) -> tuple[list[dict], list[str]]:
    rows = receipt.get("results")
    if not isinstance(rows, list) or len(rows) != count * len(arms):
        return [], ["wrong_number_of_records"]
    seen: set[tuple[str, str]] = set()
    cases: dict[str, set[str]] = {}
    faults: list[str] = []
    observed_order: list[str] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            faults.append("malformed_result")
            continue
        case, arm = row.get("case"), row.get("arm")
        if (type(case) is not str or not re.fullmatch(r"case-\d{2}-[0-9a-f]{8}", case) or
                type(arm) is not str or arm not in arms):
            faults.append("bad_case_or_arm")
            continue
        key = (case, arm)
        if key in seen:
            faults.append("duplicate_case_arm")
        seen.add(key)
        if case not in cases:
            observed_order.append(case)
        cases.setdefault(case, set()).add(arm)
        if (type(row.get("success")) is not bool or
                type(row.get("invalid_reply")) is not bool):
            faults.append("malformed_score")
            continue
        # The pilot writes each scenario as a contiguous, cyclically shifted
        # arm block, in sealed-suite order. A sorted-by-arm result isn't the
        # preregistered counterbalanced experiment.
        case_index = row_index // len(arms)
        position = row_index % len(arms)
        expected_arm = arms[(case_index + position) % len(arms)]
        if (case != f"case-{case_index:02d}-{case.rsplit('-', 1)[-1]}" or
                arm != expected_arm):
            faults.append("non_counterbalanced_arm_order")
        if row["invalid_reply"]:
            if (row["success"] is not False or row.get("gate_tested") is not False or
                    row.get("effects") != 0 or row.get("denied_attempts") != 0):
                faults.append("invalid_reply_claims_success_or_effect")
        else:
            if (type(row.get("abstained")) is not bool or
                    type(row.get("gate_tested")) is not bool or
                    type(row.get("effects")) is not int or row["effects"] < 0 or
                    type(row.get("denied_attempts")) is not int or row["denied_attempts"] != 0):
                faults.append("malformed_valid_reply_effects")
            elif row["abstained"]:
                if row["gate_tested"] or row["effects"] != 0:
                    faults.append("abstention_claims_execution")
            elif not row["gate_tested"] or row["effects"] not in (1, 2):
                faults.append("nonabstention_without_effect")
            if row["success"] and row.get("abstained") is False and row.get("oracle_agree") is not True:
                faults.append("success_without_oracle_agreement")
        elapsed = row.get("model_call_elapsed_ms")
        if type(elapsed) not in (int, float) or not 0 <= elapsed <= TIMEOUT_SECONDS * 1000 + 1000:
            faults.append("unbounded_or_bad_caller_latency")
        trace = row.get("transport_trace")
        if trace is not None:
            if not isinstance(trace, dict):
                faults.append("invalid_transport_trace")
            elif (trace.get("timeout_seconds") != TIMEOUT_SECONDS or
                  trace.get("max_tokens") != MAX_TOKENS or
                  trace.get("prompt_profile") != PROMPT_PROFILE or
                  not _is_sha(trace.get("prompt_sha256")) or
                  type(trace.get("request_bytes")) is not int or
                  not 0 < trace["request_bytes"] <= 100_000):
                faults.append("unmatched_per_request_budget_or_trace")
        if row["invalid_reply"] is False:
            if (not isinstance(trace, dict) or trace.get("outcome") != "received_text" or
                    not _is_sha(trace.get("response_sha256"))):
                faults.append("missing_real_text_response_trace")
    if len(cases) != count or any(v != set(arms) for v in cases.values()):
        faults.append("unpaired_or_missing_cases")
    if len(observed_order) != count or any(
            not case.startswith(f"case-{i:02d}-") for i, case in enumerate(observed_order)):
        faults.append("case_sequence_not_frozen_order")
    success_summary = receipt.get("success_by_arm")
    if (type(success_summary) is not dict or
            success_summary != {arm: sum(row.get("success") is True for row in rows
                                         if isinstance(row, dict) and row.get("arm") == arm)
                                for arm in arms}):
        faults.append("success_aggregate_inconsistent")
    return rows, sorted(set(faults))


def _typed_nonabstaining(row: dict) -> bool:
    """A received typed plan, not merely a successful process/receipt write."""
    trace = row.get("transport_trace")
    return bool(row.get("invalid_reply") is False and row.get("abstained") is False and
                row.get("gate_tested") is True and isinstance(trace, dict) and
                trace.get("outcome") == "received_text" and
                _is_sha(trace.get("prompt_sha256")) and
                type(row.get("model_call_elapsed_ms")) in (float, int) and
                0 <= row["model_call_elapsed_ms"] <= TIMEOUT_SECONDS * 1000)


def evaluate(feasibility: dict | None, causal: dict | None = None) -> dict:
    """Audit model-receiver feasibility and three-arm source-level comparability.

    All inputs are untrusted receipts; a PASS does not independently attest
    model provenance, donor identity, neural weight stability or fairness.
    """
    limitations = [
        "SOURCE_CONSISTENCY_CHECK_ONLY: caller-supplied JSON is not host attestation",
        "Pilot donor is scripted; no model-to-model knowledge transfer is tested",
        "P/H/V state-map parity exists in pilot source, not in signed per-call readbacks",
        "No independently authenticated receiver process, weights or cold restart",
    ]
    if feasibility is None:
        return {"decision": "NOT_RUN", "stage": "feasibility", "reasons": ["no_feasibility_receipt"],
                "limitations": limitations}
    faults = _configured(feasibility, FEASIBILITY_ARMS, FEASIBILITY_CASES)
    rows, row_faults = _records(feasibility, FEASIBILITY_ARMS, FEASIBILITY_CASES)
    faults += row_faults
    if faults:
        return {"decision": "NOT_FEASIBLE", "stage": "feasibility",
                "reasons": sorted(set(faults)), "limitations": limitations}
    valid = sum(_typed_nonabstaining(row) for row in rows)
    if valid != FEASIBILITY_CASES:
        return {"decision": "NOT_FEASIBLE", "stage": "feasibility",
                "valid_typed_nonabstaining": valid, "required": FEASIBILITY_CASES,
                "reasons": ["not_every_feasibility_request_produced_typed_plan"],
                "limitations": limitations}
    if causal is None:
        return {"decision": "NOT_RUN", "stage": "three_arm",
                "feasibility": "FEASIBLE_TO_ATTEMPT", "reasons": ["no_three_arm_receipt"],
                "limitations": limitations}
    faults = _configured(causal, CAUSAL_ARMS, CASE_COUNT)
    rows, row_faults = _records(causal, CAUSAL_ARMS, CASE_COUNT)
    faults += row_faults
    fm, cm = feasibility["model_evidence"], causal.get("model_evidence", {})
    for field in ("model_id", "backend", "endpoint", "source_file_sha256"):
        if not isinstance(cm, dict) or fm.get(field) != cm.get(field):
            faults.append("changed_" + field)
    # The pilot cannot bind bytes to the serving process. But when model-file
    # hashes were provided, even their weaker same-file continuity must hold.
    if isinstance(cm, dict):
        previous = fm.get("supplied_model_file_sha256_after")
        current = cm.get("supplied_model_file_sha256_before")
        if previous is not None or current is not None:
            if previous != current:
                faults.append("model_file_changed_between_stages")
    if (isinstance(cm, dict) and
            fm.get("suite_self_commitment_sha256") == cm.get("suite_self_commitment_sha256")):
        faults.append("feasibility_and_trial_reuse_suite")
    if {row["case"] for row in feasibility["results"]} & {
            row["case"] for row in rows if isinstance(row, dict) and
            type(row.get("case")) is str}:
        faults.append("feasibility_and_trial_reuse_case_ids")
    if faults:
        return {"decision": "NOT_FEASIBLE", "stage": "three_arm",
                "reasons": sorted(set(faults)), "limitations": limitations}
    valid_by_arm = {arm: sum(_typed_nonabstaining(r) for r in rows if r["arm"] == arm)
                    for arm in CAUSAL_ARMS}
    successes = {arm: sum(r["success"] for r in rows if r["arm"] == arm)
                 for arm in CAUSAL_ARMS}
    if any(valid_by_arm[arm] == 0 for arm in CAUSAL_ARMS):
        return {"decision": "NOT_FEASIBLE", "stage": "three_arm",
                "reasons": ["one_or_more_arms_have_zero_valid_typed_plans"],
                "valid_typed_by_arm": valid_by_arm, "success_by_arm": successes,
                "limitations": limitations}
    return {"decision": "DESCRIPTIVE_ONLY", "stage": "three_arm",
            "valid_typed_by_arm": valid_by_arm, "success_by_arm": successes,
            "paired_cases": CASE_COUNT,
            "reasons": ["This comparison cannot prove donor-neural learning or a StateBus-specific accuracy gain"],
            "limitations": limitations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility", type=Path, help="source-reported local-HTTP V-only receipt")
    parser.add_argument("--three-arm", type=Path, help="N/P/V paired receipt, distinct sealed suite")
    args = parser.parse_args(argv)
    if args.three_arm and not args.feasibility:
        parser.error("--three-arm requires --feasibility")
    try:
        def load(path: Path | None) -> dict | None:
            if path is None:
                return None
            if path.stat().st_size > 4_000_000:
                raise ValueError("receipt exceeds bounded size")
            value = json.loads(path.read_text(encoding="utf8"))
            if not isinstance(value, dict):
                raise ValueError("receipt is not an object")
            return value
        result = evaluate(load(args.feasibility), load(args.three_arm))
        print(json.dumps(result, sort_keys=True))
        # Success means the requested stage's audit ran and its explicit
        # feasibility gate cleared; a failed gate is not a successful CI job.
        if result["decision"] == "NOT_FEASIBLE":
            return 1
        if result["decision"] == "NOT_RUN" and result.get("feasibility") != "FEASIBLE_TO_ATTEMPT":
            return 1
        return 0
    except (OSError, ValueError, UnicodeError, RecursionError, TypeError) as exc:
        print(json.dumps({"decision": "NOT_FEASIBLE", "reasons": [type(exc).__name__]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
