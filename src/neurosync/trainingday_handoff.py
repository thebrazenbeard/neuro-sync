"""Fail-closed, source-test-only Neuro-Sync -> TRAININGDAY observation handoff.

The input is a *reported* result of the bounded Trek categorical-SGD pilot.
Cross-consistency checks catch malformed or promoted reports, but neither this
module nor its SHA-256 output can authenticate the caller, host, data rights,
training process, checkpoint bytes, independent evaluator, or physical launch.
No field in the returned payload is admission or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping


SCHEMA = "NEUROSYNC_TRAININGDAY_MECHANICS_HANDOFF_V1"
_QUALIFICATION = "structured-categorical-SGD-mechanics-only; not an LLM or English generalization"
_ARMS = (
    "C0_untouched",
    "C1_full_donor_copy",
    "C2_exact_base_delta",
    "C3_output_distillation",
    "C4_wrong_donor_shuffle",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"unadmitted mechanics report: {reason}")


def _sha(value: object) -> bool:
    return type(value) is str and _HEX64.fullmatch(value) is not None


def mechanics_handoff(result: Mapping) -> dict:
    """Check reported pilot invariants and emit an explicitly unqualified handoff.

    The result MUST NOT be used as trusted physical evidence. It is useful for
    routing measured trial arms to a future independent evaluator without
    mistaking tiny SGD transfer for an official Qwen adapter or protected eval.
    """
    _require(isinstance(result, Mapping), "result is not a mapping")
    _require(result.get("qualification") == _QUALIFICATION, "unexpected qualification")
    _require(result.get("dataset") in ("trek", "synthetic"), "unknown dataset")
    _require(result.get("initial_donor_equals_untouched_recipient") is True,
             "missing strict initial A/B equality")
    total = result.get("n_facts")
    _require(type(total) is int and total > 0, "invalid trial size")
    source_types = result.get("facts_source_types")
    _require(type(source_types) is list and source_types and
             all(type(item) is str and item for item in source_types),
             "missing source-type labels")
    base = result.get("base_sha256")
    trained = result.get("donor_after_sgd_sha256")
    _require(_sha(base) and _sha(trained) and base != trained,
             "missing or unchanged donor/base digest")
    arms = result.get("evals_after_donor_removed")
    updates = result.get("recipient_training")
    _require(isinstance(arms, Mapping) and set(arms) == set(_ARMS),
             "required control arms absent or unexpected")
    _require(isinstance(updates, Mapping) and set(updates) == set(_ARMS),
             "recipient update inventory mismatch")
    _require(all(type(updates[name]) is int and updates[name] >= 0 for name in _ARMS),
             "invalid recipient update count")
    _require(all(updates[name] == 0 for name in _ARMS[:3]) and
             all(updates[name] > 0 for name in _ARMS[3:]),
             "copy and receiver-training arms conflated")

    summarized = {}
    for name in _ARMS:
        observation = arms[name]
        _require(isinstance(observation, Mapping), f"{name} has no observation")
        correct = observation.get("correct")
        _require(observation.get("new_process") is True and
                 observation.get("total") == total and
                 type(correct) is int and 0 <= correct <= total and
                 _sha(observation.get("checkpoint_sha256")) and
                 isinstance(observation.get("predictions"), list) and
                 len(observation["predictions"]) == total and
                 all(type(x) is str for x in observation["predictions"]),
                 f"{name} has no consistent B-only subprocess observation")
        summarized[name] = {
            "correct": correct,
            "total": total,
            "checkpoint_sha256_reported": observation["checkpoint_sha256"],
            "recipient_updates_reported": updates[name],
            "new_process_reported": True,
        }
    _require(arms[_ARMS[0]]["checkpoint_sha256"] == base,
             "untouched control is not reported base")
    _require(arms[_ARMS[1]]["checkpoint_sha256"] == trained ==
             arms[_ARMS[2]]["checkpoint_sha256"],
             "full copy and exact-base delta disagree with trained donor")
    _require(arms[_ARMS[3]]["checkpoint_sha256"] != base,
             "receiver-native distillation reports unchanged B")

    # This digest gives integrity against accidental transformations only;
    # anyone able to mint a result can also mint its digest.
    return {
        "schema": SCHEMA,
        "source_trial": "neuro-sync/experiments/trek-transfer-v0",
        "dataset": result["dataset"],
        "source_type_labels_reported": list(source_types),
        "test_report_sha256_untrusted": _digest(result),
        "evidence_class": "SELF_REPORTED_TEST_RESULT",
        "experiment_claim": _QUALIFICATION,
        "trial_evaluations": summarized,
        "admission": {
            "official_model_base": "UNVERIFIED",
            "dataset_train_dev_rights": "UNVERIFIED",
            "independent_protected_evaluation": "UNVERIFIED",
            "real_weight_adapter_and_optimizer": "UNVERIFIED",
            "cold_reloaded_gguf_serving": "UNVERIFIED",
            "physical_launch_authority": "HOLD",
            "trainingday_ready": False,
        },
        "interpretation": (
            "Categorical-SGD taught-pair pilot only. Reported subprocess != "
            "independent host custody; public source labels != approved rights; "
            "numeric tensor transfer != official Qwen LoRA, GGUF or generalization."
        ),
    }
