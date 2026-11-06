"""Strict, UNTRUSTED Path A/B transfer proposal envelope for TRAININGDAY.

This module validates source-side syntax and internal SHA-256 references only.
It never authenticates hashes against files, grants rights, executes training,
verifies a host or evaluator, or changes release/admission state. All proposed
permissions and artifact identities remain unverified until an independent
host-owned process performs the corresponding readback.
"""

from __future__ import annotations

import hashlib
import json
import re


PROPOSAL_SCHEMA = "NEUROSYNC_TRANSFER_PROPOSAL_V1"
ENVELOPE_SCHEMA = "NEUROSYNC_TRAININGDAY_TRANSFER_ENVELOPE_V1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_METHODS = ("exact-base-delta", "receiver-native-distillation")
_ARTIFACTS = ("donor_trained", "recipient_base", "train_manifest",
              "sealed_holdout_manifest", "shuffled_control_manifest")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(f"unadmitted transfer proposal: {reason}")


def _keys(value: object, expected: set[str], name: str) -> dict:
    _require(type(value) is dict and set(value) == expected,
             f"{name} missing or unexpected fields")
    return value


def _sha(value: object, name: str) -> str:
    _require(type(value) is str and _HEX.fullmatch(value) is not None,
             f"{name} must be lowercase SHA-256")
    return value


def _label(value: object, name: str) -> str:
    _require(type(value) is str and 1 <= len(value) <= 128 and
             all(char.isprintable() and not 0xD800 <= ord(char) <= 0xDFFF
                 for char in value),
             f"{name} invalid label")
    return value


def propose_transfer(intent: dict) -> dict:
    """Return a content-addressed proposal that is ALWAYS held for host validation.

    Input comes from an untrusted caller, so validated checksums are merely
    declarations; the receiver may not infer any admission or authority.
    """
    intent = _keys(intent, {"schema", "method", "lineage", "input_artifacts",
                            "data_rights", "evaluation", "budget"}, "proposal")
    _require(intent["schema"] == PROPOSAL_SCHEMA, "wrong proposal schema")
    method = intent["method"]
    _require(type(method) is str and method in _METHODS, "unsupported method")

    lineage = _keys(intent["lineage"], {
        "donor_base_sha256", "donor_trained_sha256", "recipient_base_sha256",
        "donor_tokenizer_sha256", "recipient_tokenizer_sha256",
        "donor_config_sha256", "recipient_config_sha256", "adapter_config_sha256"
    }, "lineage")
    for key in lineage:
        if key == "adapter_config_sha256" and method == "receiver-native-distillation":
            _require(lineage[key] is None, "distillation cannot claim adapter transfer")
        else:
            _sha(lineage[key], f"lineage.{key}")
    _require(lineage["donor_base_sha256"] != lineage["donor_trained_sha256"],
             "donor training cannot report unchanged lineage")
    if method == "exact-base-delta":
        _require(lineage["donor_base_sha256"] == lineage["recipient_base_sha256"] and
                 lineage["donor_tokenizer_sha256"] == lineage["recipient_tokenizer_sha256"] and
                 lineage["donor_config_sha256"] == lineage["recipient_config_sha256"],
                 "exact-base donor/receiver base, tokenizer, or config mismatch")

    artifacts = _keys(intent["input_artifacts"], set(_ARTIFACTS), "artifact inventory")
    for role in _ARTIFACTS:
        artifact = _keys(artifacts[role], {"sha256", "size_bytes"}, f"artifact {role}")
        _sha(artifact["sha256"], f"artifact {role}")
        _require(type(artifact["size_bytes"]) is int and artifact["size_bytes"] > 0,
                 f"artifact {role} invalid byte size")
    _require(artifacts["donor_trained"]["sha256"] == lineage["donor_trained_sha256"] and
             artifacts["recipient_base"]["sha256"] == lineage["recipient_base_sha256"],
             "artifact/lineage digest mismatch")
    splits = [artifacts[role]["sha256"] for role in
              ("train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest")]
    _require(len(set(splits)) == len(splits), "train, holdout and control split digests coincide")

    rights = _keys(intent["data_rights"], {
        "record_sha256", "rights_holder_label", "usage_scope"
    }, "data rights")
    _sha(rights["record_sha256"], "rights record")
    _label(rights["rights_holder_label"], "rights holder")
    _require(rights["usage_scope"] in
             ("local-experiment-only", "explicit-permission-proposed"),
             "unsupported rights usage scope")

    evaluation = _keys(intent["evaluation"], {
        "donor_disconnected", "cold_recipient_reload", "baseline_arm",
        "negative_control_arm", "evaluator_custody"
    }, "evaluation")
    _require(evaluation["donor_disconnected"] is True and
             evaluation["cold_recipient_reload"] is True and
             evaluation["baseline_arm"] == "untouched-recipient" and
             evaluation["negative_control_arm"] == "shuffled-donor",
             "evaluation must propose donor-offline cold-reload controls")
    _require(evaluation["evaluator_custody"] == "independent-host-proposed",
             "evaluator custody is only an independent-host proposal")

    budget = _keys(intent["budget"], {"max_external_cost_usd", "max_gpu_minutes"},
                   "budget")
    _require(type(budget["max_external_cost_usd"]) is int and
             budget["max_external_cost_usd"] == 0,
             "external cost must be exactly $0")
    _require(type(budget["max_gpu_minutes"]) is int and
             0 < budget["max_gpu_minutes"] <= 1440,
             "invalid bounded GPU minutes")

    # This is self-integrity only. A forged input has a perfectly valid digest.
    payload = json.dumps(intent, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return {
        "schema": ENVELOPE_SCHEMA,
        "method": method,
        "proposal_sha256_untrusted": hashlib.sha256(payload).hexdigest(),
        "evidence_class": "UNTRUSTED_TRANSFER_PROPOSAL",
        "proposed_artifact_digests": {role: artifacts[role]["sha256"] for role in _ARTIFACTS},
        "proposed_lineage": dict(lineage),
        "proposed_rights_record_sha256": rights["record_sha256"],
        "admission": {
            "host_artifact_verification": "UNVERIFIED",
            "dataset_training_rights": "UNVERIFIED",
            "sealed_independent_evaluation": "UNVERIFIED",
            "protected_evaluator_custody": "UNVERIFIED",
            "optimizer_and_adapter_restore": "UNVERIFIED",
            "cold_recipient_reload": "UNVERIFIED",
            "serving_pid_model_binding": "UNVERIFIED",
            "physical_launch_authority": "HOLD",
            "trainingday_ready": False,
        },
        "interpretation": (
            "Self-declared content-addressed intent only. Similar hashes and a "
            "rights label do not establish source files, permissions, host grants, "
            "real PEFT transfer, independent evaluation, or neural learning."
        ),
    }
