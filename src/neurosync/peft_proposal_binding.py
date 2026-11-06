"""Fail-closed, **untrusted** joining of Path A adapter inspection and intent.

All input is caller controlled. This only detects contradictory self-reports;
it cannot authenticate physical files, rights, execution or TRAININGDAY grants.
The distinct donor checkpoint and adapter-weight roles must never be aliased.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


SCHEMA = "NEUROSYNC_PEFT_PROPOSAL_BINDING_V1"
CLAIM_SCHEMA = "NEUROSYNC_PEFT_ARTIFACT_BINDING_CLAIM_V1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_INSPECTION_SCHEMA = "NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1"
_INTENT_SCHEMA = "NEUROSYNC_TRANSFER_PROPOSAL_V1"
_ENVELOPE_SCHEMA = "NEUROSYNC_TRAININGDAY_TRANSFER_ENVELOPE_V1"
_CLAIM_FIELDS = {"schema", "donor_trained_checkpoint_sha256",
                 "adapter_weights_sha256", "adapter_config_sha256",
                 "target_map_sha256", "rights_record_sha256"}
_ADMISSION_FIELDS = {"host_artifact_verification", "dataset_training_rights",
                     "sealed_independent_evaluation", "protected_evaluator_custody",
                     "optimizer_and_adapter_restore", "cold_recipient_reload",
                     "serving_pid_model_binding", "physical_launch_authority",
                     "trainingday_ready"}
_INSPECTED_FIELDS = {"schema", "candidate_only", "trainingday_ready", "evidence_class",
                     "adapter_sha256", "config_sha256", "base_model_name_or_path",
                     "base_revision", "rank", "lora_alpha", "lora_scaling",
                     "target_modules", "lora_pairs", "holds"}
_INTENT_FIELDS = {"schema", "method", "lineage", "input_artifacts", "data_rights",
                  "evaluation", "budget"}
_ENVELOPE_FIELDS = {"schema", "method", "proposal_sha256_untrusted",
                    "evidence_class", "proposed_artifact_digests", "proposed_lineage",
                    "proposed_rights_record_sha256", "admission", "interpretation"}
_ARTIFACT_ROLES = {"donor_trained", "recipient_base", "train_manifest",
                   "sealed_holdout_manifest", "shuffled_control_manifest"}


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"unqualified PEFT proposal binding: {message}")


def _sha(value: object, field: str) -> str:
    _fail(type(value) is str and _SHA.fullmatch(value) is not None,
          f"{field} must be lowercase SHA-256")
    return value


def _canonical_sha(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError("unqualified PEFT proposal binding: malformed JSON claim") from exc
    return hashlib.sha256(encoded).hexdigest()


def _target_map(inspection: dict) -> dict:
    rank = inspection.get("rank")
    alpha = inspection.get("lora_alpha")
    targets = inspection.get("target_modules")
    pairs = inspection.get("lora_pairs")
    _fail(type(rank) is int and rank > 0, "missing inspected LoRA rank")
    _fail(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
          "invalid inspected LoRA alpha")
    _fail(type(targets) is list and targets and
          all(type(t) is str and t for t in targets) and
          len(targets) == len(set(targets)),
          "invalid inspected target modules")
    _fail(type(inspection.get("lora_scaling")) in (int, float) and
          inspection["lora_scaling"] == alpha / rank,
          "incorrect inspected ordinary LoRA scaling")
    _fail(type(pairs) is list and pairs, "missing inspected A/B tensor map")
    expected_keys = {"module", "rank", "input_features", "output_features",
                     "dtype", "target"}
    names = set()
    observed_targets = set()
    for pair in pairs:
        _fail(type(pair) is dict and set(pair) == expected_keys,
              "unsupported tensor map entry")
        name = pair["module"]
        _fail(type(name) is str and name and name not in names,
              "duplicate or missing module name")
        names.add(name)
        _fail(type(pair["rank"]) is int and pair["rank"] == rank and
              type(pair["input_features"]) is int and pair["input_features"] > 0 and
              type(pair["output_features"]) is int and pair["output_features"] > 0 and
              pair["dtype"] in ("BF16", "F16", "F32", "F64") and
              type(pair["target"]) is str and pair["target"] in targets,
              "invalid rank/dtype/shape/target")
        _fail(name == pair["target"] or name.endswith("." + pair["target"]),
              "module does not match declared target suffix")
        observed_targets.add(pair["target"])
    _fail(observed_targets == set(targets), "unpaired target in inspected map")
    return {"rank": rank, "lora_alpha": alpha,
            "target_modules": sorted(targets),
            "lora_pairs": sorted(pairs, key=lambda x: x["module"])}


def bind_peft_inspection_to_proposal(
    inspection: dict, intent: dict, envelope: dict, claim: dict
) -> dict:
    """Join two separately self-reported proposals; always HOLD real transfer.

    The explicit supplemental claim provides the *missing* adapter-weights
    artifact role without reusing #37's full donor checkpoint digest. A host
    must independently verify every physical role before execution.
    """
    _fail(type(inspection) is dict and set(inspection) == _INSPECTED_FIELDS and
          inspection.get("schema") == _INSPECTION_SCHEMA and
          inspection.get("evidence_class") == "LOCAL_FILE_INSPECTION_SELF_REPORTED" and
          inspection.get("candidate_only") is True and
          inspection.get("trainingday_ready") is False,
          "inspector is not an unpromoted Path A candidate")
    _fail(type(inspection.get("base_model_name_or_path")) is str and
          bool(inspection["base_model_name_or_path"].strip()) and
          type(inspection.get("base_revision")) is str and
          re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?",
                       inspection["base_revision"]) is not None and
          type(inspection.get("holds")) is list and inspection["holds"] and
          all(type(h) is str and h for h in inspection["holds"]),
          "inspector missing declared base reference or HOLD reasons")
    _fail(type(intent) is dict and set(intent) == _INTENT_FIELDS and
          intent.get("schema") == _INTENT_SCHEMA and
          intent.get("method") == "exact-base-delta", "requires Path A delta intent")
    _fail(type(envelope) is dict and set(envelope) == _ENVELOPE_FIELDS and
          envelope.get("schema") == _ENVELOPE_SCHEMA and
          envelope.get("method") == "exact-base-delta" and
          envelope.get("evidence_class") == "UNTRUSTED_TRANSFER_PROPOSAL",
          "incorrect source-only proposal envelope")
    _fail(type(claim) is dict and set(claim) == _CLAIM_FIELDS and
          claim["schema"] == CLAIM_SCHEMA, "missing or unexpected typed artifact roles")
    for name in _CLAIM_FIELDS - {"schema"}:
        _sha(claim[name], f"binding claim {name}")

    _fail(type(intent.get("lineage")) is dict and
          type(intent.get("input_artifacts")) is dict and
          type(intent.get("data_rights")) is dict,
          "missing proposed lineage/artifact/rights roles")
    lineage = intent["lineage"]
    for field in ("donor_base_sha256", "recipient_base_sha256",
                  "donor_tokenizer_sha256", "recipient_tokenizer_sha256",
                  "donor_config_sha256", "recipient_config_sha256",
                  "donor_trained_sha256", "adapter_config_sha256"):
        _sha(lineage.get(field), f"lineage {field}")
    _fail(lineage["donor_base_sha256"] == lineage["recipient_base_sha256"] and
          lineage["donor_tokenizer_sha256"] == lineage["recipient_tokenizer_sha256"] and
          lineage["donor_config_sha256"] == lineage["recipient_config_sha256"],
          "exact base, tokenizer and model config digests disagree")
    _fail(lineage["donor_trained_sha256"] != lineage["donor_base_sha256"],
          "donor checkpoint appears unchanged")
    artifacts = intent["input_artifacts"]
    _fail(set(artifacts) == _ARTIFACT_ROLES and all(
        type(item) is dict and set(item) == {"sha256", "size_bytes"} and
        type(item["size_bytes"]) is int and item["size_bytes"] > 0 and
        type(item["sha256"]) is str and _SHA.fullmatch(item["sha256"])
        for item in artifacts.values()), "invalid full-checkpoint/split artifact roles")
    _fail(artifacts["donor_trained"].get("sha256") == lineage["donor_trained_sha256"] and
          artifacts["recipient_base"].get("sha256") == lineage["recipient_base_sha256"],
          "full checkpoint artifact/lineage mismatch")
    _fail(len({artifacts[role]["sha256"] for role in
              ("train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest")}) == 3,
          "train, holdout and shuffled splits are not distinct")

    _fail(_sha(inspection.get("adapter_sha256"), "inspected adapter weights") ==
          claim["adapter_weights_sha256"], "adapter weights differ from inspected bytes")
    _fail(_sha(inspection.get("config_sha256"), "inspected adapter config") ==
          claim["adapter_config_sha256"] == lineage["adapter_config_sha256"],
          "adapter config differs from inspected/proposed bytes")
    _fail(claim["donor_trained_checkpoint_sha256"] ==
          lineage["donor_trained_sha256"], "trained full checkpoint role mismatch")
    _fail(claim["adapter_weights_sha256"] !=
          claim["donor_trained_checkpoint_sha256"],
          "adapter-only and full checkpoint artifact roles must be distinct")
    target_digest = _canonical_sha(_target_map(inspection))
    _fail(claim["target_map_sha256"] == target_digest,
          "inspected rank/dtype/shape/target map mismatch")

    rights = intent["data_rights"]
    _fail(set(rights) == {"record_sha256", "rights_holder_label", "usage_scope"} and
          type(rights.get("rights_holder_label")) is str and
          bool(rights["rights_holder_label"].strip()) and
          rights.get("usage_scope") in ("local-experiment-only", "explicit-permission-proposed"),
          "invalid proposed rights labels/scope")
    rights_hash = _sha(rights.get("record_sha256"), "rights record")
    _fail(rights_hash == claim["rights_record_sha256"] ==
          envelope.get("proposed_rights_record_sha256"),
          "rights record role mismatch")
    _fail(type(envelope.get("proposed_lineage")) is dict and
          envelope["proposed_lineage"] == lineage and
          type(envelope.get("proposed_artifact_digests")) is dict and
          envelope["proposed_artifact_digests"] == {
              name: value["sha256"] for name, value in artifacts.items()
          }, "proposal roles do not match intent")
    _fail(envelope.get("proposal_sha256_untrusted") == _canonical_sha(intent),
          "proposal digest differs from source intent")

    evaluation = intent.get("evaluation")
    _fail(type(evaluation) is dict and set(evaluation) == {
        "donor_disconnected", "cold_recipient_reload", "baseline_arm",
        "negative_control_arm", "evaluator_custody"
    } and evaluation.get("donor_disconnected") is True and
          evaluation.get("cold_recipient_reload") is True and
          evaluation.get("baseline_arm") == "untouched-recipient" and
          evaluation.get("negative_control_arm") == "shuffled-donor" and
          evaluation.get("evaluator_custody") == "independent-host-proposed",
          "required donor-disconnected controlled evaluation not proposed")
    budget = intent.get("budget")
    _fail(type(budget) is dict and set(budget) == {
        "max_external_cost_usd", "max_gpu_minutes"
    } and type(budget.get("max_external_cost_usd")) is int and
          budget["max_external_cost_usd"] == 0 and
          type(budget.get("max_gpu_minutes")) is int and
          0 < budget["max_gpu_minutes"] <= 1440,
          "invalid $0 external budget or GPU limit")

    admission = envelope.get("admission")
    _fail(type(admission) is dict and set(admission) == _ADMISSION_FIELDS and
          admission.get("trainingday_ready") is False and
          admission.get("physical_launch_authority") == "HOLD" and
          all(admission[key] == "UNVERIFIED" for key in _ADMISSION_FIELDS - {
              "trainingday_ready", "physical_launch_authority"
          }), "proposal attempts to promote authority")
    return {
        "schema": SCHEMA,
        "evidence_class": "SELF_REPORTED_CROSS_CONSISTENCY_ONLY",
        "adapter_sha256_untrusted": claim["adapter_weights_sha256"],
        "adapter_config_sha256_untrusted": claim["adapter_config_sha256"],
        "donor_trained_checkpoint_sha256_untrusted": claim["donor_trained_checkpoint_sha256"],
        "target_map_sha256_untrusted": target_digest,
        "rights_record_sha256_untrusted": rights_hash,
        "proposal_sha256_untrusted": envelope["proposal_sha256_untrusted"],
        "binding_claim_sha256_untrusted": _canonical_sha(claim),
        "physical_launch_authority": "HOLD",
        "trainingday_ready": False,
        "holds": [
            "unverified adapter and complete donor checkpoint file bytes",
            "unverified base weights/tokenizer/config/target math and PEFT loader",
            "unverified rights, host leases, sealed evaluator and donor-disconnected cold reload",
        ],
    }
