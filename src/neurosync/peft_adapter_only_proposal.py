"""Distinct **adapter-only** LoRA transfer intent; NEVER a full checkpoint grant.

Ordinary PEFT saves just adapter parameters, not modified base-model weights.
This proposal treats identical donor/receiver base weights + separate adapter
files as a legitimate candidate without inventing a changed full checkpoint.
Everything is caller-reported; acceptance is syntactic, never host authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from .trainingday_proposal import _keys, _label, _require, _sha


ADAPTER_ONLY_SCHEMA = "NEUROSYNC_PEFT_ADAPTER_ONLY_INTENT_V1"
ENVELOPE_SCHEMA = "NEUROSYNC_TRAININGDAY_PEFT_ADAPTER_ONLY_ENVELOPE_V1"
_REVISION = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")
_TARGET = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\Z")
_ARTIFACTS = ("donor_base", "recipient_base", "adapter_weights", "adapter_config",
              "train_manifest", "sealed_holdout_manifest", "shuffled_control_manifest")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def target_map_sha256(target_map: dict) -> str:
    """Canonical proposed shape fingerprint, compatible with PEFT bind v1.

    This validates a *declared* geometry map, not target dimensions against
    the real recipient's model layers. Those require a host+loader probe.
    """
    target_map = _keys(target_map, {"rank", "lora_alpha", "target_modules",
                                    "lora_pairs"}, "target map")
    rank, alpha = target_map["rank"], target_map["lora_alpha"]
    _require(type(rank) is int and 0 < rank <= 16384, "invalid target rank")
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
             "invalid target alpha")
    targets = target_map["target_modules"]
    _require(type(targets) is list and targets and
             all(type(x) is str and _TARGET.fullmatch(x) for x in targets) and
             len(targets) == len(set(targets)), "invalid target modules")
    pairs = target_map["lora_pairs"]
    _require(type(pairs) is list and pairs, "missing target tensor pairs")
    names, observed = set(), set()
    expected = {"module", "rank", "input_features", "output_features", "dtype", "target"}
    for pair in pairs:
        pair = _keys(pair, expected, "target tensor pair")
        name = pair["module"]
        _require(type(name) is str and _TARGET.fullmatch(name) and name not in names,
                 "duplicate or malformed target module")
        names.add(name)
        _require(type(pair["rank"]) is int and pair["rank"] == rank and
                 type(pair["input_features"]) is int and pair["input_features"] > 0 and
                 type(pair["output_features"]) is int and pair["output_features"] > 0 and
                 type(pair["dtype"]) is str and pair["dtype"] in
                 ("F16", "BF16", "F32", "F64") and
                 type(pair["target"]) is str and pair["target"] in targets,
                 "invalid target shape, dtype or rank")
        matching = [target for target in targets
                    if name == target or name.endswith("." + target)]
        _require(len(matching) == 1 and matching[0] == pair["target"],
                 "missing or ambiguous target suffix")
        observed.add(pair["target"])
    _require(observed == set(targets), "configured target missing tensor pair")
    canonical = {"rank": rank, "lora_alpha": alpha,
                 "target_modules": sorted(targets),
                 "lora_pairs": sorted(pairs, key=lambda item: item["module"])}
    return _canonical_hash(canonical)


def propose_adapter_only_transfer(intent: dict) -> dict:
    """Emit adapter-only Qwen/PEFT *candidate*, never a training/release grant."""
    intent = _keys(intent, {"schema", "method", "base", "adapter", "input_artifacts",
                            "training_provenance", "data_rights", "evaluation", "budget"},
                   "adapter-only intent")
    _require(intent["schema"] == ADAPTER_ONLY_SCHEMA and
             intent["method"] == "peft-adapter-only", "adapter-only schema/method required")
    base = _keys(intent["base"], {"donor_base_weights_sha256", "recipient_base_weights_sha256",
                                  "donor_tokenizer_sha256", "recipient_tokenizer_sha256",
                                  "donor_config_sha256", "recipient_config_sha256",
                                  "base_revision"}, "base lineage")
    for field in set(base) - {"base_revision"}:
        _sha(base[field], f"base.{field}")
    _require(type(base["base_revision"]) is str and
             _REVISION.fullmatch(base["base_revision"]) is not None,
             "immutable-looking base revision required")
    _require(base["donor_base_weights_sha256"] == base["recipient_base_weights_sha256"] and
             base["donor_tokenizer_sha256"] == base["recipient_tokenizer_sha256"] and
             base["donor_config_sha256"] == base["recipient_config_sha256"],
             "exact base/tokenizer/config lineage required for adapter-only")

    adapter = _keys(intent["adapter"], {"weights_sha256", "config_sha256",
                                        "target_map", "target_map_sha256"}, "adapter")
    for field in ("weights_sha256", "config_sha256", "target_map_sha256"):
        _sha(adapter[field], f"adapter.{field}")
    # Same-role donor/recipient frozen base references are deliberately equal.
    # Cross-role files are *not* interchangeable: base weights, tokenizer,
    # model configuration, adapter weights and adapter config have distinct
    # formats and must not be asserted as the same byte object.
    artifact_roles = (
        base["donor_base_weights_sha256"],
        base["donor_tokenizer_sha256"],
        base["donor_config_sha256"],
        adapter["weights_sha256"],
        adapter["config_sha256"],
    )
    _require(len(set(artifact_roles)) == len(artifact_roles),
             "incompatible artifact roles must not alias")
    geometry_sha = target_map_sha256(adapter["target_map"])
    _require(geometry_sha == adapter["target_map_sha256"],
             "target map digest does not match proposed geometry")

    artifacts = _keys(intent["input_artifacts"], set(_ARTIFACTS), "artifact inventory")
    sizes_by_sha: dict[str, int] = {}
    for role in _ARTIFACTS:
        entry = _keys(artifacts[role], {"sha256", "size_bytes"}, f"artifact {role}")
        _sha(entry["sha256"], f"artifact {role}")
        _require(type(entry["size_bytes"]) is int and entry["size_bytes"] > 0,
                 f"artifact {role} invalid size")
        digest, size = entry["sha256"], entry["size_bytes"]
        _require(digest not in sizes_by_sha or sizes_by_sha[digest] == size,
                 f"artifact {role} size contradicts identical digest inventory")
        sizes_by_sha[digest] = size
    bindings = {"donor_base": base["donor_base_weights_sha256"],
                "recipient_base": base["recipient_base_weights_sha256"],
                "adapter_weights": adapter["weights_sha256"],
                "adapter_config": adapter["config_sha256"]}
    _require(all(artifacts[role]["sha256"] == value for role, value in bindings.items()),
             "artifact/lineage/adapter bytes reference mismatch")
    _require(len({artifacts[role]["sha256"] for role in
                 ("train_manifest", "sealed_holdout_manifest",
                  "shuffled_control_manifest")}) == 3,
             "train/holdout/shuffle split collision")
    manifest_digests = tuple(artifacts[role]["sha256"] for role in
                             ("train_manifest", "sealed_holdout_manifest",
                              "shuffled_control_manifest"))
    _require(len(set(artifact_roles + manifest_digests)) ==
             len(artifact_roles) + len(manifest_digests),
             "manifest bytes must not alias weight, tokenizer or config artifact roles")

    provenance = _keys(intent["training_provenance"], {
        "source_revision_sha256", "optimizer_receipt_sha256", "status"
    }, "training provenance")
    _sha(provenance["source_revision_sha256"], "proposed training source")
    _require((provenance["status"] == "UNVERIFIED" and
              provenance["optimizer_receipt_sha256"] is None) or
             (provenance["status"] == "RECEIPT_REFERENCED_UNVERIFIED" and
              type(provenance["optimizer_receipt_sha256"]) is str and
              _sha(provenance["optimizer_receipt_sha256"], "proposed optimizer receipt")),
             "invalid training provenance status/reference")

    rights = _keys(intent["data_rights"], {"record_sha256", "rights_holder_label",
                                           "usage_scope"}, "data rights")
    _sha(rights["record_sha256"], "rights record")
    _label(rights["rights_holder_label"], "rights holder")
    _require(rights["usage_scope"] in
             ("local-experiment-only", "explicit-permission-proposed"),
             "unsupported rights scope")
    evaluation = _keys(intent["evaluation"], {"donor_disconnected",
                                                 "cold_recipient_reload", "baseline_arm",
                                                 "negative_control_arm", "evaluator_custody"},
                       "evaluation")
    _require(evaluation["donor_disconnected"] is True and
             evaluation["cold_recipient_reload"] is True and
             evaluation["baseline_arm"] == "untouched-recipient" and
             evaluation["negative_control_arm"] == "shuffled-donor" and
             evaluation["evaluator_custody"] == "independent-host-proposed",
             "required recipient-only cold reload and negative controls absent")
    budget = _keys(intent["budget"], {"max_external_cost_usd", "max_gpu_minutes"}, "budget")
    _require(type(budget["max_external_cost_usd"]) is int and
             budget["max_external_cost_usd"] == 0,
             "external cost must be exactly $0")
    _require(type(budget["max_gpu_minutes"]) is int and
             0 < budget["max_gpu_minutes"] <= 1440, "invalid GPU time bound")

    return {
        "schema": ENVELOPE_SCHEMA,
        "method": "peft-adapter-only",
        "evidence_class": "UNTRUSTED_PEFT_ADAPTER_ONLY_PROPOSAL",
        "intent_sha256_untrusted": _canonical_hash(intent),
        "proposed_base_weights_sha256": base["donor_base_weights_sha256"],
        "proposed_artifact_digests": {name: artifacts[name]["sha256"] for name in _ARTIFACTS},
        "target_map_sha256_untrusted": geometry_sha,
        "optimizer_receipt_sha256_untrusted": provenance["optimizer_receipt_sha256"],
        "admission": {
            "host_artifact_verification": "UNVERIFIED",
            "dataset_training_rights": "UNVERIFIED",
            "independent_target_layer_compatibility": "UNVERIFIED",
            "optimizer_and_adapter_restore": "UNVERIFIED",
            "sealed_independent_evaluation": "UNVERIFIED",
            "cold_recipient_reload": "UNVERIFIED",
            "serving_pid_model_binding": "UNVERIFIED",
            "physical_launch_authority": "HOLD",
            "trainingday_ready": False,
        },
        "interpretation": (
            "Adapter-only PEFT candidate with unchanged base and distinct adapter bytes; "
            "no changed full checkpoint claimed. Source digests, rights and optimizer "
            "references are self-declared until an independent host verifies real files."
        ),
    }
