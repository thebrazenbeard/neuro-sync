"""Qwen adapter-reload comparison INTENT validator. Always a source-only HOLD.

This is the receiver-side counterpart of model-training's pinned source-shard
bundle. It does not fetch shards, reload PEFT adapters, attest PIDs, train
weights, or adjudicate evaluation custody.
"""

import hashlib
import json
import re

SCHEMA = "NEUROSYNC_QWEN35_PEFT_TRIAD_INTENT_V1"
SOURCE_SCHEMA = "TRAININGDAY_OFFICIAL_SOURCE_SHARD_BUNDLE_V1"
HOLDOUT_SCHEMA = "NEUROSYNC_BLIND_DIALOGUE_V2"
OFFICIAL_ID = "Qwen/Qwen3.5-4B"
OFFICIAL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SHARDS = (
    ("model.safetensors-00001-of-00002.safetensors", 5329398688,
     "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"),
    ("model.safetensors-00002-of-00002.safetensors", 3990429408,
     "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"),
)
ROLES = {"untouched": "untouched-recipient",
         "candidate": "candidate-adapter",
         "shuffled_control": "shuffled-donor-control"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def expected_source_bundle():
    """Exact source metadata compatible with model-training; NOT model bytes."""
    record = {
        "schema": SOURCE_SCHEMA,
        "repository": OFFICIAL_ID,
        "revision": OFFICIAL_REVISION,
        "ordered_shards": [{"path": path, "size_bytes": size, "sha256_lfs_oid": sha}
                           for path, size, sha in SHARDS],
    }
    record["manifest_sha256"] = hashlib.sha256(
        b"TRAININGDAY_OFFICIAL_SOURCE_SHARD_BUNDLE_V1\x00" +
        _canonical(record).encode("ascii")).hexdigest()
    return record


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], value) for key, value in expected.items())
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _hexsha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_triad_intent(holdout_manifest, source_bundle, intent):
    """Check a proposed triad's internal lineage, return NO launch qualification.

    Every digest on the intent is caller-declared. Successful validation never
    grants rights, upgrades source metadata to a read-back, or verifies a model.
    """
    if not _exact(source_bundle, expected_source_bundle()):
        raise ValueError("official source bundle disagrees with pinned metadata")
    if (type(holdout_manifest) is not dict or
            set(holdout_manifest) != {
                "schema", "trial_id", "arms", "requests_sha256",
                "private_commitment_hmac_sha256", "independent_custody",
                "trained_identity_proven"} or
            holdout_manifest["schema"] != HOLDOUT_SCHEMA or
            holdout_manifest["arms"] != list(ROLES) or
            holdout_manifest["trained_identity_proven"] is not False or
            holdout_manifest["independent_custody"] != "UNVERIFIED" or
            type(holdout_manifest["trial_id"]) is not str or
            re.fullmatch(r"[0-9a-f]{32}", holdout_manifest["trial_id"]) is None or
            not _hexsha(holdout_manifest["requests_sha256"]) or
            not _hexsha(holdout_manifest["private_commitment_hmac_sha256"])):
        raise ValueError("invalid or overclaimed holdout manifest")
    if (type(intent) is not dict or set(intent) != {
                "schema", "holdout_manifest_sha256", "source_bundle_manifest_sha256", "arms"} or
            intent["schema"] != SCHEMA or
            intent["holdout_manifest_sha256"] != digest(holdout_manifest) or
            intent["source_bundle_manifest_sha256"] != source_bundle["manifest_sha256"] or
            type(intent["arms"]) is not dict or set(intent["arms"]) != set(ROLES)):
        raise ValueError("unbound Qwen triad intent")
    base_hashes = set()
    adapter_hashes = set()
    adapter_configs = set()
    for name, expected_role in ROLES.items():
        arm = intent["arms"][name]
        if (type(arm) is not dict or set(arm) != {
                "role", "base_weight_bytes_sha256", "adapter_weights_sha256",
                "adapter_config_sha256"} or
                arm["role"] != expected_role or
                not _hexsha(arm["base_weight_bytes_sha256"])):
            raise ValueError("malformed role/base byte declaration")
        base_hashes.add(arm["base_weight_bytes_sha256"])
        if name == "untouched":
            if arm["adapter_weights_sha256"] is not None or arm["adapter_config_sha256"] is not None:
                raise ValueError("untouched base must not load any adapter")
        else:
            if not _hexsha(arm["adapter_weights_sha256"]) or not _hexsha(arm["adapter_config_sha256"]):
                raise ValueError("adapter arm requires exact byte digest declarations")
            adapter_hashes.add(arm["adapter_weights_sha256"])
            adapter_configs.add(arm["adapter_config_sha256"])
    if len(base_hashes) != 1 or len(adapter_hashes) != 2:
        raise ValueError("triad must share one base and have two distinct adapters")
    source_metadata_hashes = {source_bundle["manifest_sha256"],
                              *(item["sha256_lfs_oid"] for item in source_bundle["ordered_shards"])}
    if base_hashes & source_metadata_hashes:
        raise ValueError("metadata or single shard hash cannot represent full model bytes")
    if any(adapter in base_hashes or adapter in source_metadata_hashes
           for adapter in adapter_hashes):
        raise ValueError("adapter hash aliases base or source metadata")
    if adapter_configs & (base_hashes | adapter_hashes | source_metadata_hashes):
        raise ValueError("adapter config hash aliases incompatible byte or metadata kind")
    return {
        "schema": SCHEMA,
        "holdout_manifest_sha256": digest(holdout_manifest),
        "source_bundle_manifest_sha256": source_bundle["manifest_sha256"],
        "artifact_roles": ROLES.copy(),
        "declared_intent_sha256": digest(intent),
        "source_evidence_class": "PINNED_UPSTREAM_LFS_METADATA_ONLY",
        "base_weight_bytes_verified": False,
        "adapter_reload": "UNVERIFIED",
        "adapter_optimizer_receipt": "UNVERIFIED",
        "control_label_corruption": "UNVERIFIED",
        "donor_disconnection": "UNVERIFIED",
        "cold_restart": "UNVERIFIED",
        "serving_pid_model_binding": "UNVERIFIED",
        "independent_holdout_custody": "UNVERIFIED",
        "training_cutoff_attestation": "UNVERIFIED",
        "dataset_rights": "UNVERIFIED",
        "physical_launch_authority": "HOLD",
        "trainingday_ready": False,
    }
