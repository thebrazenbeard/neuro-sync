"""Read-only, stdlib-only PEFT LoRA artifact inspection (never a transfer grant).

This intentionally supports only simple two-matrix LoRA safetensors. It parses
no tensor values and never imports torch, PEFT, pickle, or executable metadata.
The output is a candidate proposal, not an authorization or a model restore.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct


SCHEMA = "NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1"
MAX_HEADER = 1_000_000
MAX_CONFIG = 1_000_000
MAX_TENSORS = 20_000
_FLOAT_DTYPES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_LORA_NAME = re.compile(r"(?P<module>[A-Za-z0-9_.]+)\.lora_(?P<side>[AB])(?:\.default)?\.weight")
_IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?")
_SIMPLE_MODULE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")


def _unique_dict(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_json(raw: bytes) -> dict:
    try:
        obj = json.loads(raw.decode("utf8"), object_pairs_hook=_unique_dict,
                         parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("invalid UTF-8 or recursive JSON") from exc
    if type(obj) is not dict:
        raise ValueError("expected JSON object")
    return obj


def _local_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {name}")
    return path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config(path: Path) -> dict:
    size = path.stat().st_size
    if not 0 < size <= MAX_CONFIG:
        raise ValueError("adapter config size outside bounded range")
    config = _strict_json(path.read_bytes())
    if config.get("peft_type") != "LORA":
        raise ValueError("only ordinary LoRA PEFT type is supported")
    if not (type(config.get("base_model_name_or_path")) is str and
            config["base_model_name_or_path"].strip()):
        raise ValueError("base model reference required")
    revision = config.get("revision")
    if type(revision) is not str or not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("immutable base revision commit digest required")
    rank = config.get("r")
    if type(rank) is not int or not 0 < rank <= 16384:
        raise ValueError("positive integer LoRA rank required")
    alpha = config.get("lora_alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("finite positive LoRA alpha required")
    targets = config.get("target_modules")
    if (type(targets) is not list or not targets or
            any(type(item) is not str or not _SIMPLE_MODULE.fullmatch(item) for item in targets) or
            len(targets) != len(set(targets))):
        raise ValueError("non-empty exact module target list required")
    if config.get("bias") != "none":
        raise ValueError("only adapter-only bias=none supported")
    if config.get("fan_in_fan_out", False) is not False:
        raise ValueError("transposed fan_in_fan_out orientation requires a separate verifier")
    # Specialized variants require their own independently reviewed math and
    # keys. Reject rather than silently treating their weights as ordinary LoRA.
    for flag in ("use_dora", "use_rslora", "use_qalora"):
        if config.get(flag):
            raise ValueError(f"unsupported PEFT variant: {flag}")
    for field in ("modules_to_save", "rank_pattern", "alpha_pattern", "loftq_config",
                  "trainable_token_indices", "layer_replication"):
        if config.get(field):
            raise ValueError(f"unsupported PEFT extension: {field}")
    return config


def _tensor_headers(path: Path) -> dict:
    total = path.stat().st_size
    if total < 11:
        raise ValueError("safetensors file too short")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("missing safetensors length prefix")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= MAX_HEADER or 8 + length > total:
            raise ValueError("invalid or oversized safetensors header")
        raw = stream.read(length)
    if not raw.startswith(b"{"):
        raise ValueError("safetensors header must begin with '{'")
    header = _strict_json(raw)
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (type(metadata) is not dict or
                                 any(type(k) is not str or type(v) is not str
                                     for k, v in metadata.items())):
        raise ValueError("invalid safetensors metadata")
    if not 1 <= len(header) <= MAX_TENSORS:
        raise ValueError("safetensors tensor count out of bounds")
    ranges = []
    tensors = {}
    buffer_size = total - 8 - length
    for name, record in header.items():
        if type(name) is not str or type(record) is not dict or set(record) != {
            "dtype", "shape", "data_offsets"
        }:
            raise ValueError("invalid safetensors tensor entry")
        dtype = record["dtype"]
        shape = record["shape"]
        offsets = record["data_offsets"]
        if type(dtype) is not str or dtype not in _FLOAT_DTYPES:
            raise ValueError("unsupported LoRA tensor dtype")
        if (type(shape) is not list or len(shape) != 2 or
                any(type(d) is not int or not 0 < d <= (1 << 48) for d in shape)):
            raise ValueError("LoRA tensor must be a non-empty matrix")
        if (type(offsets) is not list or len(offsets) != 2 or
                any(type(i) is not int or i < 0 for i in offsets)):
            raise ValueError("invalid safetensors data offsets")
        begin, end = offsets
        if end < begin or end > buffer_size or end - begin != (
            shape[0] * shape[1] * _FLOAT_DTYPES[dtype]
        ):
            raise ValueError("tensor data span disagrees with dtype or shape")
        tensors[name] = (dtype, tuple(shape))
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("safetensors byte buffer has a hole or overlap")
        cursor = end
    if cursor != buffer_size:
        raise ValueError("safetensors file contains unindexed trailing data")
    return tensors


def inspect_peft_lora_artifacts(directory: str | Path) -> dict:
    """Inspect local PEFT files, returning **untrusted** source-only proposal.

    This never claims that the separately held base weights/tokenizer, model
    process, adapter rights, optimizer, evaluator or GGUF output were verified.
    It does not mutate or load any model and never authorizes import/execution.
    """
    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("adapter directory must be a real directory")
    config_path = _local_file(root, "adapter_config.json")
    weights_path = _local_file(root, "adapter_model.safetensors")
    baseline = tuple((p.stat().st_dev, p.stat().st_ino,
                      p.stat().st_size, p.stat().st_mtime_ns)
                     for p in (config_path, weights_path))
    config = _read_config(config_path)
    tensors = _tensor_headers(weights_path)
    pairs = {}
    for name, (dtype, shape) in tensors.items():
        match = _LORA_NAME.fullmatch(name)
        if match is None:
            raise ValueError("non-standard or non-LoRA tensor in adapter")
        module, side = match["module"], match["side"]
        info = pairs.setdefault(module, {})
        if side in info:
            raise ValueError("duplicate LoRA A/B tensor for module")
        info[side] = (dtype, shape)
    parsed = []
    matched_targets = set()
    for module, sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            raise ValueError("unpaired LoRA A/B matrices")
        a_dtype, a_shape = sides["A"]
        b_dtype, b_shape = sides["B"]
        if a_dtype != b_dtype or a_shape[0] != b_shape[1] or a_shape[0] != config["r"]:
            raise ValueError("LoRA rank or dtype mismatch")
        matching = [t for t in config["target_modules"] if module == t or module.endswith("." + t)]
        if len(matching) != 1:
            raise ValueError("LoRA module missing or ambiguously matched in config targets")
        matched_targets.add(matching[0])
        parsed.append({"module": module, "rank": a_shape[0],
                       "input_features": a_shape[1], "output_features": b_shape[0],
                       "dtype": a_dtype, "target": matching[0]})
    if matched_targets != set(config["target_modules"]):
        raise ValueError("configured LoRA target has no tensor pair")
    config_digest = _hash_file(config_path)
    weights_digest = _hash_file(weights_path)
    after = tuple((p.stat().st_dev, p.stat().st_ino,
                   p.stat().st_size, p.stat().st_mtime_ns)
                  for p in (config_path, weights_path))
    if baseline != after:
        raise ValueError("PEFT file changed during inspection")
    return {
        "schema": SCHEMA, "candidate_only": True, "trainingday_ready": False,
        "evidence_class": "LOCAL_FILE_INSPECTION_SELF_REPORTED",
        "adapter_sha256": weights_digest, "config_sha256": config_digest,
        "base_model_name_or_path": config["base_model_name_or_path"],
        "base_revision": config["revision"],
        "rank": config["r"], "lora_alpha": config["lora_alpha"],
        "lora_scaling": config["lora_alpha"] / config["r"],
        "target_modules": sorted(config["target_modules"]),
        "lora_pairs": parsed,
        "holds": [
            "unverified base-model and tokenizer bytes, model configuration and exact layer shapes",
            "unverified adapter rights and independent host authority",
            "unverified PEFT loader compatibility, donor-disconnected restore and behavioral evaluation",
            "unverified optimizer, converter and serving artifact digest chain",
        ],
    }
