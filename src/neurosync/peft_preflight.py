"""Read-only, stdlib-only PEFT LoRA artifact inspection (never a transfer grant).

This intentionally supports only simple two-matrix LoRA safetensors. It parses
no tensor values and never imports torch, PEFT, pickle, or executable metadata.
The output is a candidate proposal, not an authorization or a model restore.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct


SCHEMA = "NEUROSYNC_PEFT_ARTIFACT_PREFLIGHT_V1"
MAX_HEADER = 1_000_000
MAX_CONFIG = 1_000_000
MAX_TENSORS = 20_000
MAX_ADAPTER_BYTES = 4 * (1 << 30)
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


def _open_secure_root(root: Path) -> int:
    """Anchor every absolute directory component without following symlinks."""
    if (os.name != "posix" or not all(hasattr(os, flag) for flag in
            ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")) or
            os.open not in os.supports_dir_fd or not hasattr(os, "pread")):
        raise ValueError("secure POSIX directory handles required; unsupported host HOLD")
    if not root.is_absolute() or ".." in root.parts or len(root.parts) < 3:
        raise ValueError("adapter directory must be a narrow absolute clean path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open("/", flags)
    try:
        for component in root.parts[1:]:
            if component in ("", ".", ".."):
                raise ValueError("invalid adapter directory component")
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except (OSError, ValueError) as exc:
        os.close(fd)
        raise ValueError("adapter directory unavailable or linked") from exc


def _open_secure_leaf(root_fd: int, name: str, limit: int) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=root_fd)
    except OSError as exc:
        raise ValueError("PEFT file unavailable or linked") from exc
    try:
        st = os.fstat(fd)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or
                not 0 < st.st_size <= limit):
            raise ValueError("PEFT file must be exclusive regular bounded file")
        return fd
    except (OSError, ValueError):
        os.close(fd)
        raise


def _read_fd_range(fd: int, start: int, size: int) -> bytes:
    """Read only the bounded requested bytes from the held descriptor."""
    result = bytearray()
    while len(result) < size:
        chunk = os.pread(fd, min(1 << 20, size - len(result)), start + len(result))
        if not chunk:
            raise ValueError("PEFT file truncated during inspection")
        result.extend(chunk)
    return bytes(result)


def _hash_fd(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    cursor = 0
    while cursor < size:
        chunk = os.pread(fd, min(1 << 20, size - cursor), cursor)
        if not chunk:
            raise ValueError("PEFT file truncated during SHA-256")
        digest.update(chunk)
        cursor += len(chunk)
    return digest.hexdigest()


def _read_config(fd: int, size: int) -> dict:
    config = _strict_json(_read_fd_range(fd, 0, size))
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


def _tensor_headers(fd: int, total: int) -> dict:
    if total < 11:
        raise ValueError("safetensors file too short")
    prefix = _read_fd_range(fd, 0, 8)
    length, = struct.unpack("<Q", prefix)
    if not 2 <= length <= MAX_HEADER or 8 + length > total:
        raise ValueError("invalid or oversized safetensors header")
    raw = _read_fd_range(fd, 8, length)
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
    root_fd = _open_secure_root(root)
    config_fd = weights_fd = -1
    try:
        root_start = os.fstat(root_fd)
        config_fd = _open_secure_leaf(root_fd, "adapter_config.json", MAX_CONFIG)
        weights_fd = _open_secure_leaf(root_fd, "adapter_model.safetensors",
                                       MAX_ADAPTER_BYTES)
        config_start, weights_start = os.fstat(config_fd), os.fstat(weights_fd)
        config = _read_config(config_fd, config_start.st_size)
        tensors = _tensor_headers(weights_fd, weights_start.st_size)
        # Verify from the same open file descriptions that supplied the parsed
        # bytes; never reopen by pathname to obtain a different SHA-256 stream.
        config_digest = _hash_fd(config_fd, config_start.st_size)
        weights_digest = _hash_fd(weights_fd, weights_start.st_size)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        for old, current in ((config_start, os.fstat(config_fd)),
                             (weights_start, os.fstat(weights_fd))):
            if any(getattr(old, key) != getattr(current, key) for key in identity):
                raise ValueError("PEFT file changed during inspection")

        # Verify that each fixed pathname still names the inspected inode.
        for name, limit, old in (("adapter_config.json", MAX_CONFIG, config_start),
                                 ("adapter_model.safetensors", MAX_ADAPTER_BYTES,
                                  weights_start)):
            current_fd = _open_secure_leaf(root_fd, name, limit)
            try:
                current = os.fstat(current_fd)
                if any(getattr(old, key) != getattr(current, key) for key in identity):
                    raise ValueError("PEFT file pathname changed during inspection")
            finally:
                os.close(current_fd)

        # The original directory itself may have been moved, while the held
        # descriptor still reads old bytes; re-open the *declared* root.
        current_root_fd = _open_secure_root(root)
        try:
            root_finish = os.fstat(root_fd)
            current_root = os.fstat(current_root_fd)
            root_identity = ("st_dev", "st_ino", "st_ctime_ns", "st_mtime_ns")
            if (any(getattr(root_start, key) != getattr(root_finish, key)
                    for key in root_identity) or
                    any(getattr(root_finish, key) != getattr(current_root, key)
                        for key in root_identity)):
                raise ValueError("PEFT adapter directory changed during inspection")
        finally:
            os.close(current_root_fd)
    finally:
        if weights_fd >= 0:
            os.close(weights_fd)
        if config_fd >= 0:
            os.close(config_fd)
        os.close(root_fd)
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
