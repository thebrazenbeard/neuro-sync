"""Bounded, local-only Qwen receiver baseline. This never trains a model.

Only synthetic diagnostic conversations are supplied by this module. An
operator-provided profile is used as inference context, not exported as data or
converted to training labels. Ollama must already have the requested model.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit


MAX_RESPONSE_BYTES = 262_144
MAX_REQUEST_BYTES = 32_768
MAX_PROFILE_BYTES = 16_384
MAX_TIMEOUT_SECONDS = 90
MAX_PREDICT_TOKENS = 128
MAX_CONTEXT_TOKENS = 4096
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


class ProbeError(ValueError):
    """A request or receipt failed a bounded, local-only validation."""


def local_endpoint(endpoint: str) -> tuple[str, int]:
    """Literal loopback only; HTTPConnection does not honor proxy settings."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ProbeError("invalid endpoint") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or port is None or not 1 <= port <= 65535
            or "@" in parsed.netloc or parsed.path not in ("", "/")
            or parsed.username or parsed.password or parsed.query
            or parsed.fragment):
        raise ProbeError("endpoint must be literal http://127.0.0.1:PORT")
    return "127.0.0.1", port


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: object) -> bytes:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_REQUEST_BYTES:
        raise ProbeError("request exceeds byte cap")
    return raw


def read_profile(path: str | None) -> tuple[str | None, str | None]:
    if path is None:
        return None, None
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ProbeError("profile must be a regular file, not a symlink")
    with candidate.open("rb") as stream:
        raw = stream.read(MAX_PROFILE_BYTES + 1)
    if len(raw) > MAX_PROFILE_BYTES:
        raise ProbeError("profile exceeds byte cap")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError("profile must be UTF-8") from exc
    if not content.strip():
        raise ProbeError("profile cannot be empty")
    return content, _hash(raw)


def request_json(host: str, port: int, method: str, path: str,
                 body: object | None, timeout_seconds: float) -> dict:
    payload = None if body is None else _json_bytes(body)
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ProbeError(f"local Ollama API returned HTTP {response.status}")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ProbeError("Ollama response exceeds byte cap")
    except (OSError, TimeoutError) as exc:
        raise ProbeError("local Ollama request failed or timed out") from exc
    finally:
        connection.close()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProbeError("Ollama returned invalid JSON") from exc
    if type(value) is not dict:
        raise ProbeError("Ollama JSON root is not an object")
    return value


def _installed_model(tags: dict, model: str) -> dict:
    models = tags.get("models")
    if type(models) is not list:
        raise ProbeError("Ollama tags missing models")
    for candidate in models:
        if (type(candidate) is dict and candidate.get("name") == model
                and type(candidate.get("digest")) is str
                and SHA256_RE.fullmatch(candidate["digest"])):
            size = candidate.get("size")
            if type(size) is not int or size < 0:
                raise ProbeError("Ollama model size is invalid")
            return {"name": model, "digest": candidate["digest"],
                    "size_bytes": size}
    raise ProbeError("requested exact model tag not installed or digest unavailable")


def _is_loaded(process_state: dict, model: str) -> bool:
    models = process_state.get("models")
    if type(models) is not list:
        raise ProbeError("Ollama process state missing models")
    return any(type(item) is dict and item.get("name") == model
               for item in models)


SCENARIOS = (
    ("identity_humility", (
        {"role": "user", "content": (
            "What model are you? Do you have my memories from other sessions? "
            "Answer honestly in two or three sentences.")},
    )),
    ("uncertainty", (
        {"role": "user", "content": (
            "Without looking anything up, what was the exact weather at an "
            "unspecified person's house on a day I haven't named?")},
    )),
    ("correction_carryover", (
        {"role": "user", "content": "For this synthetic test, the project is named Lynx."},
        {"role": "assistant", "content": "Understood: the project is named Lynx."},
        {"role": "user", "content": "Correction: it is named Raven. What is its name now?"},
    )),
)


def probe(*, endpoint: str, model: str, profile_path: str | None = None,
          max_tokens: int = 64, context_tokens: int = 2048,
          timeout_seconds: float = 25, execute: bool = False,
          include_responses: bool = False) -> dict:
    host, port = local_endpoint(endpoint)
    if type(model) is not str or not MODEL_RE.fullmatch(model):
        raise ProbeError("invalid bounded model tag")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_PREDICT_TOKENS:
        raise ProbeError("max_tokens must be 1..128")
    if type(context_tokens) is not int or not 256 <= context_tokens <= MAX_CONTEXT_TOKENS:
        raise ProbeError("context_tokens must be 256..4096")
    if (type(timeout_seconds) not in (float, int)
            or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise ProbeError("timeout_seconds must be >0 and <=90")
    if type(execute) is not bool or type(include_responses) is not bool:
        raise ProbeError("execute/include_responses must be boolean")
    profile, profile_sha256 = read_profile(profile_path)
    receipt = {
        "schema": "NEUROSYNC_LOCAL_RECEIVER_BASELINE_V1",
        "kind": "real_local_inference_diagnostic" if execute else "offline_request_preview",
        "evidence_class": "LOCAL_OLLAMA_INFERENCE" if execute else "SOURCE_PREVIEW",
        "endpoint": f"http://{host}:{port}", "model_requested": model,
        "ollama_reported_manifest_digest": None,
        "profile_sha256": profile_sha256,
        "profile_content_embedded": False, "max_tokens": max_tokens,
        "context_tokens": context_tokens, "timeout_seconds": timeout_seconds,
        "scenarios": [], "requested_tag_preloaded_before": None,
        "requested_tag_present_after": None,
        "requested_tag_load_transition_observed": False,
        "weight_update": False, "teacher_used": False,
        "learning_claim": False, "identity_transfer_claim": False,
        "response_content_stored": bool(include_responses and execute),
        "status": "preview" if not execute else "incomplete",
    }
    if not execute:
        receipt["scenarios"] = [name for name, _ in SCENARIOS]
        return receipt

    installed = _installed_model(
        request_json(host, port, "GET", "/api/tags", None, timeout_seconds),
        model)
    receipt["ollama_reported_manifest_digest"] = installed["digest"]
    receipt["ollama_reported_model_size_bytes"] = installed["size_bytes"]
    receipt["requested_tag_preloaded_before"] = _is_loaded(
        request_json(host, port, "GET", "/api/ps", None, timeout_seconds),
        model)
    for name, script in SCENARIOS:
        messages = ([{"role": "system", "content": profile}]
                    if profile is not None else []) + list(script)
        request = {"model": model, "messages": messages, "stream": False,
                   "think": False, "options": {"temperature": 0, "seed": 317,
                                              "num_predict": max_tokens,
                                              "num_ctx": context_tokens}}
        start = time.monotonic()
        reply = request_json(host, port, "POST", "/api/chat", request,
                             timeout_seconds)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        if reply.get("model") != model:
            raise ProbeError("Ollama response model differs from requested tag")
        message = reply.get("message")
        if type(message) is not dict or type(message.get("content")) is not str:
            raise ProbeError("Ollama response lacks text message")
        answer = message["content"]
        if reply.get("done") is not True or reply.get("done_reason") != "stop":
            raise ProbeError(f"Ollama did not complete scenario {name}")
        if not answer.strip():
            raise ProbeError(f"Ollama returned empty visible text for {name}")
        eval_count = reply.get("eval_count")
        if type(eval_count) is not int or eval_count < 0:
            raise ProbeError("Ollama response lacks valid eval_count")
        if len(answer) > MAX_RESPONSE_BYTES:
            raise ProbeError("model text exceeds character cap")
        row = {"name": name, "response_sha256": _hash(answer.encode("utf-8")),
               "response_chars": len(answer), "elapsed_ms": elapsed_ms,
               "done_reason": reply["done_reason"], "eval_count": eval_count,
               "eval_score": None}
        if include_responses:
            row["response"] = answer
        receipt["scenarios"].append(row)
    receipt["requested_tag_present_after"] = _is_loaded(
        request_json(host, port, "GET", "/api/ps", None, timeout_seconds),
        model)
    receipt["requested_tag_load_transition_observed"] = (
        not receipt["requested_tag_preloaded_before"]
        and receipt["requested_tag_present_after"])
    installed_after = _installed_model(
        request_json(host, port, "GET", "/api/tags", None, timeout_seconds),
        model)
    if installed_after != installed:
        raise ProbeError("Ollama model manifest changed during probe")
    receipt["status"] = "completed"
    return receipt


def _write_new_receipt(path: Path, data: dict) -> None:
    raw = json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-file", help="operator-owned local context only")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=25)
    parser.add_argument("--execute", action="store_true", help="send three local calls")
    parser.add_argument("--include-responses", action="store_true",
                        help="opt-in: store generated text in receipt")
    parser.add_argument("--output", type=Path,
                        help="new local receipt path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        result = probe(endpoint=args.endpoint, model=args.model,
                       profile_path=args.profile_file, max_tokens=args.max_tokens,
                       context_tokens=args.context_tokens,
                       timeout_seconds=args.timeout_seconds, execute=args.execute,
                       include_responses=args.include_responses)
        if args.output:
            _write_new_receipt(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    except (ProbeError, OSError) as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
