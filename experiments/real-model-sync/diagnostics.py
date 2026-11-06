"""One-request, loopback-only streaming latency diagnostic (not an eval).

Default behavior is an offline request preview: NO inference or HTTP request.
With --execute, this sends exactly one synthetic streaming request to an
already running local server and measures response headers, first nonempty
content event, and completion separately.  It does NOT assess tool planning,
model identity, neural synchronization, or weights.  Every receipt is local,
unattested, and deliberately excludes raw prompt and generated model text.

This diagnostic is intentionally separate from pilot.py's frozen experiment:
its synthetic prefill profiles are NOT a substitute for the seven-arm trial.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import threading
import time
from urllib.parse import urlsplit


MAX_REQUEST_BYTES = 8192
MAX_RESPONSE_BYTES = 131072
MAX_LINE_BYTES = 8192
MAX_TIMEOUT_SECONDS = 20
MAX_TOKENS = 64
CATALOG_OPS = (
    "add_two", "add_seven", "subtract_four", "multiply_two", "multiply_three",
    "negate", "square", "absolute", "mod_eleven", "subtract_two",
    "add_five", "multiply_four", "clamp_ten",
)


def local_endpoint(url: str, backend: str) -> tuple[str, int, str]:
    """Resolve only literal loopback HTTP; no redirects or proxy environment."""
    if backend not in {"openai", "ollama"}:
        raise ValueError("backend must be openai or ollama")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid endpoint") from exc
    if (parsed.scheme != "http" or parsed.hostname not in
            {"127.0.0.1", "::1"} or port is None or
            parsed.path not in {"", "/"} or parsed.username or parsed.password or
            parsed.query or parsed.fragment):
        raise ValueError("only explicit loopback HTTP host:port endpoints are permitted")
    return parsed.hostname, port, (
        "/v1/chat/completions" if backend == "openai" else "/api/chat"
    )


def build_request(backend: str, model: str, profile: str,
                  max_tokens: int) -> tuple[str, bytes]:
    """Create a small synthetic prompt; profile changes prefill size only."""
    if type(model) is not str or not model or len(model) > 256:
        raise ValueError("nonempty bounded model identifier required")
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError("max_tokens must be 1..64")
    if profile not in {"tiny", "catalog"}:
        raise ValueError("profile must be tiny or catalog")
    prompt = "Reply exactly READY. This is a synthetic latency check."
    if profile == "catalog":
        # Stable fixture, no private data and no selected answer/goal.  All
        # 13 semantically named operations mirror the pilot's catalog count,
        # not its opaque handles or end-to-end reasoning task.
        mapping = {f"cap-{index:016x}": op
                   for index, op in enumerate(CATALOG_OPS)}
        prompt += " Read this synthetic catalog first: " + json.dumps(
            mapping, sort_keys=True, separators=(",", ":"))
    messages = [{"role": "user", "content": prompt}]
    body = ({"model": model, "messages": messages, "stream": True,
             "temperature": 0, "max_tokens": max_tokens}
            if backend == "openai" else
            {"model": model, "messages": messages, "stream": True,
             "options": {"temperature": 0, "num_predict": max_tokens}})
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("synthetic request exceeds bound")
    return ("/v1/chat/completions" if backend == "openai" else "/api/chat"), payload


def parse_stream_line(backend: str, raw: bytes) -> tuple[str, bool, bool]:
    """Return (content, terminal_event, parsed_event); empty/preamble != token."""
    line = raw.strip()
    if not line or line.startswith(b":"):
        return "", False, False
    if backend == "openai":
        if not line.startswith(b"data: "):
            raise ValueError("unexpected streaming event line")
        body = line[6:]
        if body == b"[DONE]":
            return "", True, True
        chunk = json.loads(body)
        if (type(chunk) is not dict or type(chunk.get("choices")) is not list or
                not chunk["choices"] or type(chunk["choices"][0]) is not dict):
            raise ValueError("invalid OpenAI chunk")
        delta = chunk["choices"][0].get("delta", {})
        if type(delta) is not dict:
            raise ValueError("invalid OpenAI delta")
        content = delta.get("content")
        if content is None:
            content = ""
        if type(content) is not str:
            raise ValueError("nonstring content chunk")
        return content, False, True
    item = json.loads(line)
    if type(item) is not dict or type(item.get("done")) is not bool:
        raise ValueError("invalid Ollama chunk")
    message = item.get("message") or {}
    if type(message) is not dict:
        raise ValueError("invalid Ollama message")
    content = message.get("content")
    if content is None:
        content = ""
    if type(content) is not str:
        raise ValueError("nonstring content chunk")
    return content, item["done"], True


def probe(endpoint: str, backend: str, model: str, *, profile: str = "tiny",
          timeout_seconds: float = 20, max_tokens: int = 16,
          execute: bool = False) -> dict:
    """Offline by default; explicit execute permits ONE bounded local request."""
    host, port, path = local_endpoint(endpoint, backend)
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds must be >0 and <=20")
    if type(execute) is not bool:
        raise ValueError("execute must be an explicit boolean")
    expected_path, payload = build_request(backend, model, profile, max_tokens)
    assert expected_path == path
    printable_host = f"[{host}]" if ":" in host else host
    receipt = {
        "version": 1, "kind": "synthetic_stream_latency_diagnostic",
        "evidence_class": "REQUEST_PREVIEW_NO_INFERENCE",
        "endpoint": f"http://{printable_host}:{port}{path}", "backend": backend,
        "model_id_requested_not_attested": model, "profile": profile,
        "request_bytes": len(payload),
        "request_sha256": hashlib.sha256(payload).hexdigest(),
        "max_tokens": max_tokens, "timeout_seconds": timeout_seconds,
        "status": "preview", "http_status": None,
        "headers_ms": None, "first_content_ms": None,
        "completed_ms": None, "elapsed_ms": None,
        "response_bytes": 0, "response_sha256": None,
        "content_chars": 0, "events_parsed": 0,
        "serving_process_model_file_binding_verified": False,
        "model_cold_restart_verified": False,
    }
    if not execute:
        return receipt
    receipt["evidence_class"] = "LOCAL_HTTP_UNATTESTED"
    started = time.monotonic()
    deadline = started + timeout_seconds
    raw_hash = hashlib.sha256()
    conn = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    stop_watchdog = threading.Event()
    watchdog_fired = threading.Event()
    active_socket: list[socket.socket | None] = [None]

    def deadline_watchdog() -> None:
        # A per-recv timeout is not a wall clock budget: getresponse() may
        # receive a trickled header forever. The watchdog shuts down the
        # active TCP socket to interrupt *any* blocked send/header/body read.
        if not stop_watchdog.wait(max(0.0, deadline - time.monotonic())):
            watchdog_fired.set()
            sock = active_socket[0] or conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                conn.close()
            except OSError:
                pass

    watcher = threading.Thread(target=deadline_watchdog, daemon=True,
                               name="neurosync-latency-deadline")
    watcher.start()
    try:
        conn.connect()
        active_socket[0] = conn.sock
        if conn.sock is not None:
            conn.sock.settimeout(max(0.001, deadline - time.monotonic()))
        conn.request("POST", path, body=payload, headers={
            "Content-Type": "application/json", "Accept": "text/event-stream",
        })
        if conn.sock is not None:
            conn.sock.settimeout(max(0.001, deadline - time.monotonic()))
        response = conn.getresponse()
        receipt["http_status"] = response.status
        receipt["headers_ms"] = round((time.monotonic() - started) * 1000, 2)
        if response.status != 200:
            receipt["status"] = "http_error"
            return receipt
        active_socket[0] = conn.sock or getattr(
            getattr(response.fp, "raw", None), "_sock", None)
        line = bytearray()
        # One-byte reads are deliberate: checking the absolute deadline after
        # every byte avoids an adversarial trickle that holds readline() open
        # indefinitely. The server may still finish its work after disconnect.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                receipt["status"] = "timeout"
                break
            # http.client may clear conn.sock for Connection: close while its
            # response object still owns the underlying buffered socket.
            response_socket = conn.sock or getattr(
                getattr(response.fp, "raw", None), "_sock", None)
            if response_socket is not None:
                response_socket.settimeout(remaining)
            chunk = response.read(1)
            if not chunk:
                receipt["status"] = "invalid_stream" if line else "stream_ended_without_done"
                break
            receipt["response_bytes"] += 1
            if receipt["response_bytes"] > MAX_RESPONSE_BYTES:
                receipt["status"] = "response_limit"
                break
            raw_hash.update(chunk)
            line.extend(chunk)
            if len(line) > MAX_LINE_BYTES:
                receipt["status"] = "line_limit"
                break
            if chunk != b"\n":
                continue
            try:
                content, done, parsed_event = parse_stream_line(backend, bytes(line))
            except (ValueError, UnicodeError, TypeError, KeyError, IndexError):
                receipt["status"] = "invalid_stream"
                break
            line.clear()
            if parsed_event:
                receipt["events_parsed"] += 1
            if content:
                receipt["content_chars"] += len(content)
                if receipt["first_content_ms"] is None:
                    receipt["first_content_ms"] = round((time.monotonic()-started)*1000, 2)
            if done:
                receipt["completed_ms"] = round((time.monotonic()-started)*1000, 2)
                receipt["status"] = "complete" if receipt["content_chars"] else "complete_no_content"
                break
    except (socket.timeout, TimeoutError):
        receipt["status"] = "timeout"
    except (OSError, http.client.HTTPException, ValueError) as exc:
        receipt["status"] = "transport_error"
        receipt["error_type"] = type(exc).__name__
    finally:
        stop_watchdog.set()
        receipt["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
        receipt["response_sha256"] = raw_hash.hexdigest() if receipt["response_bytes"] else None
        conn.close()
        watcher.join(timeout=0.1)
        if watchdog_fired.is_set():
            receipt["status"] = "timeout"
            receipt["completed_ms"] = None
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--backend", choices=("openai", "ollama"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=("tiny", "catalog"), default="tiny")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--execute", action="store_true", help="permit exactly one local HTTP inference request")
    parser.add_argument("--receipt", type=Path, help="create a fresh JSON receipt (refuse overwrite)")
    args = parser.parse_args(argv)
    try:
        report = probe(args.endpoint, args.backend, args.model,
                       profile=args.profile, timeout_seconds=args.timeout,
                       max_tokens=args.max_tokens, execute=args.execute)
        if args.receipt:
            with args.receipt.open("x", encoding="utf8") as stream:
                json.dump(report, stream, sort_keys=True, indent=2)
                stream.write("\n")
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))
    # The exit code is about receipt generation; report.status is the outcome.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
