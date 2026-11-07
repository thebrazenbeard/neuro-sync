"""Local receiver baseline tests. No model downloads or external HTTP."""

from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


MODULE = Path(__file__).with_name("probe.py")
spec = importlib.util.spec_from_file_location("local_successor_probe", MODULE)
probe_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(probe_module)


@contextlib.contextmanager
def fake_ollama(*, oversized: bool = False, missing_model: bool = False,
                empty: bool = False, truncated: bool = False,
                unfinished: bool = False, bad_digest: bool = False,
                bad_size: bool = False, changed_digest: bool = False,
                wrong_response_model: bool = False):
    calls = []
    state = {"loaded": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def reply(self, value):
            raw = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            calls.append(("GET", self.path))
            if self.path == "/api/tags":
                models = [] if missing_model else [
                    {"name": "qwen3.5:0.8b",
                     "digest": ("g" * 64 if bad_digest else
                                "b" * 64 if changed_digest and state["loaded"] else
                                "a" * 64),
                     "size": "not_number" if bad_size else 100}]
                self.reply({"models": models})
            elif self.path == "/api/ps":
                self.reply({"models": ([{"name": "qwen3.5:0.8b"}]
                                       if state["loaded"] else [])})
            else:
                self.send_error(404)

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(size))
            calls.append(("POST", self.path, request))
            if self.path != "/api/chat":
                self.send_error(404)
                return
            state["loaded"] = True
            if oversized:
                self.reply({"model": "qwen3.5:0.8b",
                            "message": {"content": "x" * probe_module.MAX_RESPONSE_BYTES},
                            "done": True, "done_reason": "stop", "eval_count": 5})
            else:
                self.reply({"model": "other:model" if wrong_response_model else
                            "qwen3.5:0.8b",
                            "message": {"content": "" if empty else
                            "I can only use current context."},
                            "done": not unfinished,
                            "done_reason": "length" if truncated else "stop",
                            "eval_count": 5})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


class LocalSuccessorTests(unittest.TestCase):
    def test_endpoint_must_be_exact_literal_loopback(self):
        for url in (
            "https://127.0.0.1:11434", "http://localhost:11434",
            "http://8.8.8.8:11434", "http://127.0.0.1:11434/redirect",
            "http://127.0.0.1:11434?x=y", "http://user@127.0.0.1:11434",
            "http://@127.0.0.1:11434", "http://127.0.0.1:0",
            "http://127.0.0.1:11434#fragment", "http://127.0.0.1",
        ):
            with self.subTest(url=url), self.assertRaises(probe_module.ProbeError):
                probe_module.local_endpoint(url)
        self.assertEqual(("127.0.0.1", 11434), probe_module.local_endpoint(
            "http://127.0.0.1:11434"))

    def test_preview_does_not_contact_ollama(self):
        record = probe_module.probe(endpoint="http://127.0.0.1:1",
                                    model="qwen3.5:0.8b")
        self.assertEqual("preview", record["status"])
        self.assertEqual("SOURCE_PREVIEW", record["evidence_class"])
        self.assertFalse(record["weight_update"])
        self.assertFalse(record["teacher_used"])
        self.assertFalse(record["learning_claim"])

    def test_real_local_probe_hashes_without_exporting_text(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.txt"
            profile.write_text("Keep memory claims honest.", encoding="utf-8")
            with fake_ollama() as (url, calls):
                record = probe_module.probe(
                    endpoint=url, model="qwen3.5:0.8b",
                    profile_path=str(profile), execute=True)
            self.assertEqual("completed", record["status"])
            self.assertEqual("a" * 64, record["ollama_reported_manifest_digest"])
            self.assertEqual(100, record["ollama_reported_model_size_bytes"])
            self.assertTrue(record["requested_tag_load_transition_observed"])
            self.assertEqual(3, len(record["scenarios"]))
            self.assertTrue(all("response" not in row
                                for row in record["scenarios"]))
            self.assertNotIn("Keep memory claims honest.", json.dumps(record))
            self.assertEqual(3, len([call for call in calls if call[0] == "POST"]))
            for call in calls:
                if call[0] == "POST":
                    self.assertEqual(0, call[2]["options"]["temperature"])
                    self.assertFalse(call[2]["stream"])
                    self.assertFalse(call[2]["think"])
                    self.assertEqual("system", call[2]["messages"][0]["role"])

    def test_empty_unfinished_or_truncated_reply_cannot_complete(self):
        for variant in ("empty", "truncated", "unfinished"):
            with self.subTest(variant=variant):
                with fake_ollama(**{variant: True}) as (url, _):
                    with self.assertRaises(probe_module.ProbeError):
                        probe_module.probe(
                            endpoint=url, model="qwen3.5:0.8b", execute=True)

    def test_opt_in_response_and_existing_file_refusal(self):
        with fake_ollama() as (url, _):
            record = probe_module.probe(
                endpoint=url, model="qwen3.5:0.8b", execute=True,
                include_responses=True)
        self.assertIn("response", record["scenarios"][0])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "receipt.json"
            probe_module._write_new_receipt(out, record)
            self.assertEqual(record, json.loads(out.read_text(encoding="utf-8")))
            with self.assertRaises(FileExistsError):
                probe_module._write_new_receipt(out, record)

    def test_missing_model_fails_closed(self):
        with fake_ollama(missing_model=True) as (url, calls):
            with self.assertRaisesRegex(probe_module.ProbeError, "not installed"):
                probe_module.probe(endpoint=url, model="qwen3.5:0.8b",
                                   execute=True)
            self.assertFalse(any(item[0] == "POST" for item in calls))

    def test_invalid_or_changed_ollama_manifest_fails_closed(self):
        for variant in ("bad_digest", "bad_size", "changed_digest",
                        "wrong_response_model"):
            with self.subTest(variant=variant):
                with fake_ollama(**{variant: True}) as (url, _):
                    with self.assertRaises(probe_module.ProbeError):
                        probe_module.probe(endpoint=url, model="qwen3.5:0.8b",
                                           execute=True)

    def test_bounded_response_and_model_settings(self):
        with fake_ollama(oversized=True) as (url, _):
            with self.assertRaisesRegex(probe_module.ProbeError, "byte cap"):
                probe_module.probe(endpoint=url, model="qwen3.5:0.8b",
                                   execute=True)
        for overrides in ({"max_tokens": 129}, {"timeout_seconds": 91},
                          {"context_tokens": 9999}, {"model": "../../bad"}):
            with self.subTest(overrides=overrides), self.assertRaises(
                    probe_module.ProbeError):
                args = {"endpoint": "http://127.0.0.1:1",
                        "model": "qwen3.5:0.8b"}
                args.update(overrides)
                probe_module.probe(**args)

    def test_profile_size_empty_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.txt"
            path.write_text(" ", encoding="utf-8")
            with self.assertRaisesRegex(probe_module.ProbeError, "empty"):
                probe_module.read_profile(str(path))
            path.write_bytes(b"x" * (probe_module.MAX_PROFILE_BYTES + 1))
            with self.assertRaisesRegex(probe_module.ProbeError, "byte cap"):
                probe_module.read_profile(str(path))
            link = Path(directory) / "link.txt"
            try:
                link.symlink_to(path)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(probe_module.ProbeError, "symlink"):
                probe_module.read_profile(str(link))


if __name__ == "__main__":
    unittest.main()
