"""Fixture-only timing diagnostic tests. No user model is invoked."""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import unittest


spec = importlib.util.spec_from_file_location(
    "neurosync_latency_diagnostics", Path(__file__).with_name("diagnostics.py")
)
diagnostics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostics
spec.loader.exec_module(diagnostics)


class LocalFixture(BaseHTTPRequestHandler):
    calls = 0
    payloads = []
    response_lines = ()
    response_code = 200
    header_trickle_interval = 0.0

    def do_POST(self):
        type(self).calls += 1
        size = int(self.headers["Content-Length"])
        type(self).payloads.append(json.loads(self.rfile.read(size)))
        if type(self).header_trickle_interval:
            try:
                self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                self.wfile.flush()
                # Never complete headers; drip bytes inside the socket timeout.
                for char in b"X-Pad: " + b"a" * 120:
                    time.sleep(type(self).header_trickle_interval)
                    self.wfile.write(bytes((char,)))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_response(type(self).response_code)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for item in type(self).response_lines:
            delay, line = item if isinstance(item, tuple) else (0, item)
            if delay:
                time.sleep(delay)
            try:
                self.wfile.write(line)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format, *args):
        return


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LocalFixture)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(1)

    def setUp(self):
        LocalFixture.calls = 0
        LocalFixture.payloads = []
        LocalFixture.response_lines = ()
        LocalFixture.response_code = 200
        LocalFixture.header_trickle_interval = 0.0
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def test_preview_issues_no_inference_even_when_endpoint_is_live(self):
        small = diagnostics.probe(self.url, "openai", "fixture", profile="tiny")
        large = diagnostics.probe(self.url, "openai", "fixture", profile="catalog")
        self.assertEqual(LocalFixture.calls, 0)
        self.assertEqual(small["evidence_class"], "REQUEST_PREVIEW_NO_INFERENCE")
        self.assertEqual(small["status"], "preview")
        self.assertGreater(large["request_bytes"], small["request_bytes"])
        self.assertIsNone(small["first_content_ms"])

    def test_one_openai_sse_request_records_first_content_and_completion(self):
        LocalFixture.response_lines = (
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"READY"}}]}\n\n',
            b'data: [DONE]\n\n',
        )
        result = diagnostics.probe(self.url, "openai", "fixture", execute=True)
        self.assertEqual(LocalFixture.calls, 1)
        self.assertTrue(LocalFixture.payloads[0]["stream"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["evidence_class"], "LOCAL_HTTP_UNATTESTED")
        self.assertIsNotNone(result["first_content_ms"])
        self.assertGreaterEqual(result["completed_ms"], result["first_content_ms"])
        self.assertGreater(result["response_bytes"], 0)
        self.assertEqual(result["content_chars"], 5)
        self.assertNotIn("READY", json.dumps(result))

    def test_one_ollama_ndjson_request_with_no_text_is_not_valid_content(self):
        LocalFixture.response_lines = (
            b'{"message":{"content":""},"done":false}\n',
            b'{"message":{"content":""},"done":true}\n',
        )
        result = diagnostics.probe(self.url, "ollama", "fixture", execute=True)
        self.assertEqual(LocalFixture.calls, 1)
        self.assertEqual(result["status"], "complete_no_content")
        self.assertIsNone(result["first_content_ms"])
        self.assertTrue(LocalFixture.payloads[0]["stream"])
        self.assertLessEqual(LocalFixture.payloads[0]["options"]["num_predict"], 64)

    def test_incomplete_stream_cannot_pretend_success(self):
        LocalFixture.response_lines = (b'data: {"choices":[{"delta":{"content":"R"}}]}\n\n',)
        result = diagnostics.probe(self.url, "openai", "fixture", execute=True)
        self.assertEqual(result["status"], "stream_ended_without_done")
        self.assertIsNone(result["completed_ms"])

    def test_stream_trickle_stops_at_absolute_deadline(self):
        # Several valid SSE comments can be returned just before each socket
        # timeout. A per-read timeout alone would allow arbitrarily long runs.
        LocalFixture.response_lines = tuple((0.025, b": keepalive\n") for _ in range(16))
        start = time.monotonic()
        result = diagnostics.probe(
            self.url, "openai", "fixture", execute=True, timeout_seconds=0.15)
        elapsed = time.monotonic() - start
        self.assertEqual(LocalFixture.calls, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["first_content_ms"])
        self.assertLess(elapsed, 0.6)

    def test_trickled_headers_cannot_evade_total_deadline(self):
        LocalFixture.header_trickle_interval = 0.025
        start = time.monotonic()
        result = diagnostics.probe(
            self.url, "openai", "fixture", execute=True, timeout_seconds=0.15)
        elapsed = time.monotonic() - start
        self.assertEqual(LocalFixture.calls, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertIsNone(result["completed_ms"])
        self.assertLess(elapsed, 0.6)

    def test_http_error_and_invalid_json_fail_with_no_content_leak(self):
        LocalFixture.response_code = 503
        self.assertEqual(diagnostics.probe(self.url, "openai", "fixture", execute=True)["status"],
                         "http_error")
        LocalFixture.response_code = 302
        self.assertEqual(diagnostics.probe(self.url, "openai", "fixture", execute=True)["status"],
                         "http_error")
        LocalFixture.response_code = 200
        LocalFixture.response_lines = (b'data: THIS_IS_PRIVATE_OR_MALFORMED\n',)
        result = diagnostics.probe(self.url, "openai", "fixture", execute=True)
        self.assertEqual(result["status"], "invalid_stream")
        self.assertNotIn("THIS_IS_PRIVATE_OR_MALFORMED", json.dumps(result))
        # A shape-valid JSON object with non-object choices must not escape
        # the bounded receipt path as an unhandled AttributeError.
        LocalFixture.response_lines = (b'data: {"choices":[7]}\n',)
        malformed = diagnostics.probe(self.url, "openai", "fixture", execute=True)
        self.assertEqual(malformed["status"], "invalid_stream")
        LocalFixture.response_lines = (
            b'data: {"choices":[{"delta":{"content":0}}]}\n',
        )
        self.assertEqual(diagnostics.probe(
            self.url, "openai", "fixture", execute=True)["status"], "invalid_stream")
        LocalFixture.response_lines = (b'{"message":{"content":[]},"done":false}\n',)
        self.assertEqual(diagnostics.probe(
            self.url, "ollama", "fixture", execute=True)["status"], "invalid_stream")

    def test_remote_or_unbounded_or_redirect_urls_rejected(self):
        invalid = (
            "https://127.0.0.1:11435", "http://example.com:11435",
            "http://localhost:11435",
            "http://user:pass@127.0.0.1:11435", "http://127.0.0.1:11435/evil",
            "http://127.0.0.1:11435?target=evil", "http://127.0.0.1",
        )
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                diagnostics.probe(url, "openai", "fixture")
        for timeout in (0, 21):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                diagnostics.probe(self.url, "openai", "fixture", timeout_seconds=timeout)
        for tokens in (0, 65):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                diagnostics.probe(self.url, "openai", "fixture", max_tokens=tokens)


if __name__ == "__main__":
    unittest.main()
