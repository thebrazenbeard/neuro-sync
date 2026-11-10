"""Paired local Ollama diagnostic tests; fake HTTP only, no real inference."""

from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
from pathlib import Path
import sys
import threading
import unittest


sys.path.insert(0, str(Path(__file__).parent))
paired = importlib.import_module("paired_probe")


@contextlib.contextmanager
def ollama_fake(*, loaded: list[str] | None = None,
                bad_response: bool = False, failed_response: bool = False,
                wrong_model: bool = False, changed_digest: bool = False):
    calls = []
    correction_steps = []
    names = loaded or []
    state = {"chats": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def respond(self, obj, status=200):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            calls.append(("GET", self.path))
            if self.path == "/api/ps":
                self.respond({"models": [{"name": item} for item in names]})
            elif self.path == "/api/tags":
                digest_b = ("c" if changed_digest and state["chats"] else "b") * 64
                self.respond({"models": [
                    {"name": "qwen3.5:0.8b", "digest": "a" * 64, "size": 1300},
                    {"name": "vera-successor:0.8b-context", "digest": digest_b,
                     "size": 1300},
                ]})
            else:
                self.respond({"error": "bad path"}, 404)

        def do_POST(self):
            count = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(count))
            calls.append(("POST", self.path, data))
            if self.path != "/api/chat":
                self.respond({"error": "bad path"}, 404)
                return
            state["chats"] += 1
            last = data["messages"][-1]["content"]
            if failed_response and "model generated 96 tokens" in last.lower():
                self.respond({"error": "server failed"}, 500)
                return
            if bad_response and "Plasma Prion" in last:
                self.respond({"model": data["model"],
                              "message": {"content": ""}, "done": False,
                              "done_reason": "length"})
                return
            if last == "Project name: Lynx.":
                answer = "Acknowledged Lynx."
                correction_steps.append((1, data))
            elif last == "Correction: Raven.":
                answer = "Updated to Raven."
                correction_steps.append((2, data))
            elif last.startswith("What is the current project name?"):
                answer = "Raven."
                correction_steps.append((3, data))
            else:
                answer = "I cannot verify that from this context."
            self.respond({"model": "wrong:tag" if wrong_model else data["model"],
                          "message": {"content": answer}, "done": True,
                          "done_reason": "stop"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, correction_steps
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


class PairedProbeTests(unittest.TestCase):
    def test_offline_preview_no_network_and_schema(self):
        record = paired.probe_pair(
            endpoint="http://127.0.0.1:1",
            model_a="qwen3.5:0.8b",
            model_b="vera-successor:0.8b-context")
        self.assertEqual("PREVIEW", record["status"])
        self.assertEqual(paired.SCHEMA, record["schema"])
        self.assertEqual(11, record["num_scenarios"])
        self.assertEqual([], record["rows"])
        self.assertFalse(record["weight_update"])
        self.assertFalse(record["teacher_used"])
        self.assertFalse(record["learning_claim"])

    def test_synthetic_paired_real_turns_and_no_default_text_storage(self):
        with ollama_fake() as (endpoint, calls, corrections):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096)
        self.assertEqual("COMPLETE", record["status"])
        self.assertEqual(22, len(record["rows"]))
        self.assertEqual({"a" * 64, "b" * 64},
                         {x["ollama_reported_digest"] for x in record["model_metadata"].values()})
        self.assertEqual(6, len(corrections))  # three real calls per model
        self.assertEqual("Acknowledged Lynx.", corrections[1][1]["messages"][1]["content"])
        self.assertEqual("Updated to Raven.", corrections[2][1]["messages"][3]["content"])
        correction = [r for r in record["rows"] if r["prompt_id"] == "correction_3turn"]
        self.assertEqual([3, 3], [r["turn_count_attempted"] for r in correction])
        self.assertTrue(all("answer" not in row for row in record["rows"]))
        self.assertTrue(all(row["done_reason"] == "stop" for row in record["rows"]))
        chats = [c[2] for c in calls if c[0] == "POST"]
        self.assertEqual(26, len(chats))  # 11 questions + two extra correction turns, twice
        for chat in chats:
            self.assertFalse(chat["think"])
            self.assertEqual(0, chat["options"]["temperature"])
            self.assertEqual(317, chat["options"]["seed"])
            self.assertEqual(0, chat["keep_alive"])
            self.assertEqual(128, chat["options"]["num_predict"])

    def test_opt_in_answers_for_scoring(self):
        with ollama_fake() as (endpoint, _, _):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096,
                include_responses=True)
        self.assertEqual("Raven.", next(row["answer"] for row in record["rows"]
                                        if row["prompt_id"] == "correction_3turn"))
        self.assertTrue(record["answer_content_stored"])

    def test_truncated_or_empty_reply_marked_incomplete(self):
        with ollama_fake(bad_response=True) as (endpoint, _, _):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096)
        self.assertEqual("INCOMPLETE", record["status"])
        rows = [r for r in record["rows"] if r["prompt_id"] == "invented_fact"]
        self.assertEqual(["length", "length"], [r["done_reason"] for r in rows])
        self.assertTrue(all(r["status"] == "INCOMPLETE" for r in rows))

    def test_failing_http_response_not_silent_success(self):
        with ollama_fake(failed_response=True) as (endpoint, _, _):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096)
        self.assertEqual("INCOMPLETE", record["status"])
        failed = [r for r in record["rows"]
                  if r["prompt_id"] == "empty_visible_output"]
        self.assertTrue(all(r["status"] == "INCOMPLETE" for r in failed))
        self.assertTrue(all(r["turns"][0]["status"] == "INCOMPLETE_REQUEST"
                            for r in failed))
        self.assertTrue(all(r["turns"][0]["unload_requested_in_attempt_payload"]
                            and not r["turns"][0]["unload_completion_verified"]
                            for r in failed))

    def test_wrong_model_response_is_incomplete(self):
        with ollama_fake(wrong_model=True) as (endpoint, _, _):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096)
        self.assertEqual("INCOMPLETE", record["status"])
        self.assertTrue(all(row["status"] == "INCOMPLETE"
                            for row in record["rows"]))

    def test_model_manifest_mutation_mid_run_fails_closed(self):
        with ollama_fake(changed_digest=True) as (endpoint, _, _):
            record = paired.probe_pair(
                endpoint=endpoint, model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=4096)
        self.assertFalse(record["model_tag_manifest_unchanged"])
        self.assertEqual("INCOMPLETE", record["status"])

    def test_refuses_competing_loaded_model_before_chat(self):
        with ollama_fake(loaded=["qwen3.5:4b"]) as (endpoint, calls, _):
            with self.assertRaisesRegex(paired.receiver_api.ProbeError, "eviction"):
                paired.probe_pair(
                    endpoint=endpoint, model_a="qwen3.5:0.8b",
                    model_b="vera-successor:0.8b-context", execute=True,
                    free_ram_mib=4096)
        self.assertFalse(any(c[0] == "POST" for c in calls))

    def test_rejects_insufficient_user_reported_ram_without_network(self):
        with self.assertRaisesRegex(paired.receiver_api.ProbeError, "RAM"):
            paired.probe_pair(
                endpoint="http://127.0.0.1:1", model_a="qwen3.5:0.8b",
                model_b="vera-successor:0.8b-context", execute=True,
                free_ram_mib=670, min_free_ram_mib=2048)

    def test_execution_requires_ram_observation_before_any_http(self):
        with ollama_fake() as (endpoint, calls, _):
            with self.assertRaisesRegex(paired.receiver_api.ProbeError,
                                        "requires operator-reported"):
                paired.probe_pair(
                    endpoint=endpoint, model_a="qwen3.5:0.8b",
                    model_b="vera-successor:0.8b-context", execute=True)
            self.assertEqual([], calls)

    def test_endpoint_and_tag_validation(self):
        for endpoint in ("http://127.0.0.1:0", "http://@127.0.0.1:11434",
                         "http://example.com:11434", "https://127.0.0.1:11434"):
            with self.subTest(endpoint=endpoint), self.assertRaises(
                    paired.receiver_api.ProbeError):
                paired.probe_pair(endpoint=endpoint, model_a="qwen3.5:0.8b",
                                  model_b="vera-successor:0.8b-context")
        with self.assertRaisesRegex(paired.receiver_api.ProbeError, "distinct"):
            paired.probe_pair(endpoint="http://127.0.0.1:1",
                              model_a="same", model_b="same")


if __name__ == "__main__":
    unittest.main()
