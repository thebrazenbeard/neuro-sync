"""Opt-in structured transport contract; no model or network calls."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("ns_pilot_structured", Path(__file__).with_name("pilot.py"))
pilot = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


class StructuredTransportTest(unittest.TestCase):
    TASK = {"start": 1, "goal": 4,
            "candidate_handles": ["cap-0000000000000001", "cap-0000000000000002"],
            "shared_external_state": {"operations": {"cap-0000000000000001": "add 3"}}}

    def _capture(self, output_constraint):
        captured = {}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _): return json.dumps({"choices": [{"message": {
                "content": '{"steps":[],"abstain":true}'}}]}).encode()
        class Opener:
            def open(self, request, timeout):
                captured["request"] = json.loads(request.data)
                captured["timeout"] = timeout
                return Response()
        client = pilot.LocalModelClient(
            "http://127.0.0.1:11435", "openai", "fixture",
            max_tokens=64, timeout=5, output_constraint=output_constraint)
        client.opener = Opener()
        answer, trace = client(self.TASK)
        return captured["request"], answer, trace

    def test_schema_is_explicit_opt_in_and_only_uses_public_handles(self):
        request, answer, trace = self._capture("json_schema")
        response_format = request["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"steps", "abstain"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["steps"]["maxItems"], 2)
        items = schema["properties"]["steps"]["items"]
        self.assertEqual(items["properties"]["handle"]["enum"],
                         self.TASK["candidate_handles"])
        self.assertEqual(items["properties"]["input"]["enum"], ["initial", "previous"])
        self.assertNotIn("oracle", pilot.canonical_json(request))
        self.assertNotIn("expected_possible", pilot.canonical_json(request))
        self.assertEqual(json.loads(answer), {"steps": [], "abstain": True})
        self.assertEqual(trace["output_constraint"], "json_schema")
        self.assertEqual(trace["outcome"], "received_text")

    def test_original_v1_request_is_not_silently_changed(self):
        request, _, trace = self._capture("none")
        self.assertNotIn("response_format", request)
        self.assertNotIn("output_constraint", trace)

    def test_unsupported_backend_and_unknown_constraint_fail_before_network(self):
        with self.assertRaises(ValueError):
            pilot.LocalModelClient("http://127.0.0.1:11435", "ollama",
                                   "fixture", output_constraint="json_schema")
        with self.assertRaises(ValueError):
            pilot.LocalModelClient("http://127.0.0.1:11435", "openai",
                                   "fixture", output_constraint="invalid")
        with self.assertRaises(ValueError):
            pilot.LocalModelClient("http://127.0.0.1:11435", "openai",
                                   "fixture", output_constraint=[])

    def test_schema_transport_failure_preserves_mode_and_no_effect(self):
        class TimeoutOpener:
            def open(self, *_args, **_kwargs):
                raise TimeoutError("synthetic failure")
        client = pilot.LocalModelClient("http://127.0.0.1:11435", "openai",
                                        "fixture", output_constraint="json_schema")
        client.opener = TimeoutOpener()
        case = {"id": "case-transport-only", "start": 1, "goal": 4,
                "handles": self.TASK["candidate_handles"],
                "expected_possible": True}
        row = pilot.run_one(case, "V", self.TASK["shared_external_state"], client)
        self.assertEqual(row["failure_stage"], "ADAPTER_FAILURE")
        self.assertEqual(row["effects"], 0)
        self.assertFalse(row["gate_tested"])
        self.assertEqual(row["transport_trace"]["output_constraint"], "json_schema")

    def test_v1_runner_rejects_experimental_constraint_before_files_or_calls(self):
        import tempfile
        client = pilot.LocalModelClient("http://127.0.0.1:11435", "openai",
                                        "fixture", output_constraint="json_schema")
        case = {"id": "case-receipt", "start": 1, "goal": 4,
                "handles": self.TASK["candidate_handles"],
                "expected_possible": True}
        with tempfile.TemporaryDirectory() as parent:
            destination = Path(parent) / "v1-receipt"
            with self.assertRaisesRegex(ValueError, "outside v1"):
                pilot.run_suite({"cases": [case]}, client, destination,
                                model_evidence={"real_endpoint": True},
                                max_cases=1, arms=("V",))
            self.assertFalse(destination.exists())

    def test_strict_parser_still_rejects_extra_or_malformed_text(self):
        handles = self.TASK["candidate_handles"]
        with self.assertRaises(ValueError):
            pilot.parse_plan('Here is JSON: {"steps":[],"abstain":true}', handles)
        with self.assertRaises(ValueError):
            pilot.parse_plan('{"steps":[{"handle":"bogus","input":"initial"}],"abstain":false}', handles)


if __name__ == "__main__":
    unittest.main()
