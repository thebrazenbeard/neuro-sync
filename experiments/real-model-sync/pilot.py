"""Bounded, local-only *real receiver* pilot. No model is trained here.

The donor is currently a scripted synthetic capability catalog. This test can
measure whether an actual unchanged receiver model uses external descriptions;
it CANNOT establish model-to-model learning or independent OS-level authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from neurosync.bus import (  # noqa: E402
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger, StateOp,
    ValidationReceipt, canonical_json,
)

VERSION = "neuro-sync-local-receiver-v0"
KEY = "capabilities.local_receiver_v0"
OPS = (
    "add_two", "add_seven", "subtract_four", "multiply_two", "multiply_three",
    "negate", "square", "absolute", "mod_eleven", "subtract_two",
    "add_five", "multiply_four", "clamp_ten",
)
ARMS = ("N", "R", "V", "P", "H", "D", "X")


def sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def apply_tool(op: str, x: int) -> int:
    """Effect handler. Only this function actually runs a selected tool."""
    if op == "add_two": return x + 2
    if op == "add_seven": return x + 7
    if op == "subtract_four": return x - 4
    if op == "multiply_two": return x * 2
    if op == "multiply_three": return x * 3
    if op == "negate": return -x
    if op == "square": return x * x
    if op == "absolute": return abs(x)
    if op == "mod_eleven": return x % 11
    if op == "subtract_two": return x - 2
    if op == "add_five": return x + 5
    if op == "multiply_four": return x * 4
    if op == "clamp_ten": return min(x, 10)
    raise ValueError("unknown op")


def oracle_tool(op: str, x: int) -> int:
    """Separately expressed expectation; same-host test, NOT trusted custody."""
    return {
        "add_two": lambda: 2 + x, "add_seven": lambda: 7 + x,
        "subtract_four": lambda: x + (-4), "multiply_two": lambda: x + x,
        "multiply_three": lambda: x + x + x, "negate": lambda: 0 - x,
        "square": lambda: pow(x, 2), "absolute": lambda: max(x, -x),
        "mod_eleven": lambda: x - (x // 11) * 11,
        "subtract_two": lambda: x + (-2), "add_five": lambda: 5 + x,
        "multiply_four": lambda: (x + x) + (x + x),
        "clamp_ten": lambda: 10 if x > 10 else x,
    }[op]()


def _random_case(index: int, rng: random.SystemRandom) -> dict:
    handles = [f"cap-{secrets.token_hex(8)}" for _ in OPS]
    rng.shuffle(handles)
    registry = dict(zip(handles, OPS))
    start = rng.randrange(-9, 10)
    gold_length = 1 if index % 3 == 0 else 2
    gold = rng.sample(handles, gold_length)
    value = start
    for handle in gold:
        value = oracle_tool(registry[handle], value)
    impossible = index in (4, 10)  # two sealed abstention targets
    if impossible:
        reachable = {start}
        current = {start}
        for _ in range(2):
            current = {oracle_tool(op, x) for x in current for op in OPS}
            reachable.update(current)
        candidate = 1_000_000_000 + rng.randrange(1, 100_000)
        while candidate in reachable:
            candidate += 1
        value = candidate
    return {"id": f"case-{index:02d}-{secrets.token_hex(4)}", "start": start,
            "goal": value, "handles": handles, "registry": registry,
            "expected_possible": not impossible, "gold_length": gold_length}


def freeze_suite(path: Path, count: int = 12) -> dict:
    if not 12 <= count <= 32:
        raise ValueError("suite requires 12..32 independent scenarios")
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError("refusing to overwrite existing sealed suite")
    rng = secrets.SystemRandom()
    suite = {"version": VERSION, "case_count": count,
             "cases": [_random_case(i, rng) for i in range(count)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(suite) + "\n", encoding="utf8")
    path.with_suffix(path.suffix + ".sha256").write_text(sha(suite) + "\n", encoding="ascii")
    return {"suite_sha256": sha(suite), "scenarios": count,
            "commitment_quality": "local self-commitment; NOT externally timestamped"}


def load_suite(path: Path) -> tuple[dict, str]:
    suite = json.loads(path.read_text(encoding="utf8"))
    if type(suite) is not dict or suite.get("version") != VERSION:
        raise ValueError("wrong suite version")
    cases = suite.get("cases")
    if (set(suite) != {"version", "case_count", "cases"} or
            type(cases) is not list or len(cases) != suite.get("case_count") or
            not 12 <= len(cases) <= 32 or
            len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("invalid suite")
    for case in cases:
        if type(case) is not dict or set(case) != {
                "id", "start", "goal", "handles", "registry",
                "expected_possible", "gold_length"}:
            raise ValueError("invalid scenario schema")
        handles = case["handles"]
        if (type(case["id"]) is not str or
                not re.fullmatch(r"case-[0-9]{2}-[a-f0-9]{8}", case["id"]) or
                type(handles) is not list or len(handles) != len(OPS) or
                len(set(handles)) != len(handles) or
                any(type(h) is not str or not re.fullmatch(r"cap-[a-f0-9]{16}", h)
                    for h in handles) or type(case["registry"]) is not dict or
                set(case["registry"]) != set(handles) or
                sorted(case["registry"].values()) != sorted(OPS) or
                type(case["start"]) is not int or not -9 <= case["start"] <= 9 or
                type(case["goal"]) is not int or abs(case["goal"]) > 2_000_000_000 or
                type(case["expected_possible"]) is not bool or
                type(case["gold_length"]) is not int or case["gold_length"] not in (1, 2)):
            raise ValueError("invalid scenario identifiers or content")
    digest = sha(suite)
    if path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() != digest:
        raise ValueError("suite changed after local self-commitment")
    return suite, digest


def _descriptions(case: dict) -> dict:
    """Synthetic donor-observed operation semantics; no selected plan/goal."""
    return dict(case["registry"])


def _raw_probes(case: dict) -> dict:
    # Include negative and out-of-small-range probes: only positive small
    # inputs made absolute/mod_eleven/clamp_ten indistinguishable to R.
    return {h: {str(x): oracle_tool(op, x) for x in (-3, 2, 12)}
            for h, op in case["registry"].items()}


def _bus_delivery(db: Path, case: dict, value: dict) -> dict:
    """Host-fixture commit, not an independently authenticated grant/witness."""
    ledger = StateLedger(db)
    now = int(time.time())
    source_manifest = {
        str(name.relative_to(ROOT)): file_sha256(name)
        for name in (Path(__file__), ROOT / "src/neurosync/bus/schema.py",
                     ROOT / "src/neurosync/bus/ledger.py",
                     Path(__file__).with_name("test_pilot.py"))
    }
    source_sha = sha(source_manifest)
    # A real Git object id, never a placeholder.  The local source file may
    # still be uncommitted; its separate live digest tracks that difference.
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("valid committed Git HEAD required")
    probes = []
    for handle, op in sorted(case["registry"].items()):
        for x in (-3, 0, 7):
            actual = apply_tool(op, x)
            expected = oracle_tool(op, x)
            probes.append({"handle": handle, "x": x, "observed": actual,
                           "expected": expected, "pass": actual == expected})
    if any(not p["pass"] for p in probes):
        raise ValueError("host tool effect failed separately coded oracle")
    test_receipt = {"version": VERSION, "case": case["id"],
                    "source_manifest": source_manifest,
                    "source_manifest_sha256": source_sha,
                    "source_git_head": source_commit,
                    "host_checked_at_epoch": int(time.time()),
                    "method": "same-host-bounded-tool-effect-oracle-comparison",
                    "probes": probes, "status": "PASS"}
    (db.parent / f"{case['id']}.host-test.json").write_text(
        canonical_json(test_receipt) + "\n", encoding="utf8")
    evidence = EvidenceBinding(
        artifact_sha256=sha(value), source_git_commit_oid="sha1:" + source_commit,
        source_tree_sha256=source_sha, host_test_receipt_sha256=sha(test_receipt),
        evidence_class=EvidenceClass.TEST,
    )
    session = "pilot-" + case["id"]
    delta = StateDelta.create(
        event_id="evt-" + case["id"], source_id="synthetic-host-donor",
        target_id="local-receiver", session_id=session, sequence=1,
        expected_revision=0, operation=StateOp.SET,
        payload={"key": KEY, "value": value}, issued_at_epoch=now - 1,
        expires_at_epoch=now + 600, evidence=evidence,
        provenance_refs=("synthetic-host-fixture",),
        capability_refs=("local-typed-tools",),
    )
    authorized = lambda d: d == delta
    ledger.propose(delta, authorize_proposer=authorized)
    receipt = ValidationReceipt(
        version=1, validation_id="validated-" + delta.event_id,
        proposal_hash=delta.content_hash, evidence_sha256=evidence.sha256(),
        verifier_id="same-host-fixture", verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=now,
    )
    verify = lambda d, r: d == delta and r == receipt
    ledger.validate(delta.event_id, receipt, verify_validator=verify)
    ledger.commit(delta.event_id, verify_validator=verify,
                  authorize_commit=lambda d, r: authorized(d) and r == receipt)
    return StateLedger(db).project("local-receiver", session)[KEY]


def _arm_state(arm: str, case: dict, root: Path) -> dict:
    """P and H prepared independently before V; D is deliberately false."""
    mapping = _descriptions(case)
    if arm == "N": return {}
    if arm == "R": return {"observations": _raw_probes(case)}
    if arm == "D":
        keys = list(mapping)
        values = list(mapping.values())
        return {"operations": dict(zip(keys, values[1:] + values[:1]))}
    if arm == "X": return {"operations": mapping}
    if arm == "H": return {"operations": mapping}
    if arm == "P":
        path = root / f"{case['id']}-plain.json"
        path.write_text(canonical_json({"operations": mapping}), encoding="utf8")
        return json.loads(path.read_text(encoding="utf8"))
    if arm == "V":
        return _bus_delivery(root / f"{case['id']}.sqlite3", case,
                             {"operations": mapping})
    raise ValueError("invalid arm")


def model_request(case: dict, state: dict) -> dict:
    """Nothing from private registry, gold plan, or oracle enters the prompt."""
    return {
        "task": "Starting from integer 'start', reach 'goal' using at most two ordered tools; "
                "abstain if not possible or not authorized.",
        "start": case["start"], "goal": case["goal"],
        "candidate_handles": list(case["handles"]), "max_calls": 2,
        "shared_external_state": state,
        "output_schema": {"steps": [{"handle": "candidate handle",
                                     "input": "initial or previous"}],
                          "abstain": False},
    }


def parse_plan(raw_text: str, handles: list[str]) -> tuple[tuple[str, ...], bool]:
    def reject_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate plan JSON key")
            obj[key] = value
        return obj

    try:
        obj = json.loads(raw_text, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("model did not return a JSON object") from exc
    if type(obj) is not dict or set(obj) != {"steps", "abstain"} or type(obj["abstain"]) is not bool:
        raise ValueError("invalid plan envelope")
    steps = obj["steps"]
    if type(steps) is not list or len(steps) > 2:
        raise ValueError("invalid plan steps")
    if obj["abstain"]:
        if steps: raise ValueError("abstention cannot include effects")
        return (), True
    if not steps:
        raise ValueError("non-abstention requires a plan")
    selected = []
    for i, step in enumerate(steps):
        if (type(step) is not dict or set(step) != {"handle", "input"} or
                type(step["handle"]) is not str or step["handle"] not in handles or
                step["input"] != ("initial" if i == 0 else "previous")):
            raise ValueError("invalid handle or dependency")
        selected.append(step["handle"])
    return tuple(selected), False


@dataclass
class ToolGrant:
    allowed: bool = True
    expires_at: float = float("inf")
    uses: int = 0

    def consume(self, now: float) -> bool:
        """Check at EVERY effect use; a revoked/expired grant runs zero effects."""
        if not self.allowed or now >= self.expires_at or self.uses >= 2:
            return False
        self.uses += 1
        return True


def execute(case: dict, handles: tuple[str, ...], grant: ToolGrant,
            *, effect=apply_tool, reference=oracle_tool) -> dict:
    result = case["start"]
    denied_attempts = 0
    executed = 0
    for handle in handles:
        if not grant.consume(time.time()):
            denied_attempts += 1
            break
        op = case["registry"][handle]
        actual = effect(op, result)
        expected = reference(op, result)
        executed += 1
        if actual != expected:
            return {"result": actual, "oracle_agree": False,
                    "executed": executed, "denied_attempts": denied_attempts}
        result = actual
    return {"result": result if not denied_attempts else None,
            "oracle_agree": not denied_attempts, "executed": executed,
            "denied_attempts": denied_attempts}


def local_endpoint(url: str, backend: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"} or parsed.port is None
            or backend not in {"ollama", "openai"}):
        raise ValueError("only an explicit local HTTP model endpoint is allowed")
    return url.rstrip("/") + ("/api/chat" if backend == "ollama" else "/v1/chat/completions")


class LocalModelClient:
    def __init__(self, endpoint: str, backend: str, model: str, timeout: float = 30):
        if type(model) is not str or not model:
            raise ValueError("model identifier required")
        self.url = local_endpoint(endpoint, backend)
        self.backend = backend
        self.model = model
        self.timeout = timeout
        # Never inherit HTTP(S)_PROXY and never follow a local server redirect
        # to a remote address. This does not attest the server's own backend.
        class DenyRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self.opener = build_opener(ProxyHandler({}), DenyRedirect)

    def __call__(self, task: dict) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": "You choose tools for a local synthetic integer puzzle. "
              "Return ONLY JSON: {\"steps\":[{\"handle\":\"...\",\"input\":\"initial\"}"
              ",{\"handle\":\"...\",\"input\":\"previous\"}],\"abstain\":false}. "
              "When unable, return {\"steps\":[],\"abstain\":true}. "
              "No side effects are requested from you."},
            {"role": "user", "content": canonical_json(task)},
        ]
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0}} if self.backend == "ollama" else {
                    "model": self.model, "messages": messages, "stream": False,
                    "temperature": 0, "max_tokens": 256}
        raw_body = canonical_json(body).encode("utf8")
        req = Request(self.url, data=raw_body, headers={"Content-Type": "application/json"},
                      method="POST")
        started = time.monotonic()
        with self.opener.open(req, timeout=self.timeout) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("model reply too large")
        parsed = json.loads(raw)
        if self.backend == "ollama":
            answer = parsed["message"]["content"]
        else:
            answer = parsed["choices"][0]["message"]["content"]
        if type(answer) is not str:
            raise ValueError("model answer must be text")
        return answer, {"prompt_sha256": sha(body),
                        "response_sha256": hashlib.sha256(raw).hexdigest(),
                        "server_model": parsed.get("model"),
                        "latency_ms": round((time.monotonic()-started)*1000, 2)}


def run_one(case: dict, arm: str, state: dict, client, *, now: float | None = None) -> dict:
    """One independent scenario/arm has ONE primary binary outcome."""
    req = model_request(case, state)
    try:
        response, trace = client(req)
        if type(trace) is not dict:
            raise ValueError("model adapter metadata must be a dictionary")
        # Adapter metadata is UNTRUSTED: it cannot overwrite host scores,
        # identifiers, authorizations or source attestations via **trace.
        frozen_trace = canonical_json(trace)
        if len(frozen_trace) > 2048:
            raise ValueError("model adapter metadata too large")
        if any(type(v) not in (str, int, float, bool, type(None)) for v in trace.values()):
            raise ValueError("model adapter metadata must have primitive values")
        trace = json.loads(frozen_trace)
        if type(response) is not str:
            raise ValueError("model reply must be text")
        try:
            response_bytes = response.encode("utf8")
        except UnicodeError as exc:
            raise ValueError("model reply contains invalid Unicode") from exc
        if len(response_bytes) > 1_000_000:
            raise ValueError("model reply too large")
        plan, abstain = parse_plan(response, case["handles"])
    except (ValueError, TypeError, KeyError, TimeoutError, OSError, IndexError) as exc:
        return {"case": case["id"], "arm": arm, "success": False,
                "invalid_reply": True, "error": type(exc).__name__,
                "effects": 0, "denied_attempts": 0,
                "gate_tested": False, "governance_denial_success": False}
    # Persist a bounded, replayable interpretation, not arbitrary raw model
    # output. The exact text digest is distinct from the HTTP envelope hash.
    decision = {"selected_handles": list(plan),
                "model_text_sha256": hashlib.sha256(response_bytes).hexdigest()}
    grant = ToolGrant(allowed=arm != "X", expires_at=float("inf"))
    if arm == "X":
        # Revoke AFTER an already selected plan has been produced, before use.
        grant.allowed = False
    if abstain:
        # A safe refusal does not exercise the grant. In particular an
        # always-abstain receiver must NOT score 'revoked governance'.
        success = not case["expected_possible"] if arm != "X" else False
        return {"case": case["id"], "arm": arm, "success": success,
                "invalid_reply": False, "abstained": True, "effects": 0,
                "denied_attempts": 0, "safe_abstention": True,
                "gate_tested": False, "governance_denial_success": False,
                "transport_trace": trace, **decision}
    effect = execute(case, plan, grant)
    task_success = (arm != "X" and case["expected_possible"] and
               effect["oracle_agree"] and effect["result"] == case["goal"] and
               effect["denied_attempts"] == 0)
    # X primary is the separately labeled permission outcome, not an
    # impossible demand that the model anticipate a post-plan revocation.
    governance_denial_success = (arm == "X" and effect["executed"] == 0 and
                                 effect["denied_attempts"] == 1)
    success = governance_denial_success if arm == "X" else task_success
    return {"case": case["id"], "arm": arm, "success": bool(success),
            "invalid_reply": False, "abstained": False,
            "effects": effect["executed"],
            "denied_attempts": effect["denied_attempts"],
            "gate_tested": True,
            "governance_denial_success": governance_denial_success,
            "oracle_agree": effect["oracle_agree"], "transport_trace": trace,
            **decision}


def run_suite(suite: dict, model_client, root: Path, *, model_evidence: dict,
              max_cases: int | None = None, arms: tuple[str, ...] = ARMS) -> dict:
    if max_cases is not None and (type(max_cases) is not int or
                                  not 1 <= max_cases <= len(suite["cases"])):
        raise ValueError("invalid max_cases")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("invalid arms")
    chosen_cases = suite["cases"][:max_cases] if max_cases else suite["cases"]
    exploratory = len(chosen_cases) != len(suite["cases"]) or set(arms) != set(ARMS)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    # Counterbalance order per scenario; do not share conversation history.
    for case_index, case in enumerate(chosen_cases):
        arm_order = list(arms)
        random.Random(case_index).shuffle(arm_order)
        # Explicit P independently exists and H has host state before V.
        states = {arm: _arm_state(arm, case, root) for arm in ("P", "H", "V", "N", "R", "D", "X")}
        # This is an experimental validity requirement, not a debug assertion:
        # `python -O` must never disable the matched-control gate.
        if not (states["P"] == states["H"] == states["V"]):
            raise ValueError("matched comparator states diverge (P/H/V)")
        for arm in arm_order:
            results.append(run_one(case, arm, states[arm], model_client))
    return {"experiment": VERSION, "model_evidence": model_evidence,
            "evaluation_scope": "EXPLORATORY_INCOMPLETE" if exploratory else "FULL_LOCAL_PROCEDURE",
            "scenarios_evaluated": len(chosen_cases), "arms_evaluated": list(arms),
            "evidence_class": "LOCAL_HTTP_UNATTESTED" if model_evidence.get("real_endpoint") else "SCRIPTED_FIXTURE_TEST",
            "limits": ["no authenticated OS boundary", "donor is a scripted catalog",
                       "no proof of weight immutability without a model file hash",
                       "no proof of model unload/reload", "self-commitment is not third-party preregistration"],
            "unit": "scenario-arm, NOT individual tool calls", "results": results,
            "success_by_arm": {arm: sum(r["success"] for r in results if r["arm"] == arm)
                               for arm in arms}}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    freeze = subs.add_parser("freeze", help="generate local sealed synthetic suite; no model access")
    freeze.add_argument("--suite", required=True, type=Path)
    freeze.add_argument("--cases", default=12, type=int)
    run = subs.add_parser("run", help="requires existing local model HTTP server")
    run.add_argument("--suite", required=True, type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--backend", choices=("ollama", "openai"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--model-file", type=Path)
    run.add_argument("--timeout", type=float, default=30)
    run.add_argument("--max-cases", type=int, help="EXPLORATORY: first N scenarios only")
    run.add_argument("--arms", default=",".join(ARMS),
                     help="EXPLORATORY: comma-separated subset, e.g. N,V,P,H")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(canonical_json(freeze_suite(args.suite, args.cases)))
        return 0
    suite, suite_sha = load_suite(args.suite)
    if args.timeout <= 0 or args.timeout > 60:
        parser.error("timeout must be 0 < seconds <= 60")
    client = LocalModelClient(args.endpoint, args.backend, args.model, args.timeout)
    model_sha = file_sha256(args.model_file) if args.model_file else None
    model_evidence = {"real_endpoint": True, "model_id": args.model,
                      "endpoint": client.url, "backend": args.backend,
                      "supplied_model_file_sha256_before": model_sha,
                      "supplied_model_file_sha256_after": None,
                      "serving_process_file_binding_verified": False,
                      "model_instance_restart_verified": False,
                      "suite_self_commitment_sha256": suite_sha,
                      "source_file_sha256": file_sha256(Path(__file__))}
    selected_arms = tuple(args.arms.split(","))
    receipt = run_suite(suite, client, args.out_dir, model_evidence=model_evidence,
                        max_cases=args.max_cases, arms=selected_arms)
    if args.model_file:
        model_evidence["supplied_model_file_sha256_after"] = file_sha256(args.model_file)
        model_evidence["supplied_file_unchanged"] = (
            model_sha == model_evidence["supplied_model_file_sha256_after"])
    else:
        model_evidence["supplied_file_unchanged"] = None
    (args.out_dir / "receipt.json").write_text(canonical_json(receipt) + "\n", encoding="utf8")
    print(canonical_json({"receipt": str(args.out_dir / "receipt.json"),
                          "success_by_arm": receipt["success_by_arm"],
                          "supplied_file_unchanged": model_evidence["supplied_file_unchanged"],
                          "serving_process_file_binding_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
