"""Source-only, zero-weight StateBus capability-transfer experiment.

Both nodes are deterministic *mock computational adapters*, not language models.
The receiver process only gets its immutable adapter, a SQLite path, and a task.
The StateBus payload holds a capability reference and expiry, never a task answer.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from neurosync.bus import (  # noqa: E402
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger, StateOp,
    ValidationReceipt, canonical_json,
)


CAP_REF = "cap-8e7b3c1a-ast-v1"  # opaque reference, no task-specific content
STATE_KEY = "capabilities.formatter"
POLICY = "ast-canonical-v1"
SOURCE_FILES = ("experiments/zero-weight-sync/run.py",
                "src/neurosync/bus/schema.py", "src/neurosync/bus/ledger.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_file(path: Path) -> str:
    return sha(path.read_bytes())


def source_manifest_sha256() -> str:
    """Digest of this experiment + the exact two StateBus core source files.

    This is a *scoped source manifest*, not a Git tree hash or full-repo digest.
    """
    return sha(canonical_json({name: canonical_file(REPO / name)
                               for name in SOURCE_FILES}).encode())


def trusted_source_head(requested_head: str | None) -> str:
    """Require a real HEAD commit and exact committed bytes for scoped sources."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=REPO, text=True).strip()
    if requested_head is not None and requested_head != head:
        raise ValueError("caller-provided Git OID is not trusted current HEAD")
    obj_type = subprocess.check_output(["git", "cat-file", "-t", head],
                                       cwd=REPO, text=True).strip()
    if obj_type != "commit":
        raise ValueError("source OID is not an actual commit")
    for name in SOURCE_FILES:
        archived = subprocess.run(["git", "show", f"{head}:{name}"],
                                  cwd=REPO, capture_output=True, check=True).stdout
        if archived != (REPO / name).read_bytes():
            raise ValueError(f"scoped source differs from committed HEAD: {name}")
    return head


def normalize(source: str) -> str:
    """Host-owned fixture capability; model adapter does not contain its answer."""
    if len(source.encode("utf8")) > 2048:
        raise ValueError("task source exceeds 2048 bytes")
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("task AST exceeds 256 nodes")
    return ast.unparse(tree)


HOST_CAPABILITIES = {CAP_REF: normalize}
_PINNED_HANDLER = normalize


def handler_sha256(tool) -> str:
    """Fixture-level source pin, not a runtime attestation or code signature."""
    try:
        return sha(inspect.getsource(tool).encode("utf8"))
    except (OSError, TypeError):
        return ""


def model_artifact(path: Path, *, node: str, style: str) -> None:
    if style not in {"terse", "annotated"}:
        raise ValueError("unsupported mock adapter style")
    path.write_text(canonical_json({"node": node, "style": style,
                                    "kind": "frozen-mock-adapter-v1"}), encoding="utf8")


def load_artifact(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf8"))
    if set(value) != {"node", "style", "kind"} or value["kind"] != "frozen-mock-adapter-v1":
        raise ValueError("not a permitted frozen adapter")
    if value["style"] not in {"terse", "annotated"}:
        raise ValueError("unsupported adapter style")
    return value


def donor_observe(donor_artifact: Path, ref: str) -> dict:
    """A-origin, task-blind, *scripted mock* discovery and actual tool probe."""
    donor = load_artifact(donor_artifact)
    if donor["node"] != "node-A" or donor["style"] != "annotated":
        raise ValueError("observation must originate from the designated A adapter")
    tool = HOST_CAPABILITIES.get(ref)
    if tool is None or tool is not _PINNED_HANDLER:
        raise ValueError("A cannot observe unavailable or rebound capability")
    probe = {"source": "out=(7+2)"}
    observed = {"ok": True, "artifact": tool(probe["source"])}
    donor_sha = canonical_file(donor_artifact)
    proof = host_verify(probe, observed, donor_sha)
    if not proof["passed"]:
        raise ValueError("A observation failed host check")
    return {"ref": ref, "donor_artifact_sha256": donor_sha,
            "probe_receipt_sha256": sha(canonical_json(proof).encode())}


def execute(db_path: Path, artifact: Path, task: dict, *, session: str,
            target: str = "shared-runtime",
            now: int | None = None) -> dict:
    """The receiver has NO donor object, teacher probabilities or network RPC."""
    node = load_artifact(artifact)
    if set(task) != {"source"} or not isinstance(task["source"], str):
        raise ValueError("task must contain source only")
    state = StateLedger(db_path).project(target, session)
    capability = state.get(STATE_KEY)
    if not isinstance(capability, dict):
        return {"ok": False, "reason": "missing-capability", "node": node["node"]}
    if set(capability) != {"ref", "policy", "valid_until", "handler_sha256",
                           "donor_artifact_sha256"}:
        return {"ok": False, "reason": "invalid-state", "node": node["node"]}
    if capability["policy"] != POLICY or not isinstance(capability["valid_until"], int):
        return {"ok": False, "reason": "policy-mismatch", "node": node["node"]}
    if capability["valid_until"] <= (int(time.time()) if now is None else now):
        return {"ok": False, "reason": "stale-capability", "node": node["node"]}
    tool = HOST_CAPABILITIES.get(capability["ref"])
    if tool is None:
        return {"ok": False, "reason": "unknown-capability", "node": node["node"]}
    if tool is not _PINNED_HANDLER or capability["handler_sha256"] != handler_sha256(tool):
        return {"ok": False, "reason": "handler-mismatch", "node": node["node"]}
    try:
        result = tool(task["source"])
    except (SyntaxError, ValueError, OverflowError, RecursionError, UnicodeError):
        return {"ok": False, "reason": "invalid-task", "node": node["node"]}
    style = node["style"]
    return {"ok": True, "artifact": result, "node": node["node"],
            "display": result if style == "terse" else f"Formatted source:\n{result}",
            "style": style}


def host_verify(task: dict, answer: dict, model_sha256: str) -> dict:
    """Host-executed checker, not a flag/answer supplied by the receiver.

    Its AST equivalence + canonical-form assertions run against actual output.
    This shares Python's AST implementation with the toy tool, so it is *not*
    an independent security principal or independent compiler implementation.
    """
    ok = False
    try:
        original = ast.parse(task["source"])
        candidate = ast.parse(answer["artifact"])
        ok = (answer.get("ok") is True and
              ast.dump(original, include_attributes=False) ==
              ast.dump(candidate, include_attributes=False) and
              answer["artifact"] == ast.unparse(candidate))
    except (ValueError, KeyError, TypeError, SyntaxError):
        pass
    return {"passed": ok, "input_sha256": sha(canonical_json(task).encode()),
            "output_sha256": sha(canonical_json(answer).encode()),
            "mock_adapter_artifact_sha256": model_sha256,
            "source_tree_sha256": source_manifest_sha256(),
            "policy_sha256": sha(POLICY.encode()),
            "checker": "host-python-ast-assertions-v1"}


def grant(ledger: StateLedger, donor_artifact: Path, *, source_head: str,
          session: str, recipient: str, valid_until: int,
          ref: str = CAP_REF, grant_open: dict | None = None,
          now: int | None = None, revoke_after_validate: bool = False) -> StateDelta:
    """Host-owned authority fixture; callbacks are NOT real authentication."""
    now = int(time.time()) if now is None else now
    source_head = trusted_source_head(source_head)
    observed = donor_observe(donor_artifact, ref)
    bound_sha = observed["donor_artifact_sha256"]
    tree_sha = source_manifest_sha256()
    receipt_sha = observed["probe_receipt_sha256"]
    evidence = EvidenceBinding(
        artifact_sha256=bound_sha, source_git_commit_oid="sha1:" + source_head,
        source_tree_sha256=tree_sha, host_test_receipt_sha256=receipt_sha,
        evidence_class=EvidenceClass.TEST,
    )
    delta = StateDelta.create(
        event_id=f"evt-{session}-{recipient}", source_id="node-A",
        target_id=recipient, session_id=session, sequence=1, expected_revision=0,
        operation=StateOp.SET,
        payload={"key": STATE_KEY, "value": {"ref": ref, "policy": POLICY,
                                               "handler_sha256": handler_sha256(_PINNED_HANDLER),
                                               "donor_artifact_sha256": bound_sha,
                                               "valid_until": valid_until}},
        issued_at_epoch=now - 1, expires_at_epoch=valid_until,
        evidence=evidence, provenance_refs=("host-fixture-source",),
        capability_refs=("capability-formatter",),
    )
    current = grant_open if grant_open is not None else {"allowed": True}

    def recognized(d: StateDelta, r: ValidationReceipt) -> bool:
        return (r.proposal_hash == d.content_hash and
                r.evidence_sha256 == d.evidence.sha256() and
                d.evidence.artifact_sha256 == canonical_file(donor_artifact) and
                d.payload["value"]["donor_artifact_sha256"] == bound_sha and
                d.evidence.source_tree_sha256 == source_manifest_sha256() and
                d.evidence.host_test_receipt_sha256 == receipt_sha and
                d.evidence.source_git_commit_oid == "sha1:" + source_head and
                r.verifier_id == "host-checker")

    ledger.propose(delta, authorize_proposer=lambda d: current["allowed"] and d == delta)
    validation = ValidationReceipt(
        version=1, validation_id=f"validated-{delta.event_id}",
        proposal_hash=delta.content_hash, evidence_sha256=delta.evidence.sha256(),
        verifier_id="host-checker", verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=now,
    )
    ledger.validate(delta.event_id, validation, verify_validator=recognized)
    if revoke_after_validate:
        current["allowed"] = False
    ledger.commit(delta.event_id, verify_validator=recognized,
                  authorize_commit=lambda d, r: current["allowed"] and d == delta)
    return delta


def receiver_subprocess(db_path: Path, artifact: Path, task: dict, session: str,
                        target: str = "shared-runtime") -> dict:
    """Only SQLite path, B adapter path, session and task cross process boundary."""
    command = [sys.executable, str(Path(__file__)), "receive", "--db", str(db_path),
         "--artifact", str(artifact), "--session", session, "--target", target]
    proc = subprocess.run(
        command,
        input=canonical_json(task), text=True, capture_output=True, check=True, timeout=5,
    )
    return json.loads(proc.stdout)


def run_demo() -> dict:
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        donor = base / "node-a.json"
        receiver = base / "node-b.json"
        db = base / "state.sqlite3"
        model_artifact(donor, node="node-A", style="annotated")
        model_artifact(receiver, node="node-B", style="terse")
        original_sha = canonical_file(receiver)
        source_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        task = {"source": "answer= (3+4) * 2"}  # chosen after grant, not in state
        before = receiver_subprocess(db, receiver, task, "pilot")
        now = int(time.time())
        grant(StateLedger(db), donor, source_head=source_head,
              session="pilot", recipient="shared-runtime", valid_until=now + 300, now=now)
        after = receiver_subprocess(db, receiver, task, "pilot")
        other_node = receiver_subprocess(db, donor, task, "pilot")
        verifier = host_verify(task, after, original_sha)
        return {"no_state": before, "valid_state": after,
                "other_style_same_governed_state": other_node,
                "host_verifier": verifier, "unchanged_mock_adapter_artifact":
                original_sha == canonical_file(receiver),
                "source_git_commit_oid": "sha1:" + source_head,
                "source_tree_sha256": source_manifest_sha256()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("demo", "receive", "verify"))
    parser.add_argument("--db", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--target", default="shared-runtime")
    args = parser.parse_args()
    if args.mode == "demo":
        outcome = run_demo()
    elif args.mode == "receive":
        raw = sys.stdin.read(8193)
        if len(raw) > 8192:
            outcome = {"ok": False, "reason": "invalid-task"}
        else:
            outcome = execute(args.db, args.artifact, json.loads(raw),
                              session=args.session, target=args.target)
    else:
        submission = json.loads(sys.stdin.read(16385))
        outcome = host_verify(submission["task"], submission["answer"],
                              canonical_file(args.artifact))
    print(canonical_json(outcome))


if __name__ == "__main__":
    main()
