"""Randomized causal controls for source-only, zero-weight capability routing.

Both participants are deterministic scripted mock adapters. This tests whether
A's *task-blind discovery* contributes a usable opaque tool reference to B,
not whether a neural model learns, remembers, or exchanges parameters.

The operator-owned registry and grant file are one local trust domain, not
authenticated authorities. In particular, an operator with the correct ref
can obtain the same functional result through ordinary JSON retrieval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as pilot  # noqa: E402 - existing StateBus source fixture

from neurosync.bus import (  # noqa: E402
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger,
    StateOp, ValidationReceipt, canonical_json,
)
from neurosync.bus.ledger import LedgerIntegrityError  # noqa: E402

STATE_KEY = "capabilities.affine"
POLICY = "affine-routing-v1"
PROBES = (-7, 13, 29)
MAX_BYTES = 8192
SOURCE_FILES = tuple(dict.fromkeys((*pilot.SOURCE_FILES,
    "experiments/zero-weight-sync/causal_trial.py",
    "experiments/zero-weight-sync/test_causal_trial.py")))


def digest(obj: object) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf8")).hexdigest()


def source_manifest_sha256() -> str:
    """Scoped source digest INCLUDING this evaluator and its test policy."""
    return digest({path: pilot.canonical_file(pilot.REPO / path)
                   for path in SOURCE_FILES})


def checked_source_head(head: str) -> str:
    """Reject fake OIDs and any uncommitted causal implementation/test bytes.

    The committed test file is a policy *fixture*, not an independent test
    authority. A malicious repository committer can still change both code
    and tests; a signed external approved policy is outside this experiment.
    """
    head = pilot.trusted_source_head(head)
    for path in SOURCE_FILES:
        data = subprocess.run(["git", "show", f"{head}:{path}"],
                              cwd=pilot.REPO, capture_output=True, check=True).stdout
        if data != (pilot.REPO / path).read_bytes():
            raise ValueError("causal source/policy differs from committed Git HEAD: " + path)
    return head


def bounded_json(path: Path) -> object:
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("fixture input exceeds limit")
    def no_duplicate_keys(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=no_duplicate_keys)


def write_json(path: Path, obj: object) -> None:
    path.write_text(canonical_json(obj), encoding="utf8")


@dataclass(frozen=True)
class Scenario:
    """Synthetic hidden target; never serialized into B's task or StateBus."""

    seed: int
    registry: dict[str, tuple[int, int]]
    target_ref: str

    @classmethod
    def generate(cls, seed: int, count: int = 13) -> "Scenario":
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("invalid synthetic seed")
        if not 3 <= count <= 32:
            raise ValueError("candidate count out of range")
        rng = random.Random(seed)
        mapping: dict[str, tuple[int, int]] = {}
        pairs: set[tuple[int, int]] = set()
        while len(mapping) < count:
            a, b = rng.randrange(2, 400), rng.randrange(-400, 401)
            if (a, b) in pairs:
                continue
            pairs.add((a, b))
            ref = f"cap-{rng.getrandbits(64):016x}"
            if ref not in mapping:
                mapping[ref] = (a, b)
        target_ref = list(mapping)[rng.randrange(count)]
        return cls(seed, mapping, target_ref)

    def private_signature(self) -> tuple[int, ...]:
        """A-only challenge, distinct from held-out tasks."""
        return tuple(self.evaluate(self.target_ref, x) for x in PROBES)

    def donor_view(self) -> "DonorView":
        """A's restricted data view lacks the host-sealed target reference.

        This is a structural boundary within one Python fixture, not a
        separate security principal. A receives probes and candidates only.
        """
        return DonorView(tuple((ref, pair) for ref, pair in self.registry.items()),
                         self.private_signature())

    def evaluate(self, ref: str, x: int) -> int:
        a, b = self.registry[ref]
        return a * x + b

    def registry_wire(self) -> dict[str, list[int]]:
        # The host-owned tool registry contains all handlers but no target label.
        return {ref: list(pair) for ref, pair in self.registry.items()}

    def heldout(self, n: int = 12) -> list[int]:
        rng = random.Random(self.seed ^ 0xA52767B)
        values: set[int] = set()
        while len(values) < n:
            x = rng.randrange(-100_000, 100_001)
            if x not in PROBES:
                values.add(x)
        return sorted(values)


@dataclass(frozen=True)
class DonorView:
    candidates: tuple[tuple[str, tuple[int, int]], ...]
    signature: tuple[int, ...]

    def __post_init__(self):
        if len(self.signature) != len(PROBES) or not all(type(x) is int for x in self.signature):
            raise ValueError("invalid donor probe signature")
        if not 3 <= len(self.candidates) <= 32:
            raise ValueError("invalid donor candidate count")

    def probe(self, ref: str, x: int) -> int:
        """A can query candidate tools, but has no sealed target label."""
        for candidate, coefficients in self.candidates:
            if candidate == ref:
                a, b = coefficients
                return a * x + b
        raise KeyError("unknown candidate")


def a_discover(donor_artifact: Path, view: DonorView) -> dict:
    """A probes each opaque ref against its *private* three-point signature.

    Host runs this scripted donor; no model inference, security separation,
    cryptographic proof of A identity, or autonomous tool invention is implied.
    """
    donor = pilot.load_artifact(donor_artifact)
    if donor != {"node": "node-A", "style": "annotated", "kind": "frozen-mock-adapter-v1"}:
        raise ValueError("observation requires designated donor fixture")
    if not isinstance(view, DonorView):
        raise TypeError("donor receives restricted DonorView, not host Scenario")
    signature = view.signature
    matches = [ref for ref, _ in view.candidates if
               tuple(view.probe(ref, x) for x in PROBES) == signature]
    if len(matches) != 1:
        raise ValueError("ambiguous or absent capability match")
    return {"selected_ref": matches[0], "probe_inputs": list(PROBES),
            "probe_outputs": list(signature),
            "donor_artifact_sha256": pilot.canonical_file(donor_artifact),
            "registry_sha256": digest({ref: list(pair) for ref, pair in view.candidates})}


def host_validate_discovery(scenario: Scenario, observation: object,
                            donor_artifact: Path) -> bool:
    """Sealed synthetic oracle in the same host fixture, not remote trust."""
    if not isinstance(observation, dict) or set(observation) != {
        "selected_ref", "probe_inputs", "probe_outputs", "donor_artifact_sha256",
        "registry_sha256",
    }:
        return False
    if (not isinstance(observation["probe_inputs"], list) or
            not isinstance(observation["probe_outputs"], list) or
            len(observation["probe_inputs"]) != len(PROBES) or
            len(observation["probe_outputs"]) != len(PROBES) or
            not all(type(x) is int for x in observation["probe_inputs"] + observation["probe_outputs"])):
        return False
    return (observation["selected_ref"] == scenario.target_ref and
            observation["probe_inputs"] == list(PROBES) and
            observation["probe_outputs"] == list(scenario.private_signature()) and
            observation["donor_artifact_sha256"] == pilot.canonical_file(donor_artifact) and
            observation["registry_sha256"] == digest(scenario.registry_wire()))


def capability_value(observation: dict, valid_until: int) -> dict:
    """Task-blind profile identical for StateBus and plain JSON baseline."""
    return {"ref": observation["selected_ref"], "valid_until": valid_until,
            "catalog_sha256": observation["registry_sha256"],
            "donor_artifact_sha256": observation["donor_artifact_sha256"],
            "policy": POLICY}


def grant_via_bus(db: Path, broker: Path, scenario: Scenario, donor: Path,
                  source_head: str, session: str, target: str,
                  *, observation: dict | None, now: int, valid_until: int) -> dict:
    """Commit only a host-validated A observation and register use-time grant.

    The broker sidecar is local test authority, not a signature or production
    access-control store. Revocation is checked by B at *every use*.
    """
    source_head = checked_source_head(source_head)
    if not host_validate_discovery(scenario, observation, donor):
        raise PermissionError("no valid A observation")
    value = capability_value(observation, valid_until)
    manifest = source_manifest_sha256()
    host_probe_receipt = {
        "checked_source_git_oid": "sha1:" + source_head,
        "causal_source_manifest_sha256": manifest,
        "test_policy_file_sha256": pilot.canonical_file(
            pilot.REPO / "experiments/zero-weight-sync/test_causal_trial.py"),
        "A_observation_sha256": digest(observation),
        "oracle_check_passed": True,
        "checker": "scripted-affine-host-oracle-v2",
    }
    evidence = EvidenceBinding(
        artifact_sha256=pilot.canonical_file(donor),
        source_git_commit_oid="sha1:" + source_head,
        source_tree_sha256=manifest,
        host_test_receipt_sha256=digest(host_probe_receipt),
        evidence_class=EvidenceClass.TEST,
    )
    delta = StateDelta.create(
        event_id=f"evt-affine-{session}", source_id="node-A", target_id=target,
        session_id=session, sequence=1, expected_revision=0,
        operation=StateOp.SET, payload={"key": STATE_KEY, "value": value},
        issued_at_epoch=now - 1, expires_at_epoch=valid_until,
        evidence=evidence, provenance_refs=("scripted-A-probe",),
        capability_refs=("capability-affine",),
    )
    ledger = StateLedger(db)
    authorize = lambda d: d == delta and host_validate_discovery(scenario, observation, donor)
    ledger.propose(delta, authorize_proposer=authorize)
    receipt = ValidationReceipt(
        version=1, validation_id=f"validated-{delta.event_id}",
        proposal_hash=delta.content_hash, evidence_sha256=evidence.sha256(),
        verifier_id="host-oracle", verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=now,
    )
    verify = lambda d, r: (d == delta and r.verifier_id == "host-oracle" and
                            r.proposal_hash == delta.content_hash and
                            host_validate_discovery(scenario, observation, donor))
    ledger.validate(delta.event_id, receipt, verify_validator=verify)
    ledger.commit(delta.event_id, verify_validator=verify,
                  authorize_commit=lambda d, r: authorize(d) and r == receipt)
    write_json(broker, {"active": True, "session": session,
                        "target": target, "ref": value["ref"],
                        "catalog_sha256": value["catalog_sha256"],
                        "donor_artifact_sha256": value["donor_artifact_sha256"]})
    return value


def revoke(broker: Path) -> None:
    """Synthetic host-owned use-time grant revocation."""
    state = bounded_json(broker)
    if not isinstance(state, dict) or state.get("active") is not True:
        raise ValueError("no active grant to revoke")
    write_json(broker, {**state, "active": False})


def receive(*, transport: str, registry_path: Path, broker_path: Path,
            receiver_artifact: Path, session: str, target: str, x: int,
            db_path: Path | None = None, state_path: Path | None = None,
            now: int | None = None) -> dict:
    """B's only allowed operation is invoking the referenced host handler."""
    node = pilot.load_artifact(receiver_artifact)
    if node["node"] != "node-B":
        return {"ok": False, "reason": "wrong-receiver"}
    if isinstance(x, bool) or not isinstance(x, int) or abs(x) > 100_000:
        return {"ok": False, "reason": "invalid-task"}
    try:
        registry = bounded_json(registry_path)
        if not isinstance(registry, dict) or not 3 <= len(registry) <= 32:
            raise ValueError("invalid registry")
        if not all(isinstance(k, str) and k.startswith("cap-") and
                   isinstance(v, list) and len(v) == 2 and
                   all(type(a) is int and abs(a) <= 1000 for a in v)
                   for k, v in registry.items()):
            raise ValueError("invalid registry entry")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"ok": False, "reason": "invalid-registry"}
    if transport == "statebus":
        if db_path is None:
            return {"ok": False, "reason": "missing-state"}
        try:
            state = StateLedger(db_path).project(target, session).get(STATE_KEY)
        except (LedgerIntegrityError, ValueError, TypeError):
            return {"ok": False, "reason": "invalid-ledger"}
    elif transport == "file":
        if state_path is None or not state_path.is_file():
            state = None
        else:
            try:
                state = bounded_json(state_path)
            except (OSError, ValueError, json.JSONDecodeError):
                return {"ok": False, "reason": "invalid-state-file"}
    elif transport == "none":
        state = None
    else:
        return {"ok": False, "reason": "unsupported-transport"}
    if not isinstance(state, dict) or set(state) != {
        "ref", "valid_until", "catalog_sha256", "donor_artifact_sha256", "policy",
    }:
        return {"ok": False, "reason": "missing-or-invalid-state"}
    clock = int(time.time()) if now is None else now  # test-owned clock only
    if state["policy"] != POLICY or type(state["valid_until"]) is not int or state["valid_until"] <= clock:
        return {"ok": False, "reason": "expired-or-wrong-policy"}
    if state["catalog_sha256"] != digest(registry):
        return {"ok": False, "reason": "registry-mismatch"}
    if state["ref"] not in registry:
        return {"ok": False, "reason": "unknown-reference"}
    try:
        grant = bounded_json(broker_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"ok": False, "reason": "no-active-grant"}
    if not isinstance(grant, dict) or set(grant) != {
        "active", "session", "target", "ref", "catalog_sha256",
        "donor_artifact_sha256",
    } or grant["active"] is not True or any(
        grant[k] != v for k, v in {
            "session": session, "target": target, "ref": state["ref"],
            "catalog_sha256": state["catalog_sha256"],
            "donor_artifact_sha256": state["donor_artifact_sha256"],
        }.items()
    ):
        return {"ok": False, "reason": "no-active-grant"}
    a, b = registry[state["ref"]]
    return {"ok": True, "value": a * x + b, "via": transport,
            "node": node["node"]}


def b_fresh_process(*, transport: str, registry_path: Path,
                    broker_path: Path, receiver_artifact: Path, session: str,
                    target: str, x: int, db_path: Path | None = None,
                    state_path: Path | None = None) -> dict:
    cmd = [sys.executable, str(HERE / "causal_trial.py"), "receive",
           "--transport", transport, "--registry", str(registry_path),
           "--broker", str(broker_path), "--receiver", str(receiver_artifact),
           "--session", session, "--target", target]
    if db_path is not None:
        cmd.extend(("--db", str(db_path)))
    if state_path is not None:
        cmd.extend(("--state-file", str(state_path)))
    proc = subprocess.run(cmd, input=canonical_json({"x": x}), text=True,
                          capture_output=True, timeout=5, check=True)
    return json.loads(proc.stdout)


def host_score(scenario: Scenario, ref: str, xs: list[int], results: list[dict]) -> int:
    """Scoring oracle held outside both messages and StateBus state."""
    if len(xs) != len(results):
        return 0
    return sum(r.get("ok") is True and type(r.get("value")) is int and
               r.get("value") == scenario.evaluate(ref, x)
               for x, r in zip(xs, results))


def demo(seed: int = 317) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        a, b = root / "A.json", root / "B.json"
        pilot.model_artifact(a, node="node-A", style="annotated")
        pilot.model_artifact(b, node="node-B", style="terse")
        original = pilot.canonical_file(b)
        db, reg, broker, plain = (root / name for name in
                                  ("bus.sqlite3", "registry.json", "grant.json", "state.json"))
        scenario = Scenario.generate(seed)
        write_json(reg, scenario.registry_wire())
        obs = a_discover(a, scenario.donor_view())
        now = int(time.time())
        chosen = grant_via_bus(db, broker, scenario, a,
                               checked_source_head(pilot.trusted_source_head(None)), "causal", "shared-runtime",
                               observation=obs, now=now, valid_until=now + 300)
        write_json(plain, chosen)
        tasks = scenario.heldout(8)
        common = dict(registry_path=reg, broker_path=broker,
                      receiver_artifact=b, session="causal", target="shared-runtime")
        def batch(transport: str) -> list[dict]:
            return [b_fresh_process(transport=transport, db_path=db,
                                    state_path=plain, x=x, **common) for x in tasks]
        no_state = batch("none")
        statebus = batch("statebus")
        file_baseline = batch("file")
        revoke(broker)
        revoked = batch("statebus")
        return {"kind": "scripted-mock-causal-fixture-v1", "seed": seed,
                "candidates": len(scenario.registry), "heldout": len(tasks),
                "A_discovery_validated": host_validate_discovery(scenario, obs, a),
                "no_state_score": host_score(scenario, scenario.target_ref, tasks, no_state),
                "statebus_score": host_score(scenario, scenario.target_ref, tasks, statebus),
                "plain_file_score": host_score(scenario, scenario.target_ref, tasks, file_baseline),
                "revoked_score": host_score(scenario, scenario.target_ref, tasks, revoked),
                "unchanged_B_artifact": original == pilot.canonical_file(b),
                "task_blind_bus": set(chosen) == {
                    "ref", "valid_until", "catalog_sha256", "donor_artifact_sha256", "policy"},
                "source_git_commit_oid": "sha1:" + pilot.trusted_source_head(None),
                "causal_source_manifest_sha256": source_manifest_sha256()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("demo", "receive"))
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--transport", choices=("none", "file", "statebus"))
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--receiver", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--target", default="shared-runtime")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()
    if args.mode == "demo":
        result = demo(args.seed)
    else:
        raw = sys.stdin.read(4097)
        if len(raw) > 4096:
            result = {"ok": False, "reason": "invalid-task"}
        else:
            try:
                task = json.loads(raw)
            except json.JSONDecodeError:
                task = None
            if not isinstance(task, dict) or set(task) != {"x"}:
                result = {"ok": False, "reason": "invalid-task"}
            else:
                result = receive(transport=args.transport, registry_path=args.registry,
                                 broker_path=args.broker, receiver_artifact=args.receiver,
                                 session=args.session, target=args.target, x=task["x"],
                                 db_path=args.db, state_path=args.state_file)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
