"""Fair *scripted mock* N/V/P/H controls for external capability references.

All arms invoke the same receiver subprocess with the same candidate handles,
tasks and host-owned tool access. The receiver receives neither tool code,
answer-bearing registry, broker file path, oracle, nor donor's private probe.
N=none, V=StateBus, P=ordinary JSON file, H=host/operator ref without donor.
The fixture tests information routing, not neural model learning or authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import random
import secrets
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fair_receiver_v2 import RECEIVER_POLICY, receiver_decide  # noqa: E402
from neurosync.bus import (  # noqa: E402
    EvidenceBinding, EvidenceClass, StateDelta, StateLedger, StateOp,
    ValidationReceipt, canonical_json,
)

KEY = "capabilities.fair_v2"
PROBES = (-7, 11, 29)
SCOPED_SOURCES = (
    "experiments/zero-weight-sync/fair_trial_v2.py",
    "experiments/zero-weight-sync/fair_receiver_v2.py",
    "experiments/zero-weight-sync/test_fair_trial_v2.py",
    "src/neurosync/bus/schema.py", "src/neurosync/bus/ledger.py",
)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def source_provenance() -> tuple[str, str]:
    """Require exact committed evaluator, test-policy and ledger source bytes."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip()
    if subprocess.check_output(["git", "cat-file", "-t", head], cwd=ROOT,
                               text=True).strip() != "commit":
        raise ValueError("source HEAD is not a commit")
    entries = {}
    for name in SCOPED_SOURCES:
        checked = subprocess.run(["git", "show", f"{head}:{name}"],
                                 cwd=ROOT, check=True, capture_output=True).stdout
        observed = (ROOT / name).read_bytes()
        if checked != observed:
            raise ValueError("source/test policy differs from committed HEAD: " + name)
        entries[name] = hashlib.sha256(observed).hexdigest()
    return "sha1:" + head, digest(entries)


def evaluate(family: str, parameters: tuple[int, ...], x: int) -> int:
    if family == "affine":
        a, b = parameters
        return a * x + b
    if family == "quadratic":
        a, b, c = parameters
        return a * x * x + b * x + c
    raise ValueError("unknown synthetic family")


def oracle_reference(family: str, parameters: tuple[int, ...], x: int) -> int:
    """Separately implemented host expectation via Horner evaluation.

    The executing handler above uses explicit formulas. Agreement of two
    co-designed local functions is not an independent trust principal; this
    split catches single-handler corruption and scoring tautologies.
    """
    if family not in {"affine", "quadratic"} or len(parameters) != (
            2 if family == "affine" else 3):
        raise ValueError("unknown oracle family")
    value = 0
    for coefficient in parameters:
        value = value * x + coefficient
    return value


@dataclass(frozen=True)
class DonorView:
    """A can probe tools but cannot inspect host's sealed target label."""

    family: str
    candidates: tuple[tuple[str, tuple[int, ...]], ...]
    signature: tuple[int, ...]

    def probe(self, ref: str, x: int) -> int:
        for item_ref, parameters in self.candidates:
            if ref == item_ref:
                return evaluate(self.family, parameters, x)
        raise KeyError(ref)


@dataclass(frozen=True)
class HostScenario:
    """Private host registry/oracle: never serialized into receiver input."""

    seed: int
    family: str
    tools: dict[str, tuple[int, ...]]
    target_ref: str

    @classmethod
    def generate(cls, seed: int, family: str, count: int = 13,
                 *, host_secret: bytes | None = None) -> "HostScenario":
        if type(seed) is not int or seed < 0 or seed >= 2**32 or family not in {"affine", "quadratic"}:
            raise ValueError("invalid seed/family")
        if not 3 <= count <= 32:
            raise ValueError("invalid candidate count")
        if host_secret is None:
            host_secret = secrets.token_bytes(32)
        if type(host_secret) is not bytes or len(host_secret) != 32:
            raise ValueError("host must hold a 32-byte run-private synthetic secret")
        rng = random.Random((seed << 1) ^ (0x178B if family == "quadratic" else 0x3291))
        tools = {}
        while len(tools) < count:
            params = tuple(rng.randrange(2, 40) for _ in range(2 if family == "affine" else 3))
            if params in tools.values():
                continue
            tools[f"cap-{rng.getrandbits(64):016x}"] = params
        refs = tuple(tools)
        # Candidate handles may be public and reconstructible from a seed.
        # The *target* is independently HMAC-selected using unpredictable
        # host-only bytes that never cross into the receiver subprocess.
        choice = hmac.new(host_secret, f"{seed}|{family}".encode(),
                          hashlib.sha256).digest()
        target = refs[int.from_bytes(choice, "big") % len(refs)]
        return cls(seed, family, tools, target)

    def oracle(self, x: int) -> int:
        return oracle_reference(self.family, self.tools[self.target_ref], x)

    def donor_view(self) -> DonorView:
        return DonorView(self.family, tuple(self.tools.items()),
                         tuple(self.oracle(x) for x in PROBES))

    def heldout(self, count: int = 8) -> tuple[int, ...]:
        rng = random.Random(self.seed ^ (0x720DAB if self.family == "affine" else 0xAF917A))
        values = set()
        while len(values) < count:
            n = rng.randrange(-500, 501)
            if n not in PROBES:
                values.add(n)
        return tuple(sorted(values))


def donor_discover(view: DonorView) -> dict:
    """Scripted A searches its restricted probe interface; no target_ref."""
    if not isinstance(view, DonorView) or hasattr(view, "target_ref"):
        raise TypeError("donor must receive restricted view")
    refs = [ref for ref, _ in view.candidates if
            tuple(view.probe(ref, x) for x in PROBES) == view.signature]
    if len(refs) != 1:
        raise ValueError("ambiguous donor capability observation")
    return {"ref": refs[0], "signature_sha256": digest(list(view.signature)),
            "family": view.family, "candidate_count": len(view.candidates)}


def host_validate(scenario: HostScenario, observation: object) -> bool:
    if not isinstance(observation, dict) or set(observation) != {
        "ref", "signature_sha256", "family", "candidate_count"}:
        return False
    return (type(observation["ref"]) is str and
            observation["ref"] == scenario.target_ref and
            observation["signature_sha256"] == digest(list(scenario.donor_view().signature)) and
            observation["family"] == scenario.family and
            type(observation["candidate_count"]) is int and
            observation["candidate_count"] == len(scenario.tools))


def commit_bus(db: Path, scenario: HostScenario, observation: dict,
               *, session: str, target: str) -> dict:
    """Host local fixture: StateBus commit, not authenticated witness custody."""
    if not host_validate(scenario, observation):
        raise PermissionError("A observation rejected by host oracle")
    oid, source_sha = source_provenance()
    now = int(time.time())
    value = {"ref": observation["ref"]}  # same task-blind value for V/P/H
    evidence = EvidenceBinding(
        artifact_sha256=digest(RECEIVER_POLICY),
        source_git_commit_oid=oid, source_tree_sha256=source_sha,
        host_test_receipt_sha256=digest({"observation": observation,
                                        "source_sha256": source_sha,
                                        "host_oracle_passed": True}),
        evidence_class=EvidenceClass.TEST,
    )
    delta = StateDelta.create(
        event_id=f"evt-fair-{session}", source_id="scripted-A-v2",
        target_id=target, session_id=session, sequence=1, expected_revision=0,
        operation=StateOp.SET, payload={"key": KEY, "value": value},
        issued_at_epoch=now - 1, expires_at_epoch=now + 300,
        evidence=evidence, provenance_refs=("synthetic-host-probe",),
        capability_refs=("fair-tool-reference",),
    )
    ledger = StateLedger(db)
    authorize = lambda d: d == delta and host_validate(scenario, observation)
    ledger.propose(delta, authorize_proposer=authorize)
    # `independent-verifier` is a required receipt role *string*, not proof
    # of an independently authenticated validator. This host fixture controls
    # the proposal, oracle and receipt in one trust domain.
    receipt = ValidationReceipt(
        version=1, validation_id=f"validated-{delta.event_id}",
        proposal_hash=delta.content_hash, evidence_sha256=evidence.sha256(),
        verifier_id="fixture-host-oracle", verifier_role="independent-verifier",
        decision="approve", evaluated_at_epoch=now,
    )
    verify = lambda d, r: d == delta and r == receipt and host_validate(scenario, observation)
    ledger.validate(delta.event_id, receipt, verify_validator=verify)
    ledger.commit(delta.event_id, verify_validator=verify,
                  authorize_commit=lambda d, r: authorize(d) and r == receipt)
    return StateLedger(db).project(target, session)[KEY]


def fresh_receiver(request: dict) -> tuple[dict, str]:
    """B subprocess gets only the JSON request, no host/registry/broker paths."""
    raw = canonical_json(request)
    proc = subprocess.run([sys.executable,
                           str(Path(__file__).with_name("fair_receiver_v2.py"))],
                          input=raw, text=True, capture_output=True,
                          check=True, timeout=5)
    result = json.loads(proc.stdout)
    return result, hashlib.sha256(proc.stdout.encode()).hexdigest()


def read_plain_file(path: Path) -> dict:
    """P baseline has no StateBus transport, DB object or bus callback."""
    data = path.read_bytes()
    if len(data) > 1024:
        raise ValueError("plain capability record too large")
    state = json.loads(data)
    if not isinstance(state, dict) or set(state) != {"ref"} or type(state["ref"]) is not str:
        raise ValueError("invalid plain capability record")
    return state


def host_effects(scenario: HostScenario, tasks: tuple[int, ...],
                 receiver_result: dict,
                 *, frozen_expected: tuple[int, ...] | None = None) -> dict:
    """Only host owns tool implementations and independently sealed oracle."""
    choices = receiver_result["choices"]
    if len(choices) != len(tasks):
        raise ValueError("receiver choice count mismatch")
    if frozen_expected is None:
        frozen_expected = tuple(scenario.oracle(x) for x in tasks)
    if len(frozen_expected) != len(tasks) or not all(type(x) is int for x in frozen_expected):
        raise ValueError("invalid independently prepared host expectation")
    records = []
    for x, choice, expected in zip(tasks, choices, frozen_expected):
        if (type(choice) is not dict or set(choice) != {"x", "ref", "decision"} or
                type(choice["x"]) is not int or choice["x"] != x):
            raise ValueError("forged/misaligned task selection")
        ref = choice["ref"]
        if (type(ref) is not str or ref not in scenario.tools or
                type(choice["decision"]) is not str or
                choice["decision"] not in {"uninformed-guess", "supplied-reference"}):
            raise ValueError("invalid receiver tool choice")
        observed = evaluate(scenario.family, scenario.tools[ref], x)
        records.append({"x": x, "selected_ref": ref, "decision": choice["decision"],
                        "effect": observed, "expected": expected,
                        "correct_effect": type(observed) is int and observed == expected,
                        "correct_reference": ref == scenario.target_ref})
    return {"selected_ref_score": sum(x["correct_reference"] for x in records),
            "effect_score": sum(x["correct_effect"] for x in records),
            "raw_effects": records}


def one_scenario(seed: int, family: str,
                 *, host_secret: bytes | None = None) -> dict:
    scenario = HostScenario.generate(seed, family, host_secret=host_secret)
    tasks = scenario.heldout()
    frozen_expected = tuple(scenario.oracle(x) for x in tasks)
    handles = list(scenario.tools)
    session, target = f"fair-{family}-{seed}", "mock-B"
    outputs = {}
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        # N and H execute before A is ever called. Host/operator H knows the
        # correct ref independently; the donor is *absent* in this arm.
        context = {"N": None, "H": {"ref": scenario.target_ref}}
        for arm in ("N", "H"):
            req = {"handles": handles, "tasks": list(tasks), "state": context[arm]}
            result, output_sha = fresh_receiver(req)
            outputs[arm] = {"receiver_input_sha256": digest(req),
                            "receiver_output_sha256": output_sha,
                            **host_effects(scenario, tasks, result,
                                           frozen_expected=frozen_expected)}

        observation = donor_discover(scenario.donor_view())
        if not host_validate(scenario, observation):
            raise AssertionError("scripted A observation failed sealed host oracle")
        mapped = {"ref": observation["ref"]}
        plain = base / "plain.json"
        plain.write_text(canonical_json(mapped), encoding="utf8")
        # P uses ordinary file I/O exclusively; it never opens StateLedger.
        p_state = read_plain_file(plain)
        req_p = {"handles": handles, "tasks": list(tasks), "state": p_state}
        result, output_sha = fresh_receiver(req_p)
        outputs["P"] = {"receiver_input_sha256": digest(req_p),
                        "receiver_output_sha256": output_sha,
                        **host_effects(scenario, tasks, result,
                                       frozen_expected=frozen_expected)}

        v_state = commit_bus(base / "state.sqlite3", scenario, observation,
                             session=session, target=target)
        req_v = {"handles": handles, "tasks": list(tasks), "state": v_state}
        result, output_sha = fresh_receiver(req_v)
        outputs["V"] = {"receiver_input_sha256": digest(req_v),
                        "receiver_output_sha256": output_sha,
                        **host_effects(scenario, tasks, result,
                                       frozen_expected=frozen_expected)}

    assert context["H"] == p_state == v_state  # same mapped ref/receiver inputs
    return {"seed": seed, "family": family, "candidate_count": len(handles),
            "tasks": list(tasks), "A_observed_ref": observation["ref"],
            "host_sealed_target": scenario.target_ref,
            "P_V_H_input_equal": (outputs["P"]["receiver_input_sha256"] ==
                                    outputs["V"]["receiver_input_sha256"] ==
                                    outputs["H"]["receiver_input_sha256"]),
            "arms": outputs}


def experiment(seeds: tuple[int, ...] = (3, 7, 19, 31, 47, 61, 73, 89),
               *, host_secret: bytes | None = None) -> dict:
    if len(set(seeds)) != len(seeds) or not 6 <= len(seeds) <= 12:
        raise ValueError("requires 6–12 independent synthetic scenario seeds")
    if host_secret is None:
        host_secret = secrets.token_bytes(32)
    if type(host_secret) is not bytes or len(host_secret) != 32:
        raise ValueError("invalid synthetic host secret")
    oid, manifest = source_provenance()
    trials = [one_scenario(seed, family, host_secret=host_secret)
              for seed in seeds for family in ("affine", "quadratic")]
    aggregate = {arm: sum(item["arms"][arm]["effect_score"] for item in trials)
                 for arm in ("N", "V", "P", "H")}
    return {"fixture": "fair-independent-control-v2-scripted-mock",
            "source_git_commit_oid": oid, "scoped_manifest_sha256": manifest,
            "host_secret_sha256_commitment": hashlib.sha256(host_secret).hexdigest(),
            # Synthetic commit/reveal happens ONLY after every B process exits.
            # This permits replay. Never reuse these bytes for future tests.
            "host_secret_revealed_after_trial_hex": host_secret.hex(),
            "receiver_policy_sha256": digest(RECEIVER_POLICY),
            "scenario_count": len(trials), "tasks_per_arm": len(trials) * 8,
            "aggregate_effect_scores": aggregate, "scenarios": trials,
            "nonclaims": ["not neural model learning", "not authenticated external custody",
                          "not unique StateBus functional advantage"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("demo",))
    args = parser.parse_args()
    output = experiment()
    print(canonical_json(output))


if __name__ == "__main__":
    main()
