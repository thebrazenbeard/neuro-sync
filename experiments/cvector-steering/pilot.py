"""Source-only, no-training control-vector experiment planner.

This module deliberately NEVER loads a model, invokes an inference executable,
or writes a model/control-vector file. It generates bounded argv manifests for
future separately authorized experiments and checks source-level controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "NEUROSYNC_CVECTOR_SOURCE_PLAN_V1"
EVIDENCE = "SOURCE_PLAN_ONLY_NO_MODEL_CALLS"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ARM_NAMES = ("N", "P", "V_PLUS", "V_MINUS", "V_ZERO", "V_SHAM", "PV_PLUS")


class PlanHold(ValueError):
    """Malformed or attribution-confounded proposed experiment."""


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise PlanHold(f"invalid_{name}_sha256")
    return value


def _public_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 768:
        raise PlanHold(f"invalid_{name}")
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise PlanHold(f"invalid_{name}_control_character")
    return value


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


@dataclass(frozen=True)
class Artifact:
    label: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, obj: object, label: str) -> "Artifact":
        if not isinstance(obj, dict) or set(obj) != {"path", "sha256"}:
            raise PlanHold(f"invalid_{label}_shape")
        path = obj["path"]
        if not isinstance(path, str) or not 1 <= len(path) <= 1024 or "\x00" in path:
            raise PlanHold(f"invalid_{label}_path")
        return cls(label, path, _sha(obj["sha256"], label))


def build_plan(spec: dict) -> dict:
    """Create a deterministic future-run plan; source declarations are UNVERIFIED.

    The caller must independently bind these declared hashes to local files and
    model process identity before executing any command. Commands are data.
    """
    if not isinstance(spec, dict) or set(spec) != {
        "model", "cli", "generator", "vector", "sham_vector", "model_id",
        "runtime_build", "positive", "negative", "heldouts", "persona",
        "seed", "max_tokens", "vector_scale",
    }:
        raise PlanHold("invalid_spec_shape")
    artifacts = {key: Artifact.from_dict(spec[key], key) for key in
                 ("model", "cli", "generator", "vector", "sham_vector")}
    if len({a.path for a in artifacts.values()}) != 5:
        raise PlanHold("artifact_path_alias")
    if artifacts["vector"].sha256 == artifacts["sham_vector"].sha256:
        raise PlanHold("sham_vector_same_bytes")
    model_id = spec["model_id"]
    if model_id != "Qwen/Qwen3.5-0.8B":
        raise PlanHold("unqualified_model_identity")
    _sha(spec["runtime_build"], "runtime_build")
    seed = spec["seed"]
    max_tokens = spec["max_tokens"]
    scale = spec["vector_scale"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise PlanHold("invalid_seed")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 16 <= max_tokens <= 256:
        raise PlanHold("invalid_max_tokens")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0.05 <= scale <= 2.0:
        raise PlanHold("invalid_vector_scale")
    if not isinstance(spec["positive"], list) or not isinstance(spec["negative"], list):
        raise PlanHold("invalid_pair_lists")
    if not 4 <= len(spec["positive"]) == len(spec["negative"]) <= 32:
        raise PlanHold("invalid_pair_count")
    # All seven arms occupy every run-order position equally often.
    if (not isinstance(spec["heldouts"], list)
            or not 7 <= len(spec["heldouts"]) <= 63
            or len(spec["heldouts"]) % 7 != 0):
        raise PlanHold("invalid_heldout_count")
    positives = [_public_text(x, "positive") for x in spec["positive"]]
    negatives = [_public_text(x, "negative") for x in spec["negative"]]
    heldouts = [_public_text(x, "heldout") for x in spec["heldouts"]]
    persona = _public_text(spec["persona"], "persona")
    all_text = positives + negatives + heldouts
    if len({_norm(x) for x in all_text}) != len(all_text):
        raise PlanHold("duplicated_or_contaminated_text")

    spec_sha = hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    arms = []
    for scenario_index, holdout in enumerate(heldouts):
        # Seven-position cyclic counterbalance: avoid always showing N first.
        order = ARM_NAMES[scenario_index % len(ARM_NAMES):] + ARM_NAMES[:scenario_index % len(ARM_NAMES)]
        for arm in order:
            vector = None
            vector_scale = None
            if arm in ("V_PLUS", "PV_PLUS"):
                vector, vector_scale = artifacts["vector"].path, float(scale)
            elif arm == "V_MINUS":
                vector, vector_scale = artifacts["vector"].path, -float(scale)
            elif arm == "V_ZERO":
                vector, vector_scale = artifacts["vector"].path, 0.0
            elif arm == "V_SHAM":
                vector, vector_scale = artifacts["sham_vector"].path, float(scale)
            prompt = holdout
            if arm in ("P", "PV_PLUS"):
                prompt = f"{persona}\n\n{holdout}"
            argv = [artifacts["cli"].path, "-m", artifacts["model"].path,
                    "-p", prompt, "-n", str(max_tokens), "--seed", str(seed),
                    "--temp", "0", "--no-display-prompt", "--single-turn"]
            if vector is not None:
                argv.extend(["--control-vector-scaled", f"{vector}:{vector_scale:g}"])
            arms.append({"scenario": scenario_index, "arm": arm,
                         "argv": argv, "will_execute": False})

    return {
        "schema": SCHEMA, "evidence": EVIDENCE, "spec_sha256": spec_sha,
        "declared_artifacts_unverified": {k: {"path": a.path, "sha256": a.sha256}
                                           for k, a in artifacts.items()},
        "runtime_build_sha256_declared_unverified": spec["runtime_build"],
        "generation_pairs_count": len(positives), "holdouts_count": len(heldouts),
        "arms_count": len(arms), "commands_are_data_only": True,
        "model_calls": 0, "weight_updates": 0, "inference_effect": "NOT_TESTED",
        "semantic_effect": "NOT_TESTED", "identity_transfer": "NOT_ESTABLISHED",
        "plan": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        print(json.dumps(build_plan(spec), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")))
    except (PlanHold, ValueError, OSError) as exc:
        print(json.dumps({"schema": SCHEMA, "status": "HOLD", "reason": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
