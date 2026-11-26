"""Offline, reproducible multi-fact receiver protocol with explicit negative controls.

This protocol scores forced-choice answer *preference*, never model generalization.
It does not make heldout data private from an operator who can read the file.
"""
import math
import random
import string

VERSION = "selective-associations-v1"
UNKNOWN = " UNKNOWN"


def generate_suites(seed, known_count=8, calibration_count=2, unknown_count=4):
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    if min(known_count, calibration_count, unknown_count) < 1:
        raise ValueError("each cohort must be nonempty")
    rng = random.Random(seed)
    used_ids, used_codes = set(), set()
    def new_id():
        while True:
            value = "Arvex-" + str(rng.randint(100000, 999999))
            if value not in used_ids:
                used_ids.add(value)
                return value
    def new_code():
        while True:
            value = " " + "".join(rng.choices(string.ascii_uppercase + string.digits, k=7))
            if value not in used_codes and value != UNKNOWN:
                used_codes.add(value)
                return value
    known = [(new_id(), new_code()) for _ in range(known_count)]
    calibration = [new_id() for _ in range(calibration_count)]
    hidden = [new_id() for _ in range(unknown_count)]
    def train_prompt(node):
        return f"Fictional archive record. Node {node} has authorization token:"
    def exact_prompt(node):
        return f"In the fictional registry, what token belongs to node {node}?\nToken:"
    def paraphrase_prompt(node):
        return f"Give the assigned access code for {node} in the made-up radio archive.\nCode:"
    train = [{"kind": "known", "id": key, "prompt": train_prompt(key), "answer": code}
             for key, code in known]
    train += [{"kind": "calibration_unknown", "id": key,
               "prompt": train_prompt(key), "answer": UNKNOWN}
              for key in calibration]
    test = []
    for key, code in known:
        test.append({"kind": "known", "variant": "exact", "id": key,
                     "prompt": exact_prompt(key), "answer": code})
        test.append({"kind": "known", "variant": "paraphrase", "id": key,
                     "prompt": paraphrase_prompt(key), "answer": code})
    for key in hidden:
        test.append({"kind": "unknown", "variant": "heldout_unknown", "id": key,
                     "prompt": paraphrase_prompt(key), "answer": UNKNOWN})
    candidates = sorted(used_codes) + [UNKNOWN]
    return ({"protocol": VERSION, "seed": seed, "candidates": candidates, "examples": train},
            {"protocol": VERSION, "seed": seed, "candidates": candidates, "examples": test})
def shuffled_training(training):
    """Deterministic deranged labels with identical code frequencies."""
    if type(training) is not dict or training.get("protocol") != VERSION:
        raise ValueError("invalid training protocol")
    import copy
    changed = copy.deepcopy(training)
    known = [e for e in changed["examples"] if e.get("kind") == "known"]
    codes = [e["answer"] for e in known]
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("need unique known codes for a derangement")
    for item, code in zip(known, codes[1:] + codes[:1]):
        item["answer"] = code
    changed["arm"] = "shuffled_labels"
    return changed


def assess_scores(suite, rows):
    """Validate complete blinded-score tables; no fabricated accuracy from missing data."""
    if type(suite) is not dict or suite.get("protocol") != VERSION:
        raise ValueError("unknown protocol")
    candidates = suite.get("candidates")
    examples = suite.get("examples")
    if type(candidates) is not list or len(set(candidates)) != len(candidates) or UNKNOWN not in candidates:
        raise ValueError("bad answer vocabulary")
    if type(examples) is not list or type(rows) is not list or len(rows) != len(examples):
        raise ValueError("wrong number of evaluated rows")
    expected = {(e["id"], e["variant"]): e for e in examples}
    if len(expected) != len(examples):
        raise ValueError("duplicated test key")
    selected = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("non-object score row")
        key = (row.get("id"), row.get("variant"))
        scores = row.get("scores")
        if key not in expected or key in selected:
            raise ValueError("unexpected or duplicated score key")
        if type(scores) is not dict or set(scores) != set(candidates):
            raise ValueError("incomplete answer scores")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()):
            raise ValueError("nonfinite or nonnumeric score")
        top = max(scores.values())
        if sum(value == top for value in scores.values()) != 1:
            raise ValueError("tied predictions are not decisive")
        selected[key] = next(answer for answer, score in scores.items() if score == top)
    rates = {}
    for variant, field in (("exact", "exact_accuracy"),
                           ("paraphrase", "paraphrase_accuracy"),
                           ("heldout_unknown", "unknown_rejection")):
        items = [e for e in examples if e["variant"] == variant]
        if not items:
            raise ValueError("missing required evaluation cohort")
        rates[field] = sum(selected[(e["id"], e["variant"])] == e["answer"]
                           for e in items) / len(items)
    rates["qualified"] = all(rates[name] >= .75 for name in
                             ("exact_accuracy", "paraphrase_accuracy", "unknown_rejection"))
    rates["evidence"] = "OFFLINE_FORCED_CHOICE_SCREEN_ONLY"
    rates["generalization_proven"] = False
    rates["neurosync_novelty_proven"] = False
    return rates


def compare_models(suite, base_rows, adapted_rows):
    """Screen real receiver preferences against an equal-candidate base.

    A positive screen is not independent blinded replication or neural novelty.
    """
    base = assess_scores(suite, base_rows)
    adapted = assess_scores(suite, adapted_rows)
    return {
        "base": base,
        "adapted": adapted,
        "screen_pass": bool(
            adapted["qualified"]
            and adapted["paraphrase_accuracy"] - base["paraphrase_accuracy"] >= 0.25
            and adapted["unknown_rejection"] >= base["unknown_rejection"]
        ),
        "generalization_proven": False,
        "neurosync_novelty_proven": False,
        "evidence": "SINGLE_SEED_LOCAL_SCREEN_ONLY",
    }


def review_paired_receipts(records):
    """Untrusted diagnostic for the documented three-seed selective experiment.

    This is not independent custody, a frozen qualification decision, a model
    load, or proof of retained learning. Recompute scores; never trust a claimed
    screen_pass field. Other exploratory recipes remain valid outside this
    specific paired review.
    """
    import hashlib
    import json
    import re

    seeds = (109, 110, 111)
    model = "Qwen/Qwen3.5-0.8B-Base"
    revision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    sha = re.compile(r"[0-9a-f]{64}\Z")

    def need(ok, reason):
        if not ok:
            raise ValueError(reason)

    def digest(obj):
        return hashlib.sha256(json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()

    def predictions(rows):
        # Only called after compare_models has validated the complete tables.
        return {(row["id"], row["variant"]):
                max(row["scores"], key=row["scores"].get) for row in rows}

    need(type(records) is list and len(records) == len(seeds),
         "exact seed population required")
    by_seed = {}
    for record in records:
        need(type(record) is dict and set(record) == {"seed", "true", "shuffled"},
             "invalid seed record")
        seed = record["seed"]
        need(type(seed) is int and seed in seeds and seed not in by_seed,
             "invalid or duplicate seed")
        train_suite, suite = generate_suites(seed)
        arms = {}
        for arm in ("true", "shuffled"):
            item = record[arm]
            need(type(item) is dict and set(item) == {"train", "eval"},
                 "missing arm receipt")
            t, e = item["train"], item["eval"]
            need(type(t) is dict and type(e) is dict, "invalid receipt")
            need(type(t.get("seed")) is int and t["seed"] == seed and
                 type(e.get("seed")) is int and e["seed"] == seed and
                 t.get("arm") == e.get("arm") == arm, "seed/arm mismatch")
            need(t.get("protocol") == VERSION and t.get("source") == model and
                 t.get("revision") == e.get("source_revision") == revision,
                 "model/protocol mismatch")
            need(type(t.get("steps")) is int and t["steps"] == 40 and
                 type(t.get("learning_rate")) in (int, float) and
                 t["learning_rate"] == .001, "paired recipe mismatch")
            need(type(t.get("trainable_parameters")) is int and
                 t["trainable_parameters"] > 0, "invalid parameter count")
            losses = t.get("losses")
            need(type(losses) is list and len(losses) == 40 and
                 all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                     for x in losses), "invalid loss receipt")
            expected = shuffled_training(train_suite) if arm == "shuffled" else train_suite
            need(t.get("train_manifest_sha256") == digest(expected),
                 "training manifest mismatch")
            files = t.get("adapter_files_sha256")
            need(type(files) is dict and {"adapter_model.safetensors",
                 "adapter_config.json"} <= set(files) and
                 set(files) <= {"adapter_model.safetensors",
                 "adapter_config.json", "README.md"} and
                 all(type(v) is str and sha.fullmatch(v)
                     for v in files.values()), "invalid adapter map")
            need(e.get("adapter_sha256") == files["adapter_model.safetensors"],
                 "adapter digest mismatch")
            raw = e.get("raw_rows")
            need(type(raw) is dict and set(raw) == {"base", "adapted"},
                 "missing paired raw scores")
            scored = compare_models(suite, raw["base"], raw["adapted"])
            arms[arm] = (t, scored, predictions(raw["base"]),
                         predictions(raw["adapted"]))
        t, a, ba, pa = arms["true"]
        s, b, bb, pb = arms["shuffled"]
        need(t["trainable_parameters"] == s["trainable_parameters"],
             "paired parameter count mismatch")
        need(ba == bb, "paired baseline predictions differ")
        if (t["adapter_files_sha256"]["adapter_model.safetensors"] ==
                s["adapter_files_sha256"]["adapter_model.safetensors"]):
            need(pa == pb, "same adapter digest has divergent predictions")
        by_seed[seed] = {
            "true_screen": a["screen_pass"],
            "shuffled_screen": b["screen_pass"],
            "paraphrase_delta": a["adapted"]["paraphrase_accuracy"] -
                                b["adapted"]["paraphrase_accuracy"],
            "unknown_delta": a["adapted"]["unknown_rejection"] -
                             b["adapted"]["unknown_rejection"],
        }
    return {
        "paired_local_screen": all(x["true_screen"] and
            x["paraphrase_delta"] > 0 and x["unknown_delta"] >= 0
            for x in by_seed.values()),
        "seeds": by_seed,
        "evidence": "SELF_REPORTED_PAIRED_LOCAL_DIAGNOSTIC_ONLY",
        "generalization_proven": False,
        "independent_custody_proven": False,
        "neurosync_novelty_proven": False,
    }
