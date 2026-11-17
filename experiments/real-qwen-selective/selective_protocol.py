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
