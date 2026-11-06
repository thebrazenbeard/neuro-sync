"""Separate untaught-association and recipient-prior-retention SGD controls.

This is categorical-symbol source evidence, not a sealed language-model exam.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fixtures import SYNTHETIC_FACTS, TREK_FACTS
from learner import TinyLearner, delta_into_exact_base, sha256, train
from run import one_hot


def _pair(fact):
    return [fact.subject, fact.relation]


def receiver_only_eval(model, facts, workspace):
    """Invoke a new process with B checkpoint and query IDs, no donor/labels."""
    checkpoint = model.checkpoint()
    queries = [_pair(fact) for fact in facts]
    path = Path(workspace) / "recipient-only.json"
    path.write_text(json.dumps(checkpoint, sort_keys=True), encoding="utf8")
    process = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("untaught_predictor.py")), str(path)],
        input=json.dumps(queries), text=True, capture_output=True, check=True, timeout=20,
    )
    response = json.loads(process.stdout)
    expected = {"checkpoint_sha256", "query_sha256", "predictions"}
    if type(response) is not dict or set(response) != expected:
        raise ValueError("receiver-only output schema mismatch")
    if response["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("receiver-only checkpoint mismatch")
    if response["query_sha256"] != sha256(queries):
        raise ValueError("receiver-only ordered query mismatch")
    predictions = response["predictions"]
    if (type(predictions) is not list or len(predictions) != len(facts) or
            any(type(p) is not str or p not in model.answers for p in predictions)):
        raise ValueError("invalid receiver-only predictions")
    return {
        "correct": sum(p == f.answer for p, f in zip(predictions, facts)),
        "total": len(facts), "predictions": predictions,
        "checkpoint_sha256": response["checkpoint_sha256"],
        "query_sha256": sha256(queries), "new_process": True,
    }


def experiment_untaught(dataset="synthetic", *, model_seed=11, order_seed=77,
                        epochs=700, lr=0.15):
    """Run strict untouched-B arms and a separately flagged B-prior variant.

    The first four pair keys teach donor A. The final two are untaught by A,
    withheld from all strict-arm SGD and teacher queries. This is predeclared
    PUBLIC source data, not private or independently held evaluation material.
    """
    if dataset not in ("trek", "synthetic"):
        raise ValueError("dataset must be trek or synthetic")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    facts = TREK_FACTS if dataset == "trek" else SYNTHETIC_FACTS
    taught, untaught = facts[:4], facts[4:]
    if len(taught) != 4 or len(untaught) != 2:
        raise ValueError("fixed split requires exactly six facts")
    taught_pairs = [_pair(f) for f in taught]
    untaught_pairs = [_pair(f) for f in untaught]
    if set(map(tuple, taught_pairs)) & set(map(tuple, untaught_pairs)):
        raise ValueError("heldout pair leaked into training")

    # Local self-commitment created BEFORE training. The source fixtures expose
    # both answers; this does not constitute independent evaluator custody.
    split = {"taught_pairs": taught_pairs, "untaught_pairs": untaught_pairs,
             "taught_count": len(taught), "untaught_count": len(untaught),
             "commitment_sha256": sha256({"dataset": dataset,
                 "train": [(f.subject, f.relation, f.answer) for f in taught],
                 "untaught": [(f.subject, f.relation, f.answer) for f in untaught]})}

    base = TinyLearner([f.subject for f in facts], [f.relation for f in facts],
                       [f.answer for f in facts], seed=model_seed)
    base_sha = base.digest()
    donor = base.clone()
    untouched_b = base.clone()
    if donor.digest() != base_sha or untouched_b.digest() != base_sha:
        raise AssertionError("initial A/B checkpoints differ")

    donor_targets = [one_hot(donor.answers.index(f.answer), len(donor.answers))
                     for f in taught]
    train(donor, taught, donor_targets, epochs=epochs, lr=lr, seed=order_seed)
    donor_sha = donor.digest()
    if donor_sha == base_sha:
        raise AssertionError("donor did not train")
    teacher_probs = [donor.distribution(f.subject, f.relation) for f in taught]

    full_copy = donor.clone()
    exact_delta = delta_into_exact_base(base, donor, untouched_b.clone())
    distilled = untouched_b.clone()
    distill_updates = train(distilled, taught, teacher_probs,
                            epochs=epochs, lr=lr, seed=order_seed)["updates"]
    shuffled = untouched_b.clone()
    wrong_updates = train(shuffled, taught, teacher_probs[1:] + teacher_probs[:1],
                          epochs=epochs, lr=lr, seed=order_seed)["updates"]

    # Separate experiment: B learns two local associations FIRST, then C3
    # teaches the donor's four. The B-prior facts never train donor A.
    prior_b = untouched_b.clone()
    prior_targets = [one_hot(prior_b.answers.index(f.answer), len(prior_b.answers))
                     for f in untaught]
    prior_updates = train(prior_b, untaught, prior_targets,
                          epochs=epochs, lr=lr, seed=order_seed)["updates"]
    prior_sha = prior_b.digest()

    # All receiver-only tests occur after destroying donor A reference. Only
    # B artifacts, query symbols and evaluator-held answer labels remain.
    del donor
    arms = {
        "C0_untouched": untouched_b, "C1_full_donor_copy": full_copy,
        "C2_exact_base_delta": exact_delta, "C3_output_distillation": distilled,
        "C4_wrong_donor_shuffle": shuffled,
    }
    with tempfile.TemporaryDirectory(prefix="trek-untaught-v1-") as tmp:
        strict = {
            arm: {"taught": receiver_only_eval(model, taught, tmp),
                  "untaught": receiver_only_eval(model, untaught, tmp)}
            for arm, model in arms.items()
        }
        prior_before = receiver_only_eval(prior_b, untaught, tmp)
        transfer_updates = train(prior_b, taught, teacher_probs,
                                 epochs=epochs, lr=lr, seed=order_seed)["updates"]
        prior_after = receiver_only_eval(prior_b, untaught, tmp)
        learned_after = receiver_only_eval(prior_b, taught, tmp)

    return {
        "evidence_class": "SOURCE_TEST_CATEGORICAL_UNTAUGHT", "dataset": dataset,
        "model_seed": model_seed, "order_seed": order_seed,
        "epochs": epochs, "learning_rate": lr, "base_sha256": base_sha,
        "donor_after_sgd_sha256": donor_sha, "split": split,
        "strict_untouched_b": strict,
        "strict_untouched_b_recipient_training": {
            "C0_untouched": 0, "C1_full_donor_copy": 0,
            "C2_exact_base_delta": 0, "C3_output_distillation": distill_updates,
            "C4_wrong_donor_shuffle": wrong_updates,
        },
        "separate_prior_b_retention": {
            "prior_training_updates": prior_updates,
            "transfer_training_updates": transfer_updates,
            "prior_recipient_sha256": prior_sha,
            "recipient_after_transfer_sha256": prior_b.digest(),
            "before_transfer": prior_before,
            "after_transfer": prior_after,
            "new_taught_after_transfer": learned_after,
        },
        "limitations": [
            "This fixed split is public and self-committed, not private or independently sealed.",
            "Untaught means donor SGD and teacher queries excluded pair keys; answer classes and vocabulary were predeclared.",
            "All examples are categorical symbols, not unseen English facts or paraphrases.",
            "B-prior retention deliberately trains B and is separate from strict untouched-B arms.",
            "Recipient-only subprocess is not an authenticated host or cold machine restart.",
            "Different mechanisms are not compute-matched; no LLM or semantic generalization claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("trek", "synthetic"), default="synthetic")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(experiment_untaught(args.dataset, model_seed=args.seed,
                        epochs=args.epochs, lr=args.lr), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
