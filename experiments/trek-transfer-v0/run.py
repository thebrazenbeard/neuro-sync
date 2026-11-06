"""Strict untrained-clone Trek fact experiment with actual SGD and controls."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fixtures import SYNTHETIC_FACTS, TREK_FACTS
from learner import TinyLearner, delta_into_exact_base, sha256, train


def one_hot(index, size):
    return [1.0 if k == index else 0.0 for k in range(size)]


def evaluate_in_new_process(model, facts, workspace):
    """Start fresh Python interpreter, load B only; do not pass A or teacher."""
    path = Path(workspace) / "receiver.json"
    path.write_text(json.dumps(model.checkpoint(), sort_keys=True), encoding="utf-8")
    queries = [[f.subject, f.relation] for f in facts]
    response = subprocess.run([sys.executable, str(Path(__file__).with_name("predictor.py")), str(path)],
                              input=json.dumps(queries), text=True, capture_output=True,
                              timeout=20, check=True)
    result = json.loads(response.stdout)
    if result["checkpoint_sha256"] != sha256(model.checkpoint()):
        raise AssertionError("receiver-only process loaded an unexpected checkpoint")
    predictions = result["predictions"]
    return {"correct": sum(p == f.answer for p, f in zip(predictions, facts)),
            "total": len(facts), "predictions": predictions,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "new_process": True}


def experiment(dataset="trek", *, model_seed=11, order_seed=77, epochs=700, lr=0.15):
    if dataset not in ("trek", "synthetic"):
        raise ValueError("dataset must be trek or synthetic")
    facts = TREK_FACTS if dataset == "trek" else SYNTHETIC_FACTS
    subjects = [f.subject for f in facts]
    relations = [f.relation for f in facts]
    answers = [f.answer for f in facts]
    base = TinyLearner(subjects, relations, answers, seed=model_seed)
    donor = base.clone()
    untouched_b = base.clone()
    base_sha = base.digest()
    if donor.digest() != base_sha or untouched_b.digest() != base_sha:
        raise AssertionError("initial untrained A/B clones are not byte-identical")

    # Donor A alone receives exact fact labels through supervised CE/SGD.
    targets = [one_hot(donor.answers.index(f.answer), len(donor.answers)) for f in facts]
    donor_training = train(donor, facts, targets, epochs=epochs, lr=lr, seed=order_seed)
    if donor.digest() == base_sha:
        raise AssertionError("donor training made no parameter changes")
    trained_donor_sha = donor.digest()

    # C1 and C2 are deliberately straightforward copy/transplant comparators.
    full_copy_b = donor.clone()
    delta_b = delta_into_exact_base(base, donor, untouched_b.clone())

    # C3 transfers only donor output probability vectors on aligned queries.
    # No Star Trek answer symbols or target labels enter B's SGD training loop.
    teacher_probs = [donor.distribution(f.subject, f.relation) for f in facts]
    distilled_b = untouched_b.clone()
    distill_training = train(distilled_b, facts, teacher_probs, epochs=epochs, lr=lr, seed=order_seed)

    # C4 negative control: B learns distributions from the wrong donor question.
    shuffled_probs = teacher_probs[1:] + teacher_probs[:1]
    wrong_donor_b = untouched_b.clone()
    wrong_training = train(wrong_donor_b, facts, shuffled_probs, epochs=epochs, lr=lr, seed=order_seed)

    # The prediction subprocesses receive only B checkpoints and query IDs.
    # Destroy the donor reference before performing them. They are separate
    # interpreters with independent memory and no donor file or service.
    del donor
    arms = {
        "C0_untouched": untouched_b,
        "C1_full_donor_copy": full_copy_b,
        "C2_exact_base_delta": delta_b,
        "C3_output_distillation": distilled_b,
        "C4_wrong_donor_shuffle": wrong_donor_b,
    }
    with tempfile.TemporaryDirectory(prefix="trek-transfer-v0-") as tmp:
        evaluations = {}
        for name, receiver in arms.items():
            evaluations[name] = evaluate_in_new_process(receiver, facts, tmp)

    return {
        "qualification": "structured-categorical-SGD-mechanics-only; not an LLM or English generalization",
        "dataset": dataset,
        "facts_source_types": sorted({f.source_type for f in facts}),
        "n_facts": len(facts),
        "model_seed": model_seed,
        "order_seed": order_seed,
        "epochs": epochs,
        "learning_rate": lr,
        "base_sha256": base_sha,
        "initial_donor_equals_untouched_recipient": True,
        "donor_after_sgd_sha256": trained_donor_sha,
        "donor_training": donor_training,
        "recipient_training": {
            "C0_untouched": 0,
            "C1_full_donor_copy": 0,
            "C2_exact_base_delta": 0,
            "C3_output_distillation": distill_training["updates"],
            "C4_wrong_donor_shuffle": wrong_training["updates"],
        },
        "evals_after_donor_removed": evaluations,
        "limitations": [
            "All scored subject-relation pairs were taught to A; this tests associations, not unseen facts.",
            "Fixed symbols and answer vocabulary are known to both A and B before training.",
            "B untouched is literal; B-retention-of-independent-knowledge is a separate future variant.",
            "C3 is ordinary teacher-output distillation, not novel bidirectional gradient sharing.",
            "C1/C2 copy donor parameter information; they are not no-copy interaction.",
            "Independent receiver-only subprocess proves process isolation, not a durable remote host restart.",
            "No training-compute matching between copying and SGD; updates are reported per arm.",
            "Six public QA associations give no statistical or semantic-generalization claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("trek", "synthetic"), default="trek")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(experiment(args.dataset, model_seed=args.seed,
                                epochs=args.epochs, lr=args.lr), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
