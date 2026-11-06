"""Dependency-free categorical neural learner with actual SGD gradients."""

import hashlib
import json
import math
import random


# This small Python reference learner is not a general checkpoint runtime.
# Bound accepted magnitudes so a finite-but-extreme tensor cannot overflow the
# simple Python dot products into inf/nan during softmax inference.
MAX_HIDDEN = 4096
MAX_ABS_PARAMETER = 1_000_000.0


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bounded_parameter(value):
    try:
        return (type(value) in (int, float) and math.isfinite(value) and
                abs(value) <= MAX_ABS_PARAMETER)
    except (OverflowError, ValueError):
        return False


class TinyLearner:
    """Trainable subject/relation embeddings, tanh hidden layer, softmax head.

    This is deliberately NOT a transformer or chat-capable language model.
    Answer symbols are supplied in the common vocabulary before either model
    trains; only the associations are unknown. There are no lookup assignments.
    """

    def __init__(self, subjects, relations, answers, hidden=16, seed=11, state=None):
        def canonical_vocabulary(values, label):
            values = list(values)
            if not values or any(type(value) is not str or not value for value in values):
                raise ValueError(f"invalid {label} vocabulary")
            return tuple(sorted(set(values)))

        # Constructor and checkpoint loader must accept the same domain: any
        # successfully constructed learner must be cloneable and reloadable.
        if type(hidden) is not int or not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError("invalid hidden width")
        self.subjects = canonical_vocabulary(subjects, "subjects")
        self.relations = canonical_vocabulary(relations, "relations")
        self.answers = canonical_vocabulary(answers, "answers")
        self.hidden = hidden
        if state is not None:
            self.weights = json.loads(json.dumps(state))
        else:
            rng = random.Random(seed)
            def mat(rows):
                return [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(rows)]
            self.weights = {
                "subject": mat(len(self.subjects)),
                "relation": mat(len(self.relations)),
                "output": mat(len(self.answers)),
                "bias": [0.0 for _ in self.answers],
            }
        self._validate_checkpoint(self.checkpoint())

    def checkpoint(self):
        return {
            "architecture": "categorical-embedding-tanh-softmax-v0",
            "subjects": list(self.subjects),
            "relations": list(self.relations),
            "answers": list(self.answers),
            "hidden": self.hidden,
            "weights": json.loads(json.dumps(self.weights)),
        }

    @staticmethod
    def _validate_checkpoint(check):
        if not isinstance(check, dict) or check.get("architecture") != "categorical-embedding-tanh-softmax-v0":
            raise ValueError("unknown checkpoint architecture")
        # External checkpoint bytes are data, not a trusted model shape.  The
        # former loader accepted truncated output matrices and could still
        # return a label by silently zipping fewer rows with a longer bias.
        expected_keys = {"architecture", "subjects", "relations", "answers", "hidden", "weights"}
        if set(check) != expected_keys:
            raise ValueError("invalid checkpoint metadata")
        for key in ("subjects", "relations", "answers"):
            entries = check[key]
            if (type(entries) is not list or not entries or
                    any(type(v) is not str or not v for v in entries) or
                    entries != sorted(set(entries))):
                raise ValueError(f"invalid {key} vocabulary")
        hidden = check["hidden"]
        if type(hidden) is not int or not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError("invalid hidden width")
        weights = check["weights"]
        if type(weights) is not dict or set(weights) != {"subject", "relation", "output", "bias"}:
            raise ValueError("invalid checkpoint weights")

        for key, vocabulary in (("subject", "subjects"), ("relation", "relations"),
                                ("output", "answers")):
            matrix = weights[key]
            if (type(matrix) is not list or len(matrix) != len(check[vocabulary]) or
                    any(type(row) is not list or len(row) != hidden or
                        not all(bounded_parameter(value) for value in row)
                        for row in matrix)):
                raise ValueError(f"invalid {key} weight shape or values")
        bias = weights["bias"]
        if (type(bias) is not list or len(bias) != len(check["answers"]) or
                not all(bounded_parameter(value) for value in bias)):
            raise ValueError("invalid output bias")

    @classmethod
    def from_checkpoint(cls, check):
        cls._validate_checkpoint(check)
        return cls(check["subjects"], check["relations"], check["answers"], check["hidden"], state=check["weights"])

    def clone(self):
        return self.from_checkpoint(self.checkpoint())

    def digest(self):
        return sha256(self.checkpoint())

    def _forward(self, subject, relation):
        si = self.subjects.index(subject)
        ri = self.relations.index(relation)
        s = self.weights["subject"][si]
        r = self.weights["relation"][ri]
        hidden = [math.tanh((s[j] + r[j]) / 2.0) for j in range(self.hidden)]
        logits = [sum(row[j] * hidden[j] for j in range(self.hidden)) + bias
                  for row, bias in zip(self.weights["output"], self.weights["bias"])]
        top = max(logits)
        exps = [math.exp(z - top) for z in logits]
        total = sum(exps)
        probs = [z / total for z in exps]
        return si, ri, hidden, probs

    def distribution(self, subject, relation):
        return self._forward(subject, relation)[3]

    def predict(self, subject, relation):
        probs = self.distribution(subject, relation)
        return self.answers[max(range(len(probs)), key=probs.__getitem__)]

    def sgd_step(self, subject, relation, target, lr):
        """Cross-entropy SGD; target is a distribution, not an answer overwrite."""
        if type(lr) not in (int, float) or not (0 < lr <= MAX_ABS_PARAMETER and math.isfinite(lr)):
            raise ValueError("learning rate must be positive and finite")
        if (not isinstance(target, (list, tuple)) or len(target) != len(self.answers) or
                any(type(t) not in (int, float) or not 0.0 <= t <= 1.0 or
                    not math.isfinite(t) for t in target) or
                not math.isclose(sum(target), 1.0, abs_tol=1e-8)):
            raise ValueError("invalid target distribution")
        si, ri, hidden, probs = self._forward(subject, relation)
        errors = [p - t for p, t in zip(probs, target)]
        old_output = [row[:] for row in self.weights["output"]]
        back = [(1.0 - hidden[j] ** 2) * sum(errors[k] * old_output[k][j]
                                                 for k in range(len(errors))) / 2.0
                for j in range(self.hidden)]
        # Calculate a complete candidate before touching the live checkpoint.
        # A valid pre-step checkpoint plus a permitted learning rate is not
        # enough to guarantee a valid post-step checkpoint near numeric bounds.
        # An overflow/bounds failure must leave the original model byte-equal.
        candidate = {
            "subject": [row[:] for row in self.weights["subject"]],
            "relation": [row[:] for row in self.weights["relation"]],
            "output": [row[:] for row in self.weights["output"]],
            "bias": self.weights["bias"][:],
        }
        for k, err in enumerate(errors):
            for j in range(self.hidden):
                candidate["output"][k][j] -= lr * err * hidden[j]
            candidate["bias"][k] -= lr * err
        for j, grad in enumerate(back):
            candidate["subject"][si][j] -= lr * grad
            candidate["relation"][ri][j] -= lr * grad
        # Shapes cannot change through these fixed-index arithmetic updates.
        # Check prospective *values* only here; full structural validation
        # already ran at construction and runs again on checkpoint reload.
        for key in ("subject", "relation", "output"):
            if any(not all(bounded_parameter(value) for value in row)
                   for row in candidate[key]):
                raise ValueError(f"invalid {key} weight shape or values")
        if not all(bounded_parameter(value) for value in candidate["bias"]):
            raise ValueError("invalid output bias")
        self.weights = candidate
        return -sum(t * math.log(max(p, 1e-300)) for t, p in zip(target, probs))


def train(model, facts, target_distributions, *, epochs=700, lr=0.15, seed=77):
    """Exactly epochs*len(facts) SGD updates; no hidden eval-time teaching."""
    if type(epochs) is not int or epochs < 0:
        raise ValueError("epochs must be a nonnegative integer")
    if type(lr) not in (int, float) or not (0 < lr <= MAX_ABS_PARAMETER and math.isfinite(lr)):
        raise ValueError("learning rate must be positive and finite")
    if len(facts) != len(target_distributions):
        raise ValueError("mismatched training examples")
    if not facts and epochs:
        raise ValueError("training facts cannot be empty")
    rng = random.Random(seed)
    order = list(range(len(facts)))
    last_loss = None
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            fact = facts[index]
            last_loss = model.sgd_step(fact.subject, fact.relation,
                                       target_distributions[index], lr)
    return {"updates": epochs * len(facts), "last_loss": last_loss}


def delta_into_exact_base(base, donor, receiver):
    """Only valid when receiver is the exact untouched base."""
    # Validate all three live states, not only their hashes.  A malformed
    # donor with fewer hidden columns previously raised an incidental
    # IndexError (or silently ignored extra columns) in the transfer loop.
    for model in (base, donor, receiver):
        TinyLearner.from_checkpoint(model.checkpoint())
    if receiver.digest() != base.digest():
        raise ValueError("receiver no longer matches base")
    if (base.hidden != donor.hidden or base.subjects != donor.subjects or
            base.relations != donor.relations or base.answers != donor.answers):
        raise ValueError("incompatible donor vocabulary")
    result = receiver.clone()
    for name in ("subject", "relation", "output"):
        for row in range(len(result.weights[name])):
            for j in range(result.hidden):
                # This is an exact-base delta: receiver == base.  Avoid an
                # intermediate rounded (donor-base) that can overshoot a
                # legitimate boundary donor by one ULP on re-addition.
                donor_value = donor.weights[name][row][j]
                combined = math.fsum((
                    result.weights[name][row][j],
                    donor_value,
                    -base.weights[name][row][j],
                ))
                # Preserve the original numeric representation (including
                # signed zero) when the compensated result is equal.
                result.weights[name][row][j] = donor_value if combined == donor_value else combined
    for k in range(len(result.answers)):
        donor_value = donor.weights["bias"][k]
        combined = math.fsum((result.weights["bias"][k], donor_value,
                              -base.weights["bias"][k]))
        result.weights["bias"][k] = donor_value if combined == donor_value else combined
    TinyLearner.from_checkpoint(result.checkpoint())
    return result
