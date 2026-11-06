"""Dependency-free categorical neural learner with actual SGD gradients."""

import hashlib
import json
import math
import random


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class TinyLearner:
    """Trainable subject/relation embeddings, tanh hidden layer, softmax head.

    This is deliberately NOT a transformer or chat-capable language model.
    Answer symbols are supplied in the common vocabulary before either model
    trains; only the associations are unknown. There are no lookup assignments.
    """

    def __init__(self, subjects, relations, answers, hidden=16, seed=11, state=None):
        self.subjects = tuple(sorted(set(subjects)))
        self.relations = tuple(sorted(set(relations)))
        self.answers = tuple(sorted(set(answers)))
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

    def checkpoint(self):
        return {
            "architecture": "categorical-embedding-tanh-softmax-v0",
            "subjects": list(self.subjects),
            "relations": list(self.relations),
            "answers": list(self.answers),
            "hidden": self.hidden,
            "weights": json.loads(json.dumps(self.weights)),
        }

    @classmethod
    def from_checkpoint(cls, check):
        if check.get("architecture") != "categorical-embedding-tanh-softmax-v0":
            raise ValueError("unknown checkpoint architecture")
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
        si, ri, hidden, probs = self._forward(subject, relation)
        if len(target) != len(self.answers) or not math.isclose(sum(target), 1.0, abs_tol=1e-8):
            raise ValueError("invalid target distribution")
        errors = [p - t for p, t in zip(probs, target)]
        old_output = [row[:] for row in self.weights["output"]]
        back = [(1.0 - hidden[j] ** 2) * sum(errors[k] * old_output[k][j]
                                                 for k in range(len(errors))) / 2.0
                for j in range(self.hidden)]
        for k, err in enumerate(errors):
            for j in range(self.hidden):
                self.weights["output"][k][j] -= lr * err * hidden[j]
            self.weights["bias"][k] -= lr * err
        for j, grad in enumerate(back):
            self.weights["subject"][si][j] -= lr * grad
            self.weights["relation"][ri][j] -= lr * grad
        return -sum(t * math.log(max(p, 1e-300)) for t, p in zip(target, probs))


def train(model, facts, target_distributions, *, epochs=700, lr=0.15, seed=77):
    """Exactly epochs*len(facts) SGD updates; no hidden eval-time teaching."""
    if len(facts) != len(target_distributions):
        raise ValueError("mismatched training examples")
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
    if receiver.digest() != base.digest():
        raise ValueError("receiver no longer matches base")
    if base.subjects != donor.subjects or base.relations != donor.relations or base.answers != donor.answers:
        raise ValueError("incompatible donor vocabulary")
    result = receiver.clone()
    for name in ("subject", "relation", "output"):
        for row in range(len(result.weights[name])):
            for j in range(result.hidden):
                result.weights[name][row][j] += donor.weights[name][row][j] - base.weights[name][row][j]
    for k in range(len(result.answers)):
        result.weights["bias"][k] += donor.weights["bias"][k] - base.weights["bias"][k]
    return result
