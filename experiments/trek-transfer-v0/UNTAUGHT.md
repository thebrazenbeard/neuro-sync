# Untaught-pair and prior-B-retention experiment (separate from strict v0)

**SOURCE/TEST — categorical symbols; not Qwen, English understanding, or an
independently sealed evaluator.**

This follow-up preserves the strict v0 experiment and adds a disjoint source
test for two claims that its 6/6 taught-pair result cannot support:

1. Do recipient parameters transfer correct associations on `(subject, relation)`
   keys never used for A's SGD nor B's teacher-query distributions?
2. In a **separate** B-prior variant, does B retain its own two preexisting
   associations after training on four teacher-output distributions?

```bash
python3 -m unittest discover -s experiments/trek-transfer-v0 -p 'test_*.py' -q
python3 experiments/trek-transfer-v0/untaught.py --dataset synthetic
python3 experiments/trek-transfer-v0/untaught.py --dataset trek
```

## Split and boundary

The first four fixed fixture pairs teach A and supply C3/C4 teacher queries.
The last two are never fed to A's SGD or to C3/C4 teacher queries in the
strict untouched-B comparison. Both groups are scored separately *after* A
is deleted, by child processes that receive B checkpoint plus ordered query
symbols, not answers or donor outputs. Child replies bind exact checkpoint
and ordered-query SHA-256, and parent checks shape/count/vocabulary before
scoring. Failures abort, never get silently truncated by `zip`.

A deterministic SHA-256 commitment is computed from the fixed full split
**before** training. It is **self-generated and public**: the six fixture
answers remain visible in source, and no independent evaluator holds a secret
oracle. This is a leakage-control source test, not a valid blinded test of a
language model. All six answer classes and vocabularies exist at initialization;
an unseen pair is not unseen language or a new answer token.

The prior-B variant deliberately uses the two omitted pairs to train B
locally, scores its two prior pairs, distills A's four taught distributions
into that B, and scores prior retention and new taught knowledge afterward.
It must **not** be pooled with the strictly untouched-B C0–C4 comparisons.

## Executed local source observations

At this patch's current base `89fdef1dd83536cf29875d3de7c2acb07e6f2cfa`,
with `seed=11`, `epochs=700`, `lr=0.15`, standard-library Python,
synthetic noncanonical fixtures:

| Arm | Taught keys (4) | Untaught keys (2) |
| --- | ---: | ---: |
| C0 untouched | 0/4 | 0/2 |
| C1 full donor copy | 4/4 | 0/2 |
| C2 exact-base delta | 4/4 | 0/2 |
| C3 teacher-output distillation | 4/4 | 0/2 |
| C4 wrong-key donor control | 0/4 | 0/2 |

Pretrain-B retention variant (not strict C0): prior associations **2/2**
before transfer, **2/2** after teacher-output distillation, and donor-taught
**4/4** afterward. On the official Trek fixture: strict taught C1–C3 **4/4**,
heldout C1–C3 **0/2**, and B-prior retention **2/2 → 2/2**. These are tiny,
seed-dependent observations, not a sample of statistical generalization.

## Hostile interpretation

> The positive 4/4 outcome remains **memorized taught associations**. The
> withheld 0/2 result directly limits any unsupported generalization claim.
> A model with predefined answer symbols could still guess an untaught fact;
> correct guessing would not by itself establish a transferable semantic rule.

> The independent B-prior variant uses existing two labels in B-local SGD.
> This is not Patrick's initially untouched B condition, and its 2/2 retention
> cannot be generalized to richer or conflicting knowledge without new tasks.

To qualify a real Vera behavior transfer, freeze an externally withheld,
independently scored dialogue evaluation and compare donor-disconnected cold
restarts against untouched, prompt-only, StateBus/file, shuffled-target, and
negative-transfer controls. Verify exact model+adapter+tokenizer artifacts,
authorization, rights, workload and no answer leakage before claims of
neural generalization or personality persistence.
