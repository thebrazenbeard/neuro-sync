# Transfer falsifier: when recipient B has learned something already

**Status: local source/test experiment using the existing tiny categorical SGD
learner. No language model, cross-architecture transfer, external verification,
or generalization result.** Python standard library only; $0.

The strict Trek v0 experiment initializes A and B as exact clones, trains only
A, and then measures untouched B. Under that premise, an exact-base donor delta
is **mathematically donor copying**, so its success cannot establish a novel
transfer method. This study removes the simplifying premise: **B learns two
disjoint synthetic facts first**. Donor A independently learns four other
facts. One subject (`KEPLER_HARBOR`) occurs in both partitions with a different
relation, so the retention task is not trivially separated by subject identity.
Donor supervision never uses B-only correct labels, but donor probability
vectors span the entire shared answer vocabulary, including those tokens.
All scored questions were taught in their respective A/B partition. This is
**not held-out generalization**; it tests preservation of taught B associations.

## Exploratory, descriptive three-seed observation

Seed set `(11,23,47)`, 400 SGD epochs per training stage, learning rate 0.15.
This shared-subject partition was selected after inspecting an easier
disjoint-subject pilot: **posthoc exploratory, not preregistered**. A zero-SGD
untouched base, a wrongly paired teacher-distribution sham, and a direct-label
supervised B comparator are included. C3 and C5 have equal B-side SGD step
counts (1,600 each) and optimizer settings; C5 receives **true answer labels**
for donor facts, whereas C3 receives only donor numeric distributions. The
recipient-only evaluator runs in a fresh Python subprocess for every arm
and both partitions. Denominators are **four donor-taught queries** and **two
B-retention queries** per seed, not an independent large sample.

| Receiver arm | Donor-taught correct /4 | B retention correct /2 | Meaning |
| --- | ---: | ---: | --- |
| C0 untouched base | 0 | 0 | Zero-SGD baseline at these seeds |
| B retention-only | 0 | 2 | B learned only its original two |
| C1 full donor copy | 4 | 0 | Replacement gained A facts but erased B facts |
| C3 ordinary teacher-output distillation into learned B | 4 | 1 | New facts acquired but one old fact forgotten |
| C4 wrongly paired donor outputs into learned B | 0 | 1 | Sham pairing, not a universal baseline |
| C5 direct-label supervision into learned B | 4 | 1 | Equal B-side updates; **matched C3's scores** |

All three chosen seeds produced these exact counts in the local run. That is
a **reproducible tiny synthetic observation**, not a confidence interval or
evidence of a persistent advantage across tasks. The script reports individual
predictions, fresh-process checkpoint digests, training update counts, source
fixture digest, and descriptive means; it does not invent a significance test.
**No teacher-distillation advantage over direct-label supervised B was
observed**: C3 and C5 tied on both scored partitions for every selected seed.
Matching B update counts is not matching full pipeline computation: A's own
supervised training and teacher-output generation add work to C3's route.

The existing `delta_into_exact_base` **correctly rejects** the learned
recipient in all three trials. That rejection is an important falsifier of the
idea that subtracting A's initial weights from its trained weights is enough to
merge new learning into *an already changed B* while preserving B's knowledge.
A compatible nonlinear weight merger remains unimplemented and unqualified.

## Reproduce

From repository root:

```sh
python3 experiments/transfer-falsifiers/retention_study.py --epochs 400
python3 -m unittest discover -s experiments/transfer-falsifiers -p 'test_*.py' -v
```

No network, account, GPU, model download, or Qwen checkpoint is used. No
service is started or altered. The tiny model's closed answer symbols are
shared up front, and the B-only queries were **trained on B**, so retention
must never be misreported as zero-shot generalization. The six arms do not
match total training FLOPs or wall clock. This study does not settle whether more
advanced weight alignment, residual adapters, sparse task vectors, or model
merging can preserve both knowledge sets in larger networks.

## Accelerated hostile review questions

> **Purist:** Does exact-base delta preserve B's old knowledge when B has
> independent updates? No: the API refuses the input; any claim that it merges
> such states is falsified by its precondition.

> **Pragmatism:** Does ordinary distillation beat an equally trained B given
> direct labels? Not in these small runs. Both C3 and C5 achieve 4/4 donor
> facts and retain only 1/2 B-held facts.

> **Bridge:** Is this neural model-to-model synchronization? No. C3 is familiar
> output-distribution training over aligned symbolic examples, with identical
> learner architectures and vocabulary. Nothing here establishes transfer
> between heterogeneous LLMs or a cold-restarted Qwen.

**Quorum determination: REFINE.** Preserve the goal of non-destructive learning
transfer; reject exact-base donor copy as a nontrivial solution for a B that
has changed since cloning. Next experiment must include genuinely novel
receiver facts, held-out test items, wider independent seeds, and compute-
matched training baselines before claiming a general mechanism.
