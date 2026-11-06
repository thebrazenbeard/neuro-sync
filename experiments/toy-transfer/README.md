# Neuro-Sync: controlled toy probes

This isolated scratch prototype explores two **different** mechanisms. It does
not implement the `thebrazenbeard/neuro-sync` GitHub README, perform LLM
training, or establish a unified cognitive topology.

## Probe A — differentiable heterogeneous coupling (PyTorch)

`neuro_sync_poc.py` has distinct frozen CPU neural encoders: 4→13→7 and
4→11→5, plus a frozen target classification head. A 7→5 projector is learned
via autograd. Only the target-local 5→3→5 adapter, projector, and sigmoid gate
train. The mixing factor `alpha` is explicit; `alpha=0` gives an exact
target-local adapter baseline, with no cross-source effect or alignment loss.
There is a paired-source and deterministic shuffled-source negative control.
No raw gradients are added to mismatched parameter shapes; no `.data` writes;
no cross-model LoRA aliases; no impossible rectangular-isometry constraint.
The **scalar gate** is bounded; the projector output magnitude, total residual,
gradient norm, and training dynamics are **not** bounded by that gate alone.

Run with **preexisting** CPU PyTorch (never installed by this task):

```bash
python -m unittest -v test_neuro_sync_poc.py
python neuro_sync_poc.py
```

The second command prints three *training losses*, not a claim of transfer
quality. Held-out paired-vs-shuffled evaluation, many seeds, scaling, memory
budgets, and continual-learning effects remain unproven.

## Probe B — exact-base delta transplant (pure Python)

`adapter_transplant.py` applies a declared delta matrix to a **matching**
receiver base after checking local-test permission, target ID, dimensions,
base SHA-256, and delta SHA-256. It constructs immutable receiver weights,
so donor data can disappear without changing the receiver. Rollback refuses
to overwrite a receiver that changed after the transplant. The permission
manifest here is a toy fixture, **not** authenticated consent or rights proof.

Run without external dependencies:

```bash
python -m unittest -v test_adapter_transplant.py
```

It measures copying and isolation mechanics only. A copied matrix behaving
independently does not prove learned knowledge, activation transfer, general
intelligence, or durable cognition.

## Probe C — synthetic structured fact quiz (pure Python)

`toy_factual_experiment.py` creates two identical **untrained zero-weight**
associative models. Only donor A receives three small **fictional, noncanonical
Starfleet-style** subject/relation/answer facts. Separate receiver B arms are
queried **after A is deleted**: untouched before; text-only exposure followed
by context reset; full A matrix copy; hash-checked exact-base delta transplant;
and activation-score distillation with **B-local** learning. The interaction
arm's weights differ from A's and survive disconnection, but that mechanism is
ordinary supervised distillation, not a novel Neuro-Sync method.
An additional shuffled-pairing negative control receives activations for the
wrong cues and fails the three exact-key questions.

```bash
python -m unittest -v test_toy_factual_experiment.py
python toy_factual_experiment.py
```

The fixed vocabulary and training cues prevent testing generalization to
unknown facts, new language, or paraphrased questions. All quizzes are blind
to the removed donor; they are not held-out semantic generalization exams.

## Patrick's falsifiable Star Trek experiment — next real milestone

**Literal experimental question:** Instantiate a small model from scratch;
duplicate its *untrained* exact weights to branches A and B. Teach **A only**
bounded, sourced Star Trek facts. Connect A and untouched B through an
explicitly measured Neuro-Sync interaction. Disconnect/remove A and reset B's
context. Can B still answer previously unseen Trek questions accurately,
using only B's persisted parameters or adapter state?

Preserve the same held-out question set and explicit source/contamination
checks for these **separate** arms:

1. **B before** — untouched B, with no A exposure.
2. **B after text-only chat** — B sees comparable examples in context, then
   context is reset; check for genuine persistence.
3. **B after full copy** — copy A's trained state to B; a deliberate positive
   control, clearly distinct from Neuro-Sync.
4. **B after adapter/delta transplant** — exact-base compatible A update
   transplanted and preserved in B, without A online.
5. **B after true interaction** — no weight/adapter copy; only the specified
   coupling channel and allowed B-local learning updates, then donor disconnect
   and reset.

Pre-register exact starting-base hashes, train/eval split, donor training set,
allowed communication channels, B update permissions, changed tensor hashes,
optimizer state, checkpoints, restart procedure, and evaluation scoring.
Report both gain and wrong-answer/interference rates against every control.
If B's parameters/adapter and other persistent memory have **not** changed,
and there is no retained external donor, post-reset knowledge gain has no
identified physical storage mechanism. If A's weights are copied, that is the
**copy** control, not evidence of a novel coupling mechanism.

Maintain any advanced, episode-specific misleading-premise QA oracle **only**
outside this public repository and all training/retrieval artifacts. Commit at
most a nonreversible item hash/ID after private oracle creation; keep its exact
prompt, expected answer, source material, and scoring rubric sealed until
evaluation. Do not claim the toy passed that private real-language holdout.

**Status at publication:** PyTorch is absent in the execution environment;
those tests are written but could not be executed there. The pure-stdlib
transplant tests ran. No Lappy, VN2, production model, or shared runtime was
modified; repository publication is source-only.
