# Trek transfer v0: actual SGD, controlled copying, and ordinary distillation

**Status:** a dependency-free, tiny *categorical neural learner* experiment. It is
not a pretrained language model, Transformer, English paraphrase benchmark,
model-to-model gradient bridge, or a demonstration of unified cognition.

This tests Patrick's *strict initial condition*: initialize random weights once,
clone them byte-identically into A and untouched B, train **only A** on a small
set of sourced Star Trek factual associations, and test the independent B
receiver **after deleting donor A**. Do not quietly pretrain B or add unrelated
knowledge in this strict experiment; B-retention is a separate later variant.

## Run locally for $0

Python 3.9+ standard library only; no GPU, network, accounts, tokens, or model
downloads. From the repository root:

```bash
python -m unittest discover -s experiments/trek-transfer-v0 -p 'test_*.py' -v
python experiments/trek-transfer-v0/run.py --dataset trek
python experiments/trek-transfer-v0/run.py --dataset synthetic
```

The script prints a JSON receipt with initial model hash, trained donor hash,
training update counts, independent-process B checkpoint hashes, ordered query
transcript hashes, exact-query predictions, accuracy and limitations. The parent
rejects malformed subprocess responses, wrong ordered-query hashes, or missing,
extra, or out-of-vocabulary predictions before scoring. This detects mismatched
receipts; it is **not** a cryptographic attestation of independent execution.
Temporary B checkpoint files are
discarded when the command ends. Nothing is uploaded or installed.

## What the neural learner actually does

- Fixed closed vocabulary of subject, relation, and answer *symbols*, not English.
- Random trainable subject and relation embeddings; `tanh` hidden activations;
  softmax classifier with trainable output weights and biases.
- Explicit cross-entropy gradients and SGD updates for **A**, rather than
  constructing an associative answer-column lookup.
- A is trained on six public `(subject, relation) -> answer` associations. B's
  initial hash is identical and receives **zero training** until a named arm.
- `C3` performs ordinary output-distribution distillation: B receives six
  numerical donor probability vectors associated with its queries. It does not
  receive donor checkpoint weights, answer labels, or answer text in that arm.
- B-only evaluation runs in a **fresh Python subprocess** executing
  `predictor.py`, loading only the saved recipient checkpoint and structured
  query symbols. It is not an independently rebooted computer and has no donor
  file, shared memory, service, adapter, or inference context to consult.

The answer classes themselves are shared symbols at initialization; the
experiment tests *learned associations between symbols*, not acquiring wholly
new answer-token vocabulary or natural-language understanding. No claim of
generalization is supported by training and scoring the same six query IDs.

## Controls

| Arm | Exposure | Recipient updates | Interpretive ceiling |
| --- | --- | --- | --- |
| C0 untouched | None | 0 | Random baseline, exactly original B state |
| C1 full donor copy | Clone trained A parameters into disposable B | 0 SGD; full state replacement | Straightforward copying, not Neuro-Sync |
| C2 exact-base delta | Apply `A_after - A_initial` to exactly matching untouched B | 0 SGD; effective parameter transplant | Equivalent to A to floating-point precision, not independent learning |
| C3 output distillation | B trains against A's numerical answer distributions for aligned queries | B-local SGD | Ordinary knowledge distillation, persistent without donor |
| C4 shuffled wrong donor | B receives donor distributions for different queries | B-local SGD | Negative control for pairing/label alignment |

Donor A's supervised update count and each B arm's update count are reported.
Copying is not compute-matched to distillation; the experiment **does not**
support claims about superiority, latency, efficiency, or state-of-the-art
effectiveness. This is a controlled mechanical demonstration only.

## Sources, rights, and reserved tests

`fixtures.py` contains six independently encoded, short canonical Trek factual
triples with a per-row **official StarTrek.com URL** and `source_type=official`:
Voyager captain/registry/class, Spock heritage, Deep Space Nine commander,
and Worf species. There are **no scripts, captions, episode dialogue, synopses,
retrieved copyrighted paragraphs, or private evaluation items** in this
directory. Short factual identifiers do not purport to license downstream
reuse of any external expressive work. Review source rights for larger datasets.

The separate `--dataset synthetic` option uses original, explicitly
`synthetic_noncanon` fabricated identifiers; no synthetic row claims to be
canonical Star Trek. Both variants use the same mechanics.

No reserved episode-specific misleading-premise test prompt, answer, scoring
rubric, or reversible encoding of those items is included. Keep any such oracle
private and fully excluded from training, distillation, retrieval, and source
publication.

## What would constitute the next experiment

1. Seal a larger fact/phrasing split independently before any training. Test
   unseen paraphrases of *taught* facts separately from truly *untaught* facts.
2. Add a separate B-retention variant with B-local non-Trek learning **after**
   the initial clone, measure degradation after transfer, and keep its results
   distinct from this strictly untouched-B pilot.
3. Introduce an actual activation bridge with no donor answer labels or text
   and a differentiable B-native objective; compare against C3 distillation.
4. Match total A+B forward/backward compute where comparing efficiency, use
   several independent seeds, and run receiver-only fresh-process evaluation.
5. Verify authorized access, checkpoint provenance, and persistence on an
   actual runtime separately before any Lappy/VN2 deployment.

The results, even if numerically perfect, do not establish LLM learning,
semantic transfer, new knowledge beyond taught associations, consciousness,
or a distributed inference service. Mechanically **persistent** recipient
learning can occur through ordinary distillation, and simple copying can be
an equally valid answer to the narrow transfer question.
