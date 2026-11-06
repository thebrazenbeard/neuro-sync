# Neuro-Sync: Source Architecture and Qualification Boundaries

**Status: research code, not a distributed LLM runtime.** This document describes
the committed toy experiments on `main` at `e95b591da42e7671eda51e884988ec273aa2e6d3`
and the separately proposed small-library API. Future PRs may change the source
tree; inspect the exact Git head before treating a component as implemented.

## What exists and what does not

| Component | Main at `e95b591` | Claim ceiling |
| --- | --- | --- |
| `experiments/toy-transfer/adapter_transplant.py` | Implemented, stdlib | Compatible toy tensor/delta copy and rollback mechanics |
| `experiments/toy-transfer/toy_factual_experiment.py` | Implemented, stdlib | Synthetic cue-to-answer association controls, **not** learned Trek knowledge |
| `experiments/toy-transfer/neuro_sync_poc.py` | Implemented, optional PyTorch | Local projection/recipient-adapter gradient mechanics; no real LLM transfer |
| `src/neurosync/` | **Not in this base head** | Separately proposed auditable stdlib library; do not import from main until merged |
| Network transport, VN2/Lappy execution, shared optimizer, model checkpoints | **Absent** | No runtime or host qualification claim |

The historical `core/`, `hooks/`, `mesh/`, and `runtime/` sketch named paths that
never existed. An `mmap` file is not network-distributed weight synchronization.
Independent models do not share a gradient graph simply because their tensors
can be transmitted to one another.

## Proposed public small-library contract (pending its own PR)

The core developer's planned `src/neurosync/core.py` API consists of immutable
`Tensor` values, `ModelSnapshot` identifiers and named tensors, `ScopedGrant`,
`DeltaBundle`/`DeltaManifest`, and `TransferReceipt`, with operations:

```python
build_delta_bundle(donor_base, donor_trained, receiver_id, transfer_id)
apply_delta_bundle(receiver, bundle, export_grant, import_grant, strength=1.0)
rollback_transfer(current, receipt)
```

These are **target signatures, not working imports on the base head**. The
implementation PR must separately prove exact-base/schema/shape compatibility,
finite numeric values, no writable donor-recipient tensor aliasing, immutable
source snapshots, bounded transfer strength, deterministic artifact fingerprints,
and rollback against tampered receipts. A fingerprint that intentionally excludes
`model_id` identifies **content**, not the operator or authorization subject.

`ScopedGrant` in this toy API is a structured permission *fixture*, not an
authenticated rights-holder credential, digital signature, proof of consent,
or anti-replay service. Real remote coordination needs independently qualified
identity, key custody, time-bounded grants, revocation, replay controls, and
durable independent source/recipient records.

The package metadata in `pyproject.toml` has no mandatory runtime dependencies.
The optional `torch` extra supports only the PyTorch toy. A publishable install
and supported import API remain gated on merging and testing `src/neurosync/`.
No package publication or implied external redistribution license is authorized.

## Compatibility tiers (separate evidence classes)

| Tier | Compatible input | Permissible claim after passing its tests |
| --- | --- | --- |
| 0 — mechanics | Synthetic equal-shape tensors and toy facts | Transformation, isolation, and deterministic rollback behave as coded |
| 1 — exact base | Same architecture, initialization/checkpoint hashes, tokenizer and adapter mapping | Target may accept compatible delta/LoRA update; test persisted capability separately |
| 2 — related backbone | Closely related checkpoints with mapped layers | Evaluate raw copy, TIES/DARE, and mismatch risks; no automatic compatibility |
| 3 — heterogeneous models | Different coordinates, architectures or tokenizers | Requires learned projection/target-native adapter or distillation and independent evaluation |
| 4 — distributed | Qualified local tier plus hosts, leases, transport, state custody | Only then assess live cross-device transport and failure recovery |

Tier 0 is not a weaker version of Tier 4 proof: it tests a different subject.
Existing PEFT, mergekit and PorTAL research are background approaches, **not
dependencies or imported code in this source tree**.

## Strict two-clone Star Trek pilot

The experiment starts with byte-identical randomly initialized models A and B.
**Only A** sees the authorized, bounded teaching facts. B remains untouched
until the explicit transfer arm. A separately labeled retention variant may
pretrain B on unrelated material, but must not be mixed into strict-pilot
results. Random initial weights cannot be presumed natural-language competent.

The public synthetic toy under `experiments/toy-transfer/` is a mechanics
fixture with constructed cue/answer weights, not a sourced canonical Trek fact
dataset. The future factual teaching set may contain short, independently
verified CC0 Wikidata structured statements and original phrasing. It must
never wholesale ingest scripts, episode transcript dialogue, book text,
publisher synopses, artwork, or unlicensed wiki prose. Patrick's
`trek-data-core` contains **candidate metadata**, not accepted story evidence
or a cleared long-text training corpus. Its proprietary license is not a
blanket third-party content license.

Keep three disjoint classes:

1. **A-taught fact, new question wording:** within-fact transfer/representation
   check. Freeze entity/episode grouping before constructing paraphrases.
2. **Untaught fact:** negative guessing/retrieval control. Correct answers are
   not scored as evidence of transfer.
3. **Privately sealed misleading-premise oracle:** never committed to public
   GitHub, exposed to model prompting/retrieval, included in examples, or copied
   into training data. Public tests may verify only manifest/hash discipline,
   without containing oracle answer strings or revealing sealed questions.

Pretraining exposure is a major confounder for public Trek facts; separately
generated clearly **NONCANON** original entities can test actual novelty.
Consent/rights manifests must identify source, license, acquisition timestamp,
source/revision hash, split seed and grouping, and how to revoke/rollback.
Shared upstream Wikidata claims are one evidence lineage, however many files
repeat them.

## Donor-disconnect proof, baseline and safety

Deleting one Python variable or disabling one in-process caller is insufficient
to establish persistence. The meaningful test writes B's derived artifact,
terminates the donor and all caches, then starts a **separate recipient-only
process** from B's artifact and scores the held-out test without donor access.
Every run records exact baseline B hash, newly saved B hash, transfer receipt,
and rollback result. A nonpersistent activation-only arm is a control, not
failure of model-to-model communication.

Compare no transfer, compatible copy, randomly shuffled updates/activations,
untrained donor, distillation, and the experimental bridge under matched
budgets. Check leakage, long-form dialogue, negative transfer and forgetting;
do not equate agreement on answers with weight transfer. Any claimed learned
benefit requires independent held-out statistics beyond the deterministic toy.

Original model weights, keys, credentials, private oracles, and prompt logs
must never enter public CI or GitHub artifacts. Do not use protected hosts
without fresh operator approval and resource checks. No cloud compute,
paid APIs, hosted models, large runners, or remote deployments are part of
this implementation.

## Reproduce the source-level checks

From the repository root (no package install or network request):

```bash
python3 -m unittest discover -s experiments/toy-transfer -p 'test_*.py' -v
```

After the separately proposed stdlib core implementation is merged:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_core_*.py' -v
```

The toy projection's six tests skip explicitly when `torch` is absent.
GitHub Actions CI uses only a **standard free runner while the repository is
public**, read-only checkout, the interpreter shipped on that runner, and
source-level unittest checks; it neither installs PyTorch nor downloads
model weights. No CI result qualifies real-model performance, donor-disconnected
cross-process persistence, host readiness, or distributed transfer.

