# Neuro-Sync: Source Architecture and Qualification Boundaries

**Status: source-level research code, not a distributed LLM runtime.** Main at
`5be0e399ee5e4b635963cf9b5bd678597918a9e3` includes the historical toy
baseline, the strict Trek categorical-SGD experiment (PR #4), the independent
exact-base stdlib transfer API (PR #3), and the signed-zero fingerprint
regression fix (PR #6), plus cross-component mechanical integration tests
(PR #7). This document describes those checked-in components; inspect the
exact Git head for later changes.

## What exists and what does not

| Component | Main at `5be0e39` | Claim ceiling |
| --- | --- | --- |
| `experiments/toy-transfer/adapter_transplant.py` | Implemented, stdlib | Compatible toy tensor/delta copy and rollback mechanics |
| `experiments/toy-transfer/toy_factual_experiment.py` | Implemented, stdlib | Synthetic cue-to-answer association controls, **not** learned Trek knowledge |
| `experiments/toy-transfer/neuro_sync_poc.py` | Implemented, optional PyTorch | Local projection/recipient-adapter gradient mechanics; no real LLM transfer |
| `experiments/trek-transfer-v0/{fixtures.py,learner.py,predictor.py,run.py}` | **Merged in #4**, stdlib | Small categorical SGD learner; six short sourced Trek symbol triples plus six separate synthetic noncanon triples |
| `experiments/trek-transfer-v0/test_transfer.py` | **Merged in #4**, stdlib | Eight targeted tests of initialization, controls, and fresh-process recipient readback |
| `src/neurosync/core.py`, `src/neurosync/__init__.py` | **Merged in #3**, stdlib | Exact-base tensor-delta extraction, bounded transfer, typed receipt, rollback; no authenticated rights or real model adapters |
| `tests/test_core_transfer.py` | **Merged in #3**, stdlib | Adversarial core mechanics, numeric-integrity checks, no real-model effect |
| `tests/test_core_trek_integration.py` | **Merged in #7**, stdlib | Five cross-component tests: train A via Trek TinyLearner, map its parameters into exact-base core snapshot, transfer, subprocess-score B; still same taught symbolic queries |
| `docs/STATEBUS.md` | **Proposed higher-level wire/trust protocol**, not installed | Path C source contract; JCS and independent witness not implemented |
| `src/neurosync/bus/{schema.py,ledger.py,__init__.py}` + `tests/test_bus_core.py` | **Merged #9**, source/local SQLite mechanics; metadata integrity follow-up **#11** | Typed StateDelta, session-scoped revision/PROPOSE→VALIDATE→local COMMIT, not external verified authority |
| `src/neurosync/bus/bridge.py` + `tests/test_bus_bridge.py` | **Merged #10**, source tests | User/tool-mediated JSON import/export and bounded opt-in loopback client; not installed ChatGPT MCP or Qwen connection |
| Deployed remote transport, VN2/Lappy execution, shared optimizer, production model checkpoints | **Absent** | Source loopback client does not prove any live host or distributed runtime |

The historical `core/`, `hooks/`, `mesh/`, and `runtime/` sketch named paths that
never existed. An `mmap` file is not network-distributed weight synchronization.
Independent models do not share a gradient graph simply because their tensors
can be transmitted to one another.

## Current small-library API and limits

The implemented `src/neurosync/core.py` API consists of immutable
`Tensor` values, `ModelSnapshot` identifiers and named tensors, `ScopedGrant`,
`DeltaBundle`/`DeltaManifest`, and `TransferReceipt`, with operations:

```python
build_delta_bundle(donor_base, donor_trained, receiver_id=receiver_id, transfer_id=transfer_id)
apply_delta_bundle(receiver, bundle, export_grant, import_grant, strength=1.0)
rollback_transfer(current, receipt)
```

These are real public source functions, re-exported by `src/neurosync/__init__.py`.
The implementation enforces exact-base/schema/shape compatibility, finite
numeric values, isolated immutable result tensors, `0≤strength≤1`, deterministic
artifact fingerprints, and receipt-bound rollback against accidental tampering.
These checks are **mechanical**, not an authenticated grant service or formal
proof of mathematical/semantic safety. In particular, strength bounds a
coefficient, **not** the update norm or its behavioral effect. A fingerprint
that intentionally excludes `model_id` identifies **content**, not the
operator or authorization subject.

`ScopedGrant` in this toy API is a structured permission *fixture*, not an
authenticated rights-holder credential, digital signature, proof of consent,
or anti-replay service. Real remote coordination needs independently qualified
identity, key custody, time-bounded grants, revocation, replay controls, and
durable independent source/recipient records.

The package metadata in `pyproject.toml` has no mandatory runtime dependencies.
The optional `torch` extra supports only the PyTorch toy. Source import uses
`PYTHONPATH=src`; this packaging PR was also smoke-tested with an offline,
no-dependency wheel build and installation to an isolated temporary target,
followed by import of the public symbols. That establishes **packaging
mechanics for this exact tree**, not a published release or a production API.
No package publication or implied external redistribution license is authorized.

## Three transfer paths and StateBus qualification

The [StateBus v0 proposal](STATEBUS.md) is a distinct **Path C zero-weight** state/context mediation route. The existing exact-base library and Trek SGD fixture investigate **Path A compatible weight copy** and a bounded **Path B ordinary donor-output distillation control**; neither means a ChatGPT session has native access to another model's neural parameters. StateBus can carry claims and evidence about these activities but cannot authorize their protected effects on its own.

| Track | Required actual effect | What current source does **not** prove |
| --- | --- | --- |
| A — weight copy | Exact compatible donor/receiver checkpoint+schema and authorized independent B artifact | Heterogeneous compatibility, learned generalization, model self-consent |
| B — distillation | Authorized teacher signals, actual B-local optimization, B-only persisted checkpoint | Novel cross-gradient transfer or improvements beyond ordinary distillation |
| C — zero-weight bus | Typed, scoped, externally committed state/context under verified actor permissions | B parameter change, ChatGPT native memory overlay, distributed cognition |

Immediately available *in principle* is client-mediated export/import of bounded messages. Official OpenAI docs also describe an **optional installed custom MCP server** using Server URL or Secure MCP Tunnel, subject to actual install/auth and cost gate; it is tool access, not hidden token-stream or weight hooks. An optional qualified local model endpoint likewise requires fresh host, exact model, RAM/workload and permission checks. Do not infer an active plugin, loopback server, private Blotter bus, local Qwen spare capacity, remote consensus or zero-cost tunnel from documents. A full secure tunnel may require Platform credentials and billable usage; no paid path is authorized here.

**Evidence ceiling:** the higher-level wire/trust protocol remains proposed; the merged StateDelta/StateLedger and bridge have source-level tests only, with no authenticated external witness or host deployment. PROPOSE → VALIDATE → COMMIT requires an independent validator and external durable witness for a protocol-level commit. A method called StateLedger.commit() that merely appends locally is **LOCAL_APPEND_UNWITNESSED** until separately attested. Typed claimed_evidence_class and model-provided sender/surface names have no independent authority; neither do caller-minted fixture grants. Transient functional context must never be auto-promoted to memory, plasticity or identity.

Qualification compares **(A) B with no bus/no state**, **(B) B with read-only authorized StateBus context but unchanged weights**, and **(C) B with changed, hashed recipient weights cold-restarted while bus disconnected**. Add stale/decoy/replay, secret-sentinel, cross-session, capability-revocation and fork tests. A result limited to B is context mediation, not neural transfer. A 50-turn, Qwen3.5-35B or fully native integration claim requires independent actual runtime receipts; none is established by these docs.

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
dataset. The separately merged `experiments/trek-transfer-v0/` uses six short,
manually encoded Trek factual triples with first-party StarTrek.com source URLs,
and six **NONCANON** original triples in a separate fixture. These facts train
and test the exact same symbolic queries; even a perfect score does **not**
prove unseen-question generalization or fresh pretrained-LM knowledge. Future
fact sets may use short, independently verified CC0 Wikidata structured
statements and original phrasing. No experiment may wholesale ingest scripts,
episode transcript dialogue, book text, publisher synopses, artwork, or
unlicensed wiki prose. Patrick's
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
without fresh operator approval and resource checks. No **paid external
model-training compute**, hosted model APIs, GitHub larger/GPU runners, or
remote deployments are part of this implementation. Standard free public
GitHub Actions CI executes only short source-level tests.

## Reproduce the source-level checks

From the repository root (no package install or network request):

```bash
python3 -m unittest discover -s experiments/toy-transfer -p 'test_*.py' -v
python3 -m unittest discover -s experiments/trek-transfer-v0 -p 'test_*.py' -v
```

The Trek source's local commands are
`python3 experiments/trek-transfer-v0/run.py --dataset trek` and
`python3 experiments/trek-transfer-v0/run.py --dataset synthetic`.
Both run a tiny structured learner and evaluate B in a fresh Python subprocess;
neither trains or transfers between actual LLMs or accesses a remote donor.

For the committed stdlib core and StateBus source:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_core_*.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_core.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_bus_bridge.py' -v
PYTHONPATH=src python3 -c 'from neurosync import Tensor, ModelSnapshot, build_delta_bundle, apply_delta_bundle, rollback_transfer; print(Tensor.from_nested([[1.0, 2.0]]).shape)'
```

The toy projection's six tests skip explicitly when `torch` is absent.
GitHub Actions CI uses only a **standard free runner while the repository is
public**, read-only checkout, the interpreter shipped on that runner, and
source-level unittest checks; it neither installs PyTorch nor downloads
model weights. The Trek fixture does exercise recipient-only **subprocess**
readback after removing the donor. No CI result qualifies donor-disconnected
**real-LLM** persistence, novel factual generalization, host readiness, or
distributed transfer.

