# neuro-sync

**Status: research proposal and bounded experiments — not a working distributed LLM-synchronization runtime.**

`neuro-sync` investigates whether two independently running models can **consensually transfer a learned capability** through compatible weight changes, target-native adapters, distillation, or activation communication. A successful transfer must leave the *recipient* measurably more capable **after the donor disconnects**. Merely exchanging messages, sharing an activation during one inference, or obtaining similar answers does not establish a persistent weight transfer.

This project concerns **model-to-model transfer**, not human EEG or biological neural synchronization. “Synchronization,” “neural,” and “shared cognition” are research metaphors, not evidence of a common mind, consciousness, identity, or merged topology. Interacting models do **not** automatically expose their weights to one another: access to checkpoints, adapters, activations, or gradients must be explicitly engineered and authorized.

## Research question

Suppose models A and B start as two **byte-identical, untrained copies** of a small model. A learns Star Trek facts; B learns a separate, non-Trek set. If they interact through a specified, authorized transfer mechanism, can B retain A's facts **in B's own checkpoint** without forgetting its original facts? This is the initial falsifiable pilot, not an assertion that it already works.

Here, “untrained” means random initial parameters, **not** a chat-capable pretrained LLM. The strict pilot uses structured `(subject, relation) → answer` items because raw random weights do not already understand English. A separate, later experiment can clone an identical *pretrained* base and test natural-language learning. Do not silently combine evidence from those different experiments.

The simplest successful explanation may be **copying a compatible adapter** or **ordinary distillation**. Those are legitimate transfer mechanisms and essential baselines; they are not evidence that novel bidirectional gradient cross-feeding was required.

## Mechanisms: distinct hypotheses, not interchangeable implementations

| Pathway | Mechanism | Persistence in B | Constraint / current claim ceiling |
| --- | --- | --- | --- |
| **Exact-base adapter transplant** | With the exact same base revision, tokenizer, layer targets, and parameter conventions, copy an authorized LoRA adapter or its effective update into B | Yes, if committed and independently reloaded | Mechanically plausible and supported by existing PEFT practice; task retention still requires testing |
| **Related same-architecture delta merge** | Apply a bounded donor update in verified compatible parameter coordinates; compare with direct copy and established merge methods | Possible | Shapes alone do not establish semantic compatibility; negative transfer is possible |
| **Heterogeneous model transfer** | Learn an explicit activation representation map, train a **target-native** adapter, and/or distill donor behavior into B | Possible only if B actually trains/saves the result | Different tensor widths, tokenizer layouts, parameter bases, and sequence alignments rule out naïve tensor copying |
| **Online activation sharing** | During one forward pass, map donor representations into B and gate the result | No, absent a separately specified update/training step | An inference-time bridge is not lasting weight plasticity |
| **Bidirectional online gradient co-adaptation** | Specify a genuine differentiable joint objective, tensor mapping, update ownership, and synchronization protocol | Untested | Research hypothesis; original README equations did not define a valid implementation |

### Compatible adapter transfer: precise version

For a recipient linear layer with weight `W ∈ ℝ^(d_out × d_in)`, an ordinary LoRA update with `A ∈ ℝ^(r × d_in)` and `B ∈ ℝ^(d_out × r)` is

\[
\Delta W = \frac{\alpha}{r} BA, \qquad W' = W + \lambda\Delta W,\quad 0 \leq \lambda \leq 1.
\]

This is an **effective-weight** update, not evidence that raw LoRA factors can be transplanted between unrelated parameter bases. Even when dimensions match, different checkpoints can use different representations. Check exact base hashes, adapter target names, factor orientation, rank/scaling, quantization behavior, tied weights, and compatible licenses. Copy into an **independent recipient artifact**; never alias the donor's writable memory or mutate an original checkpoint. Same-base `λ = 1` can reproduce the donor adapter's effective update if all relevant conditions match; behavioral equality must still be checked.

### Heterogeneous transfer: candidate bounded bridge

Given an activation summary `a = pool(h_A) ∈ ℝ^(d_A)` and a recipient state `b ∈ ℝ^(d_B)`, learn `W_AB ∈ ℝ^(d_A × d_B)`, a target-local adapter `θ`, and a **nonnegative bounded gate**:

\[
u = \operatorname{norm}(\operatorname{stopgrad}(a) W_{AB}),\qquad
b' = b + g_\theta(b,u)u,\qquad 0\leq g_\theta\leq g_{\max}.
\]

Train the bridge and recipient adapter against a **defined recipient loss**; do not label an arbitrarily injected donor gradient the derivative of a joint loss. Explicitly align examples and sequence positions; pooling is a minimal toy design, not proof of semantic alignment. For durable transfer, train and save B's adapter, disconnect A, terminate any activation cache, restart B, and test B independently. Each direction (A→B and B→A) needs its own grant and evaluation.

### What was invalid in the original sketch

The initial README described code paths and effects that did **not** exist in the repository and contained mathematical inconsistencies:

1. An orthogonal `P ∈ ℝ^(d_A × d_B)` satisfying `PᵀP = I_(d_B)` is impossible if `d_A < d_B`; a generic learned map need not be orthogonal.
2. The written hidden-gradient multiplication used inconsistent tensor orientation, and `[batch, sequence, hidden]` activation gradients cannot be directly added to `[out, in]` weight gradients with `param.grad.data.add_(...)`.
3. For independent losses, `∂L_A/∂h_B = 0` **without a specified differentiable connection**. Cross-feeding a mapped direction is a heuristic unless a real joint objective and chain rule are defined.
4. The proposed `2 sigmoid(z) − 1` similarity gate becomes **negative** below threshold, potentially reversing rather than suppressing updates. A bounded gate should be nonnegative.
5. Applying `−γW` indiscriminately could decay frozen backbone weights. Adapter-only updates need explicit ownership and bounds.
6. `mmap` or CUDA IPC is not universal zero-copy **network** transfer. Concurrent readers/writers, host ownership, transport, backpressure, timing, and crashes require separate designs. An inference-only `no_grad` pass cannot update learned weights by magic.

These defects invalidate the **specific equations and claimed implementation**, **not** the broader possibility of compatible weight transfer or trained cross-model representation bridges.

## Pilot: two clones and donor-disconnect test

1. Generate a tiny model once, save its initialization hash, and clone it byte-for-byte as `M_A` and `M_B`. Keep clean, restorable copies.
2. Partition sourced Star Trek facts for A and disjoint non-Trek facts for B. Hold out unseen items and alternate phrasings before training. Keep training/validation/test contamination records. Separate factual-source provenance from synthetic mechanics fixtures.
3. Train A only on Trek and B only on its unrelated facts. Record checkpoint hashes and both baseline capabilities.
4. Evaluate **each** named transfer arm from the same baseline: no transfer; donor activations only; activation-assisted **recipient training**; compatible full-state copy; exact-base LoRA/adapter copy; ordinary response distillation; random/shuffled activations; untrained donor. Add heterogeneous learned mapping only when the same-base case is controlled.
5. **Disconnect A, clear caches, restart B from its saved checkpoint**, then test B independently on held-out Trek questions, B's original facts, and longer generation/dialogue. Verify B's state changed where an actual training arm claims plasticity; retained facts without the donor are the essential outcome.
6. Repeat under preregistered seeds, split hashes, compute accounting, and matched wall/resource budgets. Report failures, reversals, catastrophic forgetting, and the strongest competing baseline—not merely successful examples.

The initial structured toy experiment, proposed under `experiments/toy-transfer/`, is a **mechanics fixture**. Its donor facts are synthetic and its cues/score vectors are deliberately constructed. Small scores such as **3/3 on those three cues** (or **0/3 on shuffled cues**) are **by construction**, not a claim of generalization, Star Trek learning, transferable semantic representations, or a real LLM “mind meld.” The code proposal and its tests must be inspected on their own exact commit; **this README-only change does not add or run them**.

### Qualification gates before distributed execution

| Gate | Minimum evidence | If it fails |
| --- | --- | --- |
| Mechanical correctness | Compatible shapes/checkpoint hashes, finite gradients for learned bridges, isolated source and destination states, reproducible rollback | Stop; repair implementation |
| Toy persistence | B's saved/reloaded checkpoint works with **A fully offline** and no hidden cache/service | Classify as online sharing, not weight transfer |
| Comparative efficacy | Preregister ≥5 seeds and ≥1,000 held-out examples per task for a larger pilot; compare strongest eligible baseline with matched **A+B total compute** and task exposure. Target ≥3 percentage points absolute gain, 95% paired-bootstrap interval entirely above zero, ≥4/5 successful seeds, and ≤1 percentage point original-task regression | Do not promote to distributed runtime; report null or negative transfer |
| Robustness | Long-generation dialog, paraphrases, out-of-domain negatives, leakage checks, donor dropout, restart and rollback | Restrict claim to mechanics/fixture |
| Distributed readiness | Host resource qualification, latency/bytes budget, transport integrity, leases, crash containment, consent replay prevention, and exact artifact readback | Keep CPU/local and $0 |

The thresholds are **proposed preregistration criteria**, not achieved results. Five seeds alone do not establish independent samples; analyze task-item and seed variation appropriately. First prove a local effect. Only then evaluate whether VN2/Lappy networking adds value over a local baseline, subject to host authorization, idle resource checks, and the **$0 external-compute rule**.

## Data, state, and authority

No model's weights are implicitly “consensual” just because a process can read them. Require an explicit rights-holder/operator grant with donor and recipient identities, immutable base/model and adapter hashes, allowed tensors and transformation, purpose, destination, time scope, data license/rights, and revocation/rollback procedure. Store authorization separately from descriptive metadata. Preserve provenance for every derived artifact; original checkpoints are immutable; adapter changes are independently versioned and recoverable. Logs should not leak prompts, training data, weights, or credentials. External cloud services and costs are not authorized by this proposal.

Conceptual ecosystem references such as the BT2 Hyperconnectome and Plasticity/State Governance project sources suggest provenance and failure-containment questions; **they are not evidence of a deployed neuro-sync runtime**. Runtime qualification requires actual host probes and readback, not a source document or review consensus.

## Repository status and planned boundaries

The original `main` at commit `48245d5e59e1fe9261899e63b2fd156e6d686847` contained **only this README** and **no runtime implementation**. Experimental code, when accepted separately, belongs under `experiments/toy-transfer/` and must include its own executable test commands, dependencies, exact tested commit, and limitations. The former `core/`, `hooks/`, `mesh/`, `runtime/`, and `neurosync` import example were *illustrative paths and pseudocode*, **not present modules or a runnable API**. Do not advertise them as installed.

Suggested progression: (1) exact-base adapter transplant and signed/provenance-checked rollback; (2) controlled toy fact transfer with donor disconnect; (3) target-native learned bridge/distillation with fair baselines; (4) real-model trials; (5) qualified optional distributed transport. Each stage may disprove the need for the more elaborate one. Preserve negative results.

## Primary background and implementation references

- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), [multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html), and [distributed RPC limitations](https://docs.pytorch.org/docs/stable/rpc.html).
- [Hugging Face PEFT adapter APIs](https://huggingface.co/docs/peft/main/en/package_reference/peft_model): existing portable LoRA adapter operations, with compatibility/quantization caveats.
- [Communicating Activations Between Language Model Agents (ICML 2025)](https://proceedings.mlr.press/v267/ramesh25a.html): precedent for useful activation communication, not proof of persistent recipient weight change.
- [Activation Space Interventions Can Be Transferred Between Large Language Models (ICML 2025)](https://proceedings.mlr.press/v267/oozeer25a.html): learned cross-model maps as an empirical research direction.
- [Can Heterogeneous Language Models Be Fused? (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.939/): topology-aligned heterogeneous adapter transfer research; not generic raw-weight equivalence.
- [Transport and Merge (ICML 2026)](https://proceedings.mlr.press/v306/cui26j.html): cross-architecture transport/merge research; full-scale requirements differ from this tiny pilot.
- [mergekit](https://github.com/arcee-ai/mergekit) and [git-re-basin](https://github.com/samuela/git-re-basin): parameter-space merging and alignment baselines; do not assume all model architectures share a coordinate system.

**Bottom line:** the question is not whether two models *can* exchange weights. Compatible files and adapter deltas can be exchanged. The difficult, falsifiable question is whether the recipient acquires a **specific new ability**, retains it **without the donor**, and outperforms straightforward copying or distillation without unacceptable forgetting.
