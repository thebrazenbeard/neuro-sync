# Neuro-Sync evidence contract: falsify capability transfer first

This document separates **experimental attribution** from passing source tests. It is a preregistration target, not a receipt for an executed Qwen3.5-4B/ChatGPT experiment. It does not replace the Path A/B weight/distillation baselines in [ARCHITECTURE.md](ARCHITECTURE.md).

## What exactly changed?

Use different names for these different measured effects:

| Effect | Minimum direct observation | Claim not justified |
| --- | --- | --- |
| *Reference discovery* | A supplies a task-blind capability reference B did not previously have; the host independently verifies it | A taught B a new neural skill |
| *External-state utility* | An unchanged B produces a newly correct verified effect only when it receives permitted state, with donor disconnected | B's own parameters changed |
| *Protocol utility* | Same reference and task are safer or more recoverable over StateBus than over direct-file/manual transfer under matched controls | StateBus was needed for the answer itself |
| *Model-to-model contribution* | A extracts useful information that the broker, B, and public fixture could not directly provide; without A, equally scoped runs fail | Mere A→B message forwarding qualifies as learning |
| *Weight-space retention* | B's checkpoint/adapter hash actually changes and B still performs better after a cold restart without external state | A tool/context result proves trained plasticity |

An A-style marker in a mock JSON artifact is not an independent computational ability. A preinstalled host registry invoked through a retrieved key is a legitimate *reference-discovery* demonstration, not a learned model capability. The old source-only zero-weight pilot intentionally denies the no-state branch in code, so its success alone cannot measure generalization or the incremental value of StateBus over ordinary retrieval.

## Counterfactual evaluation matrix

Hold the B artifact, tasks, host tool catalog, permitted budget, and scorer fixed across arms. Change **one assigned condition** at a time and run each receiver in a fresh process. Keep the evaluator's sealed task answers outside B's inputs and the committed bus payload.

| Arm | A contribution | B state route | Expected diagnostic value |
| --- | --- | --- | --- |
| N: no state | Same as V, unavailable to B | None | Baseline against hidden context or automatic tool discovery |
| V: valid StateBus | A discovers task-blind reference under separate consent | Governed authorized read | Candidate external-state capability gain |
| P: plain-file/manual | Same reference as V | Direct scoped input, no bus | Whether StateBus adds anything beyond information delivery |
| D: decoy | Same availability/size, wrong reference | Governed authorized read | Negative control for prompt length, retrieval placebo and lucky guesses |
| X: stale or revoked | Previously valid reference, now expired/revoked | Governed read/use denied | Whether revocation is enforced at the actual effect boundary |
| F: forged A | No genuine A observation | Producer-crafted proposal | Whether a self-certified donor can bypass host validation |
| H: host-only | No model A; operator/broker supplies the same reference | Governed authorized read | Whether A contributed beyond an ordinary registry lookup |

Do not declare H or P a defect if it ties V on task correctness: **that is the expected result when A merely supplies a locator to a preinstalled deterministic tool**. Report the tie. To justify A-specific value, design a later task where A's unique observation is necessary and ethically/technically available, then compare H and A-absent baselines under the same rights policy.

### What the source-only fixture can demonstrate

Synthetic, scripted A/B adapters can test message serialization, per-session scope, reference replay refusal, local crash/restart readback, task-blind capability discovery, and externally scored tool effects. Do **not** infer complete revocation: the original source fixture only checked a grant before commit, and a reproduced hostile test showed that toggling its grant after commit did **not** prevent B's later execution. The original `StateLedger.project(target_id,session_id)` is also a local DB projection without a caller/principal authorization parameter: knowledge of the DB path and identifiers can reveal state. Until fixed and independently retested, both live **use-time execute** and **read** grants are unqualified. These fixtures cannot demonstrate Qwen or ChatGPT reasoning, heterogeneous model semantics, native ChatGPT runtime memory, weight exposure, or independently authenticated validator custody. A B-only Python subprocess is **not** a cold restart of a neural model.

### What a real-model qualification must demonstrate

1. Verify exact host identity, active workloads, user ownership and an isolated `$0` process. Do not stop or restart a shared loaded Ollama/KoboldCPP service; reserve an idle test-owned runtime first.
2. Pin the exact B model file(s), tokenizer/configuration, adapters and runner version. Hash the same artifacts before and after every no-training arm. Keep a distinct manifest for external StateBus state.
3. Supply one synthetic, unfamiliar capability map with **no task answer** and a verified rights grant. Control tool exposure and available operations identically for N/V/P/D/X/F/H. Randomize blinded task order and reference assignment; store only safe fixture IDs and hashes.
4. Terminate B's process and any contextual cache, disconnect A, then start B from the pinned artifact. Verify the model process ID/start time and model checksum, not just the name in an API header.
5. Have a separately approved host verifier execute/inspect **actual outputs** against a sealed task oracle and exact source/test revisions. Do not count model self-grading or a verifier supplied by the producer as independent evidence.
6. Report per-arm successes, abstentions, errors, bytes, runtime/RAM and failed grants as well as wins. Repeat with at least two model architectures only if their endpoints and resource use are genuinely qualified.

An actual ChatGPT↔local bridge requires a supported, explicitly installed and authenticated ChatGPT tool/client mediation surface. A source `bridge.py` adapter or documentation of MCP support is not proof of an installed bridge. No paid API route, private corpus, credentials, or personal transcript is authorized for this experiment.

## Receipt schema (minimum fields)

Each externally reviewed result should record: `experiment_id`, `arm`, randomization seed, source Git commit OID, model runtime/provider, exact model manifest/checksum(s), donor/receiver process lifecycle, state file/hash/revision, rights grant reference, payload and capability reference hash, verifier principal/policy/test hash and test environment, hidden split hash (not answers), raw task output artifact digest, observed scorer outcome, duration/resource budget, and an evidence class (`SOURCE_INSPECTION`, `TEST`, `RUNTIME_PROBE`, or independently witnessed `FORMAL_VERIFY`).

Do **not** issue the last class when the proposer and its host callback control both the test and witness. A SQLite hash chain can detect some corruption; a database administrator can rewrite/rechain it without an external trust anchor. Separate `LOCAL_APPEND_UNWITNESSED` from a genuine externally witnessed commit.

## Stop and promotion rules

- A **green CI run** promotes the exact tested mechanics only. It does not establish a model effect.
- If V only differs from N because N is explicitly code-gated, report *reference-gated operation*, not learned capability.
- If P and V tie on task outcomes, look for genuine policy/replay/recovery benefits before claiming protocol value.
- If H and V tie, A's unique contribution is **not established**.
- If task answers or secret sentinel strings appear in the bus payload, logs, public CI, or prompts, the attribution/privacy test fails.
- If live model hashes, restart traces, host-owned verifier, or independent task oracle are missing, the real-model acceptance remains **UNKNOWN**, not PASS.
- If the current projection/execute path lacks a fresh principal-specific read/use grant, **NO-GO for a hosted or multi-tenant StateBus**, regardless of local replay/TTL tests.
- If any path entails a nonzero external compute or tunnel cost, do not execute it without Patrick's explicit authorization.

The central question remains open: **Can heterogeneous inference engines use the same governed capabilities, with useful verified effects and unchanged neural weights?** This matrix gives that claim a way to lose.
