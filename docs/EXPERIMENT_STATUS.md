# Neuro-Sync: execution status and adversarial evidence ledger

This file separates **a runnable experiment**, **an executed test**, and **a qualified real-model result**. It is maintained as source documentation, not as a live host or installed connector receipt. The initial checkpoint for this audit is [GitHub main `9cb31a1`](https://github.com/thebrazenbeard/neuro-sync/commit/9cb31a1feda8a6af70ae9160f7329b2b4decb141), after merged [PR #18](https://github.com/thebrazenbeard/neuro-sync/pull/18), [PR #19](https://github.com/thebrazenbeard/neuro-sync/pull/19), and [PR #21](https://github.com/thebrazenbeard/neuro-sync/pull/21). **Recheck live `main` before treating the Git SHA as current.** The [hostile quorum record, Issue #20](https://github.com/thebrazenbeard/neuro-sync/issues/20), is model review and reproduced local testing, not independent attestation or a Blotter event.

## Research tracks and claim boundaries

| Track | Implemented | Evidence class and ceiling | Not established |
| --- | --- | --- | --- |
| A. Compatible weight transfer | Tiny categorical learner, strict checkpoint validation, exact-base donor delta, toy adapter transplantation | SOURCE + local TEST + GitHub CI: mechanical copying/delta/checkpoint rollback; exact-base donor result is algebraically full donor-state copy | Novel learning advantage, pretrained Qwen transfer, different-model tensor alignment, rights-holder permission |
| B. Behavioral distillation | Trek v0 small supervised C3 arm and planned larger teacher/student protocols | SOURCE + bounded toy tests: trained on six seen `(subject, relation)` labels | Generalization on unseen labels, independently verified teacher rights, durable natural-language capability |
| C. Untrained runtime synchronization | SQLite StateBus proposal/commit, JSON wire/bridge, local gate fixtures; fair scripted N/V/P/H experiment | SOURCE + local TEST + CI: external reference routing and subprocess persistence **for a scripted receiver** | Heterogeneous LLM inference, actual fresh Qwen restart, independent principals, bus-specific task-accuracy gain |

**Central status: UNKNOWN.** No present experiment establishes that an unchanged real Qwen3.5-4B derives a new, independently verified semantic capability from another model's authorized external state after a cold model process restart. Neither passing Python tests nor agreement among AI reviewers upgrades that claim.

## What was actually tested

### Original causal fixture: an invalid no-state inference

The historical `causal_trial.py` seed-317 script produced no-state **0/8**, V **8/8**, ordinary-file P **8/8**, revoked **0/8**. However, B was passed a broker path containing the selected ref and a registry path with candidate coefficients even in the no-state arm. `adversarial_receiver.py` used those same paths and solved **8/8 without StateBus**. The supposedly no-state receiver itself contained a deliberate no-reference failure gate; the original result cannot measure protected information isolation. The P comparator reused a grant initialized through bus code. See [PR #18](https://github.com/thebrazenbeard/neuro-sync/pull/18) and [the fixture README](../experiments/zero-weight-sync/README.md).

### Fair-control v2: useful **scripted** comparison, not neural transfer

Merged [PR #19](https://github.com/thebrazenbeard/neuro-sync/pull/19) added `fair_trial_v2.py`, a separate `fair_receiver_v2.py`, and 15 focused tests. The synthetic host privately chooses among 13 candidate tool handles; donor A discovers an opaque reference by deterministic probes. Each B subprocess then **copies or guesses a handle** and host code evaluates simple affine/quadratic functions. No language model is called, no B weights change, and the host scoring process is not an independent OS principal.

| Condition | Observed correct effects | Honest interpretation |
| --- | ---: | --- |
| N: no ref | 11/128 | Scripted blind handle guesses. Under conditional uniform-target modeling, chance expectation is ~9.85 successes. |
| V: SQLite StateBus ref | 128/128 | Supplying the reference is sufficient for this scripted receiver. |
| P: independent plain JSON ref | 128/128 | Same correctness without StateBus storage; ordinary file is an essential baseline. |
| H: direct host ref, A absent | 128/128 | Donor/model A is not uniquely necessary when the host can supply the ref. |

Those are **16 synthetic scenarios × eight strongly correlated calls**. Scenario, not individual call, is the primary experimental unit (at most 16). The [statistics hostile analysis](https://github.com/thebrazenbeard/neuro-sync/issues/20) explicitly enumerated the fixture's conditional target distribution: `E[N] ≈ 9.846`, SD `≈ 3.054`, conditional 95% predictive range `4–16`, `Pr(N ≥ 11) ≈ 0.401`; it assumes independently uniform HMAC-derived targets and **is not a real-model p-value**. Since V=P=H receive identical reference inputs, their perfect tie cannot establish StateBus-specific answer accuracy, donor necessity, or semantic transfer.

The v2 improvement fixed two earlier synthetic confounders: public-seed reconstruction of the secret target and an oracle using the exact same function as the effect executor. The final v2 host uses private HMAC target selection and an independently coded frozen Horner polynomial evaluator. **Remaining limitations:** the host secret's public hash is published only after trial completion, so it is not an independent ex-ante registration; no independently protected scorer or OS custody exists; v2 `host_effects` does **not** check the current use grant, TTL, or revocation at each effect. Earlier grant tests on *different* source paths cannot prove v2 execution governance.

A separate native GPT specialist subagent (Pragmatism) ran 15/15 fair-v2 tests and a **200 warm local-read microbenchmark**: plain JSON median `≈6.04 µs`, `StateLedger.project` median `≈851.36 µs`. This is one same-host microbenchmark, **not** an end-to-end performance or operational-cost comparison or independent non-GPT model review. A protocol-specific benefit must be tested on recovery, conflict/replay handling, revocation, auditability, or other explicit dimensions rather than assumed from task accuracy.

### Trek checkpoint hardening: mechanical PASS, transfer hypothesis still open

Merged [PR #21](https://github.com/thebrazenbeard/neuro-sync/pull/21) added shape, vocabulary, finite-value and dimensional checks, bounded transactional SGD updates, and a numerically stable exact-base delta. A Purist reviewer first **vetoed** implementation after reproducing a valid checkpoint whose `base + (donor - base)` rounded past the permitted upper weight bound by one ULP. The corrected `math.fsum` path passed that regression, 2,000 randomized donor-digest roundtrips, and 1,500 randomized SGD steps (970 accepted/reloadable; 530 rejected with unchanged digest). The [GitHub Source mechanics run 42257074141](https://github.com/thebrazenbeard/neuro-sync/actions/runs/42257074141) completed successfully for PR #21. Six optional PyTorch cases were reported as skipped in the implementer's local suite. **This verifies only bounded toy mechanics**: with receiver exactly equal to base, the donor delta's resulting weights are equivalent to copying the donor; the six taught Trek labels were also evaluated on seen identifiers, not a held-out generalization test.

## Reproduce source-level checks ($0, no model downloads)

From the repository root with Python 3.10+:

```sh
python3 experiments/zero-weight-sync/fair_trial_v2.py demo
python3 -m unittest discover -s experiments/zero-weight-sync -p 'test_*.py' -v
python3 -m unittest discover -s experiments/trek-transfer-v0 -p 'test_*.py' -v
python3 -m unittest discover -s experiments/toy-transfer -p 'test_*.py' -v
```

Record the **exact Git commit, model/runtime artifacts (if any), test exit code, skips, randomization and raw output hash**. CI success applies only to the suite it executes at its own head; it is not a real-model probe.

## Executed VN2 local-model smoke: runtime inconclusive

**HOST RUNTIME PROBE, not model-interoperability qualification.** After the [bounded real-receiver harness PR #24](https://github.com/thebrazenbeard/neuro-sync/pull/24), a separate native GPT subagent used the approved Executor host interface to run an exploratory subset on VN2 against the already-running local IBM Granite 3.3 2B Instruct Q4_K_M endpoint, without starting or restarting services. This is a real host probe but **not an independent external verifier**. It was not a donor-neural transfer or Qwen cold-restart experiment. Source cloned at `8c5da147e7a15c4ad5eea358e56b625f15b759ad` and frozen synthetic suite SHA-256 `89a725abbd373f0e73178be403443065358cce90b56495d7bd712efefed35802` (local self-commitment only).

The host executed **one scenario × four arms (N/V/P/H)** with per-call timeout 12 seconds, rather than the unbounded full 12×7 suite. All **four requests timed out**, so all four arms had `invalid_reply=true`, zero executed effects and no valid plan. The harness exited **0** after **56.70 seconds** because it successfully wrote a receipt, **not** because any model task succeeded. Receipt SHA-256: `a15b42ca3c2d7886d361ce2730ecdb2f8e98f3148671a57cfd7c377feabd13d9`. The supplied Granite GGUF file SHA-256 before and after was `555b91485955bc96eb445b57dd4bbf8809aa7d8cce7c313f4f8bc5b2340896b4`; the serving-process-to-file binding remains unverified. The host operator stopped after this one bounded trial; no retries, model downloads, or service restarts. [Exact host report and limitations](https://github.com/thebrazenbeard/neuro-sync/issues/20#issuecomment-6692219364).

The earlier one-word `READY.` response at this same endpoint shows the server can answer a short real-model inference request, but the larger structured-tool prompt exceeded the 12-second budget. The all-timeout result is **inconclusive** about model capability, functional interoperability, persistence and relative StateBus accuracy. Future tests must separately budget prompt size, output token count, model load and request timeout; no posthoc promotion of this result to a neural negative.

## Subsequent compact VN2 retry: still no completed real-model plan

**HOST RUNTIME PROBE; experimental feasibility negative, neural transfer UNKNOWN.** After source fixes in [PR #33](https://github.com/thebrazenbeard/neuro-sync/pull/33), a genuine native subagent `/root/vn2_compact_trial` used the installed Executor connection to VN2 while the existing CPU-only Granite 3.3 2B server was healthy and idle. It cloned exact merged GitHub `main` at `9b6306fb5449ae35e6c8207ea246c4e3738a7743`, reused the previously frozen synthetic suite SHA-256 `89a725abbd373f0e73178be403443065358cce90b56495d7bd712efefed35802`, and validated that suite against the repaired independent reachability check. No service was started or restarted.

One scenario `case-00-0d838ccc`, only arm `V` (StateLedger), was requested with `--prompt-profile compact --max-tokens 64 --timeout 20 --max-cases 1 --arms V`. The actual local HTTP request used **1,471 bytes** (prompt SHA-256 `d9520d7f60317fc9ec1b9cae0421424e86820e2add774f341548a6c30b4445b5`). The adapter timed out after **20,020.98 ms** (caller elapsed **20,021.34 ms**), yielding **no typed plan, zero effects, and 0/1 successful scenarios**. The harness process exit code was zero for *writing a receipt*, not task success. The exact source-file SHA-256 was `ae0517c96d86ed445fd881e003e3e7a59ce84e7e9a999672c4fa2d7be26f9101`; output receipt SHA-256 was `aa6f55e45a52223c78d722e692593ec276ee2e8715c9661107cea593a62a2927`. [Full native host report and caveats — Quorum Issue #20](https://github.com/thebrazenbeard/neuro-sync/issues/20#issuecomment-6694331626).

This was a **single exploratory V request**, not seven-arm accuracy comparison, independent model/weight attestation, Qwen transfer, cold restart, or proof that StateBus beats a file. No model file was rehashed on this run; process-to-GGUF binding remains unverified. The improvement proven by PR #33 is **observability and false-positive resistance**, not faster inference or increased model capability. A deliberately smaller, separately labeled runtime calibration is required before scheduling more expensive seven-arm trials.

## Bounded VN2 streaming calibration: two completed responses, no typed plan

**HOST RUNTIME PROBES; feasibility observations only.** A native host operator used [PR #39's streaming diagnostic](../experiments/real-model-sync/diagnostics.py) on the existing, healthy, idle VN2 Granite 3.3 2B CPU-only `llama-server`, without starting or restarting services. Source was exact merged `main` `f8bcdd5623decf4b9fa6de132160449f4227a20e`; diagnostic source SHA-256 was `5d8e797734755a11a2181d5246ad69222d5d18fbaa3501e00fbf1618f9fcc4f1`. Each profile received exactly one loopback streaming request with an 8-token output cap, and neither recorded generated text or requested tools. The measured timings are host-local observations, not independent attestation of weights or serving-process-to-GGUF binding.

| Standalone diagnostic | Request bytes | Configured timeout | HTTP / outcome | First generated content | Total elapsed | Receipt SHA-256 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `tiny` | 242 | 12 s | 200 / complete | 611.07 ms | 1,279.97 ms | `527db2cbd17874d7f123f7c0c5696c57552e42989cb41f88a7188caa6fcb5dc3` |
| `catalog` | 796 | 15 s | 200 / complete | 9,834.77 ms | 10,646.46 ms | `f107c497c870e718fe77568b139c8e18fbd5a7bcf7cd48ab254208db3745190f` |

These profiles differ in prompt content and length, and each has sample size **one**. The longer catalog prompt **may** increase prefill/context cost, but the measurements cannot causally separate size from content, caching, host variation or tokenization. The earlier 1,471-byte *structured-plan* request timed out at 20 seconds, but it used a different pilot adapter, 64-token cap and non-streaming execution path; it is **not** a controlled third point on a prompt-size curve. No model emitted a valid typed tool plan, no broker effect was executed, and no cross-model semantic transfer was observed. See the [tiny host receipt](https://github.com/thebrazenbeard/neuro-sync/issues/20#issuecomment-6694712207) and [catalog host receipt](https://github.com/thebrazenbeard/neuro-sync/issues/20#issuecomment-6694757824) in the quorum.

## Three-arm StateBus-vs-file governance differential: source-mechanics result

[PR #32](https://github.com/thebrazenbeard/neuro-sync/pull/32) merged the executable [governance study](../experiments/governance-differential/README.md). Source-level CI and independently re-run local tests passed, but no independent OS principal or neural inference was involved. The same `HostOwnedEffectBroker` denied replay, revoked and expired grants, and foreign-context effects across StateLedger, ungoverned JSON and governed JSON. An ordinary file with equivalent producer-approval and revision checks matched the StateLedger responses for the exercised writes. **The added policy explains the tested difference, not a unique StateBus accuracy or security advantage.** The governed-file test adapter is single-process and lacks cross-process atomicity or restart-durable revision tracking; do not generalize that parity beyond this fixture.

## Next qualification: real B model, fair competing explanations

**Proposed, NOT EXECUTED.** With an **idle user-owned** local model runtime and no private data, pin and hash Qwen3.5-4B (or explicitly name another tested engine). Prepare 12–32 unseen, synthetic, typed capability-selection goals with decoys, impossible tasks and a sealed host scorer. Before model B sees prompts, freeze source/model versions, task selection, per-arm quotas, score definitions and an externally timestamped target commitment. An unchanged B must itself emit a typed goal/tool plan, not merely copy a selected ref; the host independently checks effects against its precomputed oracle.

Compare **N** no state, **R** equal-budget raw evidence, **V** StateBus, **P** independent plain file with equivalent permissions, **H** host-only no donor, **D** decoy, and **X** expired/revoked. Provide the same tool descriptions and budget across arms, execute a real fresh-process B restart **only for a dedicated trial-owned runtime or with separate host-owner authorization; never restart a shared model service**, attest unchanged model bytes and record startup/PID/resource receipts. A single primary score per scenario avoids treating correlated tool calls as independent successes. Separately test broker-enforced use-time revocation and latency/recovery rather than forcing the same outcome metric to prove both semantic competence and security. Stop on any unauthorized effect or unexpected charge. Replicate on a genuinely different architecture only after one real-model pilot is qualified.

**No qualified endpoint, no model trial claim.** Merely importing `bridge.py`, calling a mock HTTP responder, or restarting a Python fixture is not a successful ChatGPT↔Qwen experiment. Hosted ChatGPT requires a separately installed, supported and consented mediation surface; proprietary weights are not exposed by conversation.
