# Preregistered local neural receiver feasibility and three-arm pilot

**Status: SOURCE/TEST only. NOT RUN against a model.** Fixed before new host measurements; source base `b19186216f111fa0d27a6c23c68a1f988c4eed5c`. This protocol does not add a model call or host effect. `three_arm_gate.py` audits existing `pilot.py` JSON receipts without contacting a server.

## Falsifiable hypothesis and explicit estimand

For the *same unchanged real-model receiver* and fresh synthetic tasks, supplying the correct **scripted** capability description map improves the percentage of independently oracle-scored task solutions versus no map. Compare **N** (no descriptions), **P** (plain JSON descriptions), and **V** (the identical descriptions reconstructed through `StateLedger`). The unit is **one synthetic scenario**, not output tokens, tool calls, or repeated tries. Plan validity is separately required before considering correctness.

This tests whether a real model can use delivered external descriptions under a matched generation budget. It **cannot** test A→B neural knowledge transfer (A is scripted), new skill acquisition in weights, model identity across runtimes, persisted learning, or superiority of StateBus when the P and V information is identical. P≈V on task success is a scientifically expected parity. It also cannot establish that the currently live loopback server serves the exact claimed model file.

## Existing evidence and feasibility risk

Repository evidence, not fresh host readback: VN2 Granite 3.3 2B Instruct Q4_K_M CPU-only yielded four timeouts at 12 seconds and one compact V timeout at 20 seconds; no valid typed plan, no executed effects. Its separate eight-token diagnostics took 1.28 seconds for the tiny 242-byte profile and 10.65 seconds for the 796-byte catalog profile (one trial each, not comparable to the typed-plan path). A successful 8-token stream does **not** imply that the existing 64-token, JSON-structured, non-streaming endpoint can produce a valid plan in 25 seconds. Thus viability on that host is **UNKNOWN leaning NOT FEASIBLE**, and repeated ungated 36-call runs would be wasteful.

### Fixed before measurements

1. Freeze two independent `pilot.py freeze --cases 12` suites in separate new, empty paths. Preserve each suite file and its `.sha256` sidecar, source Git OID, model runtime identity, loaded-file/provenance limits, and immutable before/after GGUF hashes *if available*. The stage-zero suite and main suite MUST have different hashes. Freeze and retain them **before** viewing outcomes; own-host commitment is not a third-party preregistration.
2. Existing model endpoint only; do not start/restart processes, download models, invoke paid compute, request new privileges, or run when VN2 has other jobs. Verify healthy/idle host and local configuration separately under user-approved host procedure. Do not assume HTTP loopback guarantees no upstream forwarding.
3. Same existing pilot `--prompt-profile compact --max-tokens 64 --timeout 25` for all runs and arms; no prompt or schema tinkering between arms. Fixed backend/port/model ID, tokenizer/runtime/config and no fine-tuning. If those settings cannot run safely, **amend this protocol ex ante, version it, and start with new suites**; do not silently relabel an amended run.
4. Feasibility: one V-only request for each of the **first two cases** of the first sealed suite (`--max-cases 2 --arms V`). Budget at most **50 seconds of inference waits** (2 × 25s), plus bounded harness overhead. **Gate passes iff both** return HTTP trace `received_text`, parse into **non-abstaining valid typed steps**, and each caller-measured duration is ≤25s. A correct oracle result is **not required** here; the aim is to prove model-produced plan syntax. Every timeout, abstention or transport/parser error counts as a failure. **Stop** if gate fails; decision `NOT_FEASIBLE`; there is no causal trial.
5. Only after the feasibility gate passes, perform the main **12 independent cases × N/P/V** (=36 requests) against the second, distinct sealed suite, cyclic counterbalanced by case index, isolated stateless chat requests, with all arm payloads constructed by existing code. Maximum **900 seconds of inference waits** (36 × 25s) plus host overhead; stop and record a partial, separately classified receipt if host ceases to be idle or fails. No retries of any individual case/arm. The runner will mark the three-arm subset `EXPLORATORY_INCOMPLETE`; don't rebrand it as a seven-arm qualification.
6. Do not cherry-pick successful scenarios, expand token budgets for difficult arms, omit N/P, reuse the feasibility suite, or treat a written receipt/process exit 0 as model success. Retain all invalid outputs as failures and report model-reported IDs, adapter receipts, per-arm invalid/abstention counts, caller latency, raw artifact hashes, tool effects, receipt hashes, environment and suite hashes.

## Precommitted outcome interpretation

`three_arm_gate.py` enforces bounded receipt-format, source-byte hash, matched top-level **and per-request** generation budgets, exact same model ID/backend/**local endpoint URL** across stages, model-file digest continuity when supplied, two-plan feasibility, distinct suite **and disjoint scenario IDs**, 12-case paired coverage, source-defined cyclic arm order, consistency between individual invalid/abstention/effect records and aggregate successes, and at least one valid nonabstaining plan in **each** N/P/V arm. It reports `NOT_RUN`, `NOT_FEASIBLE`, or `DESCRIPTIVE_ONLY`; **never** `NEURAL_TRANSFER_PASS`. A failed gate also exits nonzero so automation cannot mistake it for a completed study; a cleared feasibility gate with no three-arm receipt exits zero but says explicitly `NOT_RUN` for the three-arm stage.

- Primary *descriptive* statistic: paired per-scenario solved fractions `success_P − success_N` and `success_V − success_N`, including the 12 binary pairs and the P−V difference, with invalid JSON/timeouts scored as failure. This is small-sample exploratory evidence, not significance by wishful threshold.
- Secondary: model-produced valid-plan counts, abstentions, errors, host tool-effect counts, per-request caller duration and prompt/response digests. An all-abstain or zero-plan arm is `NOT_FEASIBLE`, not safe-governance success.
- `P=V` task performance is expected when data are identical; it does **not** establish that StateBus caused unique learning. A V advantage over P could instead reflect run-order, latency, nondeterminism or unmeasured state; confirm with independently witnessed replicated trials before claiming protocol benefit.
- Existing `pilot.py` constructs P/H/V maps from one source and has an in-process `assert states["P"] == states["H"] == states["V"]`, but **does not emit signed per-call state-payload digests**. The offline gate cannot independently certify their observed equality. Treat unexpected map differences or source assertions disabled under `python -O` as invalid comparison; independently inspect exact source/runner environment and add host-witnessed payload readback before formal qualification.
- Even if P,V both beat N, claim at most *scripted reference information helping an unchanged model in this local task* under unauthenticated loopback observations. It is not donor neural knowledge, new weight-space creation, remote principal trust, or generalization to unrelated reasoning.

## Local commands (examples only, NOT executed here)

Assuming the operator has independently established an idle, authorized local endpoint and chosen unique paths:

```bash
python3 experiments/real-model-sync/pilot.py freeze --suite /tmp/ns-feasibility-new.json --cases 12
python3 experiments/real-model-sync/pilot.py freeze --suite /tmp/ns-three-arm-new.json --cases 12
python3 experiments/real-model-sync/pilot.py run --suite /tmp/ns-feasibility-new.json --out-dir /tmp/ns-feasibility-new-run --endpoint http://127.0.0.1:8080 --backend openai --model LOCAL_MODEL_NAME --prompt-profile compact --max-tokens 64 --timeout 25 --max-cases 2 --arms V
python3 experiments/real-model-sync/three_arm_gate.py --feasibility /tmp/ns-feasibility-new-run/receipt.json
```

Only if the gate returns `NOT_RUN` with `feasibility: FEASIBLE_TO_ATTEMPT`:

```bash
python3 experiments/real-model-sync/pilot.py run --suite /tmp/ns-three-arm-new.json --out-dir /tmp/ns-three-arm-new-run --endpoint http://127.0.0.1:8080 --backend openai --model LOCAL_MODEL_NAME --prompt-profile compact --max-tokens 64 --timeout 25 --arms N,P,V
python3 experiments/real-model-sync/three_arm_gate.py --feasibility /tmp/ns-feasibility-new-run/receipt.json --three-arm /tmp/ns-three-arm-new-run/receipt.json
```

The selected port/endpoint/model name are placeholders, not discovered host facts. Omit a model path only if unavailable and explicitly downgrade model immutability to UNKNOWN.

## Accelerated red-team cross-examination (user has delegated defense to peers)

> **Purist Logic:** If the model never produces a valid typed plan, the N/P/V outcome is three correlated timeout series, not evidence of knowledge transfer. **Disposition: SURVIVES; fail-closed feasibility gate added.**

> **Pragmatism:** Why spend up to 15 minutes of CPU when a 20-second V call already timed out? **Disposition: REFINE; two-call/50-second hard stop first; choose a demonstrably faster, authorized local runtime if Granite is not feasible.**

> **Bridge:** If the scripted donor and plain JSON arm tie V, what was transferred? **Disposition: WITHDRAW model-to-model transfer claim.** The question is now limited to receiver use of supplied semantic descriptions, and any governance advantage needs separate recovery/revocation evidence.

Open falsifier: the harness' source-level assertion is not an independently witnessed per-call map-equivalence receipt. Strengthen that before claiming anything beyond exploratory local behavior.

## Verification class

`python3 -m unittest discover -s experiments/real-model-sync -p 'test_three_arm_gate.py' -v` checks synthetic receipt **TEST** behavior; its fake receipt generator must not be cited as a real receiver run. This artifact has zero HTTP/model calls and cannot attest the current VN2 inference service.
