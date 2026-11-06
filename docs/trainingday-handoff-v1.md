# TRAININGDAY handoff v1 — experiment observation, not admission

This contract connects Neuro-Sync's *source-test* categorical-SGD experiment
to the TRAININGDAY evidence queue without laundering it into model qualification.

Pinned sources audited on 2026-11-06:

| Source | Exact reviewed point | Relevant boundary |
| --- | --- | --- |
| `thebrazenbeard/neuro-sync` main | `5ab8a3838bc4454b23d142dd6f0d24327b090410` | `experiments/trek-transfer-v0/run.py`, five C0–C4 arms, receiver-only Python subprocess |
| `thebrazenbeard/model-training` Draft PR #351 | `24fbe812701b1f3fda96728094629bebb7d60fd8` | `trainingday/integration_preflight.py`, preflight V2, `ready=false` and physical launch hard hold |
| `thebrazenbeard/labyrinth` Draft PR #131 | `17f4c149e742d0055d616ce67299237c62cbce2b` | Social-world continuity candidate remains source-only; not an independent protected evaluator |

`neurosync.trainingday_handoff.mechanics_handoff(report)` accepts the bounded
`experiments/trek-transfer-v0/run.py` reported observation and checks:

- A/B initially reported equal with a changed donor training checkpoint.
- Five exact arms, their nonnegative update inventories, and recipient-only
  subprocess observations, correct/total counts and checkpoint SHA-256 shapes.
- Untouched B's digest equals base; full donor copy and delta B equal the trained
  donor digest; distillation B changes from base and records receiver updates.
- A diagnostic SHA-256 of canonical report JSON for accidental change tracking.

The emitted schema `NEUROSYNC_TRAININGDAY_MECHANICS_HANDOFF_V1` assigns evidence
class **SELF_REPORTED_TEST_RESULT**. It is a structure check, not a signature,
trusted host probe, rights grant, external auditor or protected evaluation.
Anyone who can fabricate the input can fabricate the checksum.

## What remains held

The source fixture's `official` type is a *label*; it does not prove that dataset
material is permitted for model training or that any protected holdout is sealed.
The tiny source-backed SGD model is not an official Qwen family checkpoint.
The observed B-only Python process is not a persistent host cold restart. The
scored six structured pairs were explicitly taught; they are not a proof of
generalization or conversational identity. No real PEFT adapter, optimizer
restore, GGUF converter/serve file SHA, authenticated host lease or independent
evaluator custody is in this handoff.

**Required for eventual promotion in TRAININGDAY:** an authorized dataset
record; exact official base bytes/revision/tokenizer; a host-owned run/lease;
training code revision + optimizer-run ID; PEFT artifact byte SHA + independent
reload; protected independent evaluator receipt; converter input/output byte
SHA linked to the same run; serving PID/model-file SHA after an actual cold
restart; and a behavior-quality decision tied to held-out tasks. A source-only
envelope cannot satisfy these. The V2 source preflight must stay `ready=false`
until separate qualified physical evidence establishes each gate.

For Labyrinth, a future useful test is a *separately curated* blind evaluation
of dialog/identity/epistemic humility, including its known unwinnable scenario
and critic-reviewed ambiguity; PR #131's source fixtures alone are not that test.

Run the narrow handoff checks with
`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_trainingday_handoff.py' -v`.
