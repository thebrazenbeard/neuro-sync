# Mini-Vera semantic compiler — experimental Path C

This is a **zero-weight, source-only prototype** of the corrected idea: derive
an executable *candidate semantic kernel* from explicit public contracts, then
use Qwen3.5 0.8B only as a language decoder. There is no LoRA, SGD, synthetic
finetuning, private-chat transfer, production deployment, or host inference.

## What the compiler actually does

1. Accepts explicitly selected `PUBLIC_APPROVED` UTF-8 source bytes with
   caller-stated exact Git commits and checked by SHA-256. It does not fetch
   repositories or authenticate commit-to-blob bindings. The output labels
   source pins `CALLER_ASSERTED_PIN_UNVERIFIED` pending trusted fetcher proof.
2. Compiles only caller-authored, typed `SOURCE_ASSERTION` claims whose exact
   quoted evidence exists in those bytes and contains the claim's literal
   subject/relation/object tokens. This is only **lexical support**, not a
   valid proof of semantic entailment. Independent meaning review is still
   required. Arbitrary prose does not become an executable rule or permission.
3. Builds a provenance-bearing graph and reports conflicting objects for the
   same subject/predicate without silently choosing a winner.
4. Builds finite procedural dependency plans. Every `HOST_EFFECT` remains
   blocked pending an independent, authenticated host authority decision.
5. Creates deterministic, bounded evidence context for a decoder; rejects
   arbitrary effect requests even when the evidence contains prompt injection.
   In-memory compiled-kernel mutation is detected by a local self-hash; that
   hash is **not** independent integrity attestation.
6. Persists a local hash-chained observation trace for restart tests, explicitly
   labeled `LOCAL_UNWITNESSED`. Same-user file rewrite is not prevented.

Run Python 3.10+ without additional dependencies:

```bash
cd smallvera_semantic_compiler_c
python -m unittest -q test_semantic_kernel
```

## Distinction from other approaches

This path compiles **external program semantics**. It does not claim
Qwen's weights have changed or that the kernel is Vera's current, installed
native control/memory. By contrast, real neural distillation would modify and
test a recipient-native adapter (now explicitly out of scope).

### Porting into the real ecosystem

- Use `vera-mono` as runtime composition root, not a new competing authority.
- Use Semantic Atlas provenance roles and DeepMemory `EVIDENCE_SEARCH` boundaries.
- Use `pre-active` resident scheduling and host-owned effect gates only after
  source, install, authority, and runtime verification.
- Use Neuro-Sync Path C only if its transport/qualification adds value; its
  scripted research found plain JSON tied StateBus on task correctness.
- Promote source assertions to live policies **only** through the existing
  independently verified control-owner mechanism. This compiler never does so.
- Compare the actual 0.8B decoder plus compiled context against exact-token
  ordinary-document retrieval and equal-budget prompt baselines. Test blind
  transfer tasks, contradiction handling, tool-denial, stale updates, restart,
  and memory revocation before claiming a functional mini-Vera successor.

All repository references above were inspected live through GitHub README and
architecture sources on 2026-11-07. This local code is not an installed service.
