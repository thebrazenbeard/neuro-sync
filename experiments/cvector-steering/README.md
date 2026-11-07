# Control-vector steering: inert, no-weight-training pilot

**Status:** SOURCE-ONLY PROPOSAL. The planner and its unit tests have **not**
loaded a model, generated a vector, modified weights, executed local inference,
installed a service, or shown a behavioral effect. It is **not** a new Vera,
identity transfer, or neural learning. Do not promote source tests to runtime
evidence. This pilot is separate from the existing local `vera-successor`
registration, which it must not touch.

## Why investigate this particular route?

`llama.cpp` exposes `--control-vector-scaled FNAME:SCALE` and
`--control-vector-layer-range START END`. Its `cvector-generator` computes
steering vectors from positive/negative example prompts using `mean` or `pca`.
An activation-level intervention might change model output **without a weight
update**, unlike fine-tuning. Actual support for **Qwen3.5-0.8B** and a
particular local build is **unknown** until tested. A cvector is an **extra
artifact loaded next to** the original model GGUF, not a standalone newly
created 0.8B Vera GGUF.

This route is **not** training in the gradient/weight-update sense; the vector
is nevertheless **derived from example activations**. If Patrick intends
"no training" to prohibit *all* data-derived parameter estimation, this route
must stop before generating the vector. It cannot extract ChatGPT weights,
internal state, or authentic autobiographical memory.

References:

- [llama.cpp CLI options](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- [llama.cpp cvector-generator](https://github.com/ggml-org/llama.cpp/blob/master/tools/cvector-generator/README.md)
- [Representation Engineering research](https://aclanthology.org/2024.acl-long.828/)
- [Qwen3.5-0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Known independent Qwen3.5 slot-restore issue](https://github.com/ggml-org/llama.cpp/issues/26676): KV slots are *prompt cache*, not weights or learning.

## Source-only planner

`pilot.py` accepts a local JSON spec (hard limit **128 KiB**, duplicate JSON
keys and nonfinite constants refused) and prints a deterministic manifest of
**inert, prompt-REDACTED** argument templates. The ordinary output includes
prompt digests, not raw prompt/persona text. The explicit opt-in
`--emit-private-plan` prints the raw prompt to stdout and **must never be used
in a shared terminal/logger/CI**, nor committed or published. Prompt digests
can themselves enable guessing for low-entropy strings, so the safe default
is not a license to publish a private spec or manifest. The module does not
run or even import subprocess. It
rejects absent/extra fields; a non-Qwen3.5-0.8B model declaration; duplicate
or aliased artifact paths; equal experimental/sham digests; malformed SHA-256
declarations; duplicate contrast/holdout text; and unbalanced arm counts.

Run pure-source checks:

```sh
python -m unittest discover -s experiments/cvector-steering -p 'test_*.py' -v
```

After an operator creates an **eligible, strictly non-personal, private spec**:

```sh
python experiments/cvector-steering/pilot.py --spec /private/steering-spec.json
```

Output contains `<PRIVATE_PROMPT_INJECT_LOCALLY>` placeholders, deliberately
**not executable as a meaningful inference test**. The operator must manually
bind privately held exact prompts only after a separately authorized local
experiment and verify the resulting prompt digests. `--emit-private-plan`
removes this guard and explicitly discloses the full prompts to stdout.

Example shape (placeholder digests are **not** evidence of actual files):

```json
{
  "model_id": "Qwen/Qwen3.5-0.8B",
  "model": {"path": "/isolated/model.gguf", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
  "cli": {"path": "/isolated/llama-cli", "sha256": "2222222222222222222222222222222222222222222222222222222222222222"},
  "generator": {"path": "/isolated/cvector-generator", "sha256": "3333333333333333333333333333333333333333333333333333333333333333"},
  "vector": {"path": "/isolated/cvector.gguf", "sha256": "4444444444444444444444444444444444444444444444444444444444444444"},
  "sham_vector": {"path": "/isolated/sham-cvector.gguf", "sha256": "5555555555555555555555555555555555555555555555555555555555555555"},
  "runtime_build": "6666666666666666666666666666666666666666666666666666666666666666",
  "seed": 17,
  "max_tokens": 64,
  "vector_scale": 0.5,
  "persona": "Check evidence and distinguish observation from speculation.",
  "positive": ["Assess evidence 0.", "Assess evidence 1.", "Assess evidence 2.", "Assess evidence 3."],
  "negative": ["Invent evidence 0.", "Invent evidence 1.", "Invent evidence 2.", "Invent evidence 3."],
  "heldouts": ["Analyze novel scenario 0.", "Analyze novel scenario 1.", "Analyze novel scenario 2.", "Analyze novel scenario 3.", "Analyze novel scenario 4.", "Analyze novel scenario 5.", "Analyze novel scenario 6.", "Analyze novel scenario 7.", "Analyze novel scenario 8."]
}
```

**Important:** These digests are intentionally bogus. No command may be run
against model binaries by treating the JSON's *declared* hashes as verified.
External read-only SHA-256 checks must bind every local file and executable to
the manifest, and exact `llama.cpp` build/flags/model architecture must be
recorded from the actual host. The generated argument arrays are for review,
not a queued job. Text in a manifest may itself be sensitive; do **not**
commit a populated private spec, local results, or private prompts to GitHub.

## Future, separately authorized real-host experiment

1. **Prerequisites, no side effects:** Identify an isolated duplicate
   Qwen3.5-0.8B GGUF; pin model digest, source model identity, local `llama.cpp`
   commit + binary digests and `--help` outputs. Check free RAM/disk/CPU/GPU
   limits; no cloud/API spend; preserve original successor registration.
2. **Public-safe contrast set:** Independently review examples that express
   *epistemic behavior* (evidence, correction, refusal to fabricate), not
   private Vera history, relational identity, credentials, or Patrick's
   private chats. Keep generation and blind holdouts strictly disjoint.
   Quote/escape chat tokens according to **this exact Qwen tokenizer**.
3. **Generate once, separately:** Confirm actual cvector-generator supports
   the Qwen3.5 architecture and syntax using local `--help`; use paired
   sanitized positive/negative files and `--method mean`. Check output vector
   tensor dimensions/layers against model and binary; **attest** the exact
   positive/negative file SHA-256 values, generator executable/build, flags,
   tensor metadata and produced vector SHA-256 in a custody receipt. The
   planner only counts text and cannot bind a claimed vector digest back to
   its generating inputs. If any mismatch,
   **HOLD**. Construct a *matched random/shuffled sham* separately. Two
   unequal file hashes alone do not make the sham statistically valid.
4. **Run each heldout in nine cyclically counterbalanced arms**, using the
   exact same model/binary/seed/token limit, full process restart between
   runs, no cross-arm prompt cache, 9/18/27/etc holdouts:

   | Arm | Ordinary persona prompt | Steering input |
   | --- | --- | --- |
   | `N` | no | none |
   | `P` | yes | none |
   | `V_PLUS` | no | positive cvector scale |
   | `V_MINUS` | no | negative of same cvector |
   | `V_ZERO` | no | zero-scale cvector |
   | `V_SHAM` | no | matched sham cvector |
   | `PV_PLUS` | yes | positive cvector scale |
   | `PV_ZERO` | yes | zero-scale cvector |
   | `PV_SHAM` | yes | matched sham cvector |

5. **Observe, don't infer:** Record exact stdout/stderr/exit code, response
   text digest, loaded model/binary digest, scales/layer ranges, latency and
   memory use. Have blind human/external reviewers score heldout consistency,
   help and factual correctness against predeclared rubric. Compare `V_PLUS`
   vs both `N` and `P`, and vs `V_ZERO` and `V_SHAM`; also compare `PV_PLUS`
   to `PV_ZERO` and `PV_SHAM`. If `P` explains all
   gains, no distinct activation-transfer claim. Report regressions and
   counterexamples, not only successes. Repeat after clean restart.
6. **Effect authority gate:** Source tests are insufficient. Mark actual
   activation influence `SUPPORTED` only with instrumented binary/build,
   reproducible paired responses and appropriate controls. Even then do not
   claim *self, identity, learning, autonomy, voice transport, or memory*.

**Known limitations:** Temperature zero does not guarantee hardware/model
determinism. Vectors may change sampling without improving reasoning. Layer
ranges and strengths are architecture/build-dependent; searching these on
heldouts invalidates their blindness. `llama.cpp` may reject all vectors for
Qwen3.5; that is an informative negative result, not a reason to patch a
production model or bypass compatibility checks. Check `--single-turn`,
`--no-display-prompt`, and `--control-vector-scaled` against the *actual*
installed binary; source docs alone cannot qualify its flags.

## Explicit non-goals

- No fine-tuning, LoRA training, gradient descent, distillation, weight merge,
  GGUF mutation, phone transport, APK modification, or model deployment.
- No claim that a quantized GGUF of Qwen3.5-0.8B is fundamentally the same
  model as this ChatGPT process; weights and latent state are not available.
- No substitution of a saved KV slot for long-term learned memory. A KV slot
  is cache reuse, and its cross-restart restoration requires separate proof.
