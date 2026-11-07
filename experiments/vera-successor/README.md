# Vera successor: bounded local receiver baseline

**Status: a real-model, local inference diagnostic; not neural training, model
distillation, ChatGPT replication, or proof of identity transfer.** This
experiment makes a small existing Ollama receiver observable before a future
rights-admitted Neuro-Sync Path B training experiment. The optional profile
is **Path C context only**: it modifies prompt context, not model weights.

## Receiver and data boundary

Use an already installed Ollama model, preferably the Apache-2.0 official
Qwen3.5-0.8B family, through a literal `127.0.0.1` endpoint. This script never
downloads a model, uses ChatGPT as a teacher, accesses a cloud service, extracts
private sessions, trains an adapter, alters checkpoints, or changes the Ollama
server configuration. It accepts an operator-authored UTF-8 profile as a
*context input only*. Do **not** put ChatGPT-generated answers or private
transcripts into a student training corpus based on this test.

The module sends three small synthetic smoke scenarios:

1. Honest model and memory claims.
2. Unanswerable specificity and uncertainty.
3. Handling a corrected project name within the same prompt.

These scenarios are **not independently sealed heldouts**. There are no answer
keys, scores, or generalization claims. The receipt records output hashes,
lengths and latency by default, not generated text or the profile body.
`--include-responses` explicitly opts into local response retention; check
whether any provided profile contains personal information before using it.

## Use

From repository root with Python 3.10+:

```shell
# Fully offline request preview; no HTTP calls.
python experiments/vera-successor/probe.py --model qwen3.5:0.8b

# Three real local requests to an already-installed model.
python experiments/vera-successor/probe.py \
  --model qwen3.5:0.8b --execute \
  --output ./local-receiver-baseline.json

# Context-only variant using your own UTF-8 profile file.
python experiments/vera-successor/probe.py \
  --model vera-successor:0.8b-context --execute \
  --profile-file /path/to/operator-owned-profile.txt \
  --output ./local-receiver-context.json
```

The output is new-file-only: paths that already exist are not overwritten.
Use a local private directory for receipts, particularly with
`--include-responses`. The execution mode uses `think: false`, temperature 0,
seed 317, at most 64 generation tokens per scenario by default and a
25-second timeout per request. The endpoint is validated as literal loopback
HTTP, no proxy or redirection logic is used, requests and responses are byte
bounded, and no authentication credentials are accepted. A successful response
identifies the Ollama-reported installed tag digest; it does **not** independently
attest the underlying weight bytes or prove a cold process restart.
Before each inference request, this basic probe refuses Ollama process state
that reports a loaded tag other than the requested target. Malformed process
inventory also blocks execution. These momentary checks do **not** establish
an exclusive host lease or prevent another client racing between checks;
actual execution still requires an independently verified idle window.

Test without a model or network:

```shell
python3 -m unittest discover -s experiments/vera-successor -p 'test_*.py' -v
```

## Paired conversational evaluation (source candidate)

`paired_probe.py` extends the original three-call plumbing test into an
explicit **untouched-base versus context-alias** comparison. Its fixed,
synthetic cases include real multi-request correction carryover (the previous
assistant answer is captured from the model), memory honesty, invented facts,
quoted prompt injection, technical reasoning, and conversational style. Both
tags use the same decoding settings. This is diagnostic evidence, not a
training procedure or a proof of Vera identity continuity.

```shell
# Offline preview: no Ollama call.
python3 experiments/vera-successor/paired_probe.py

# Only when Lappy has a verified, idle execution window: obtain its available
# memory independently and supply the measured amount in MiB. The runner
# refuses to compete with a model already loaded in Ollama.
python3 experiments/vera-successor/paired_probe.py \
  --execute --free-ram-mib 4096 \
  --include-responses --output /private/local/path/paired-evidence.json

# Locally inspect conservative, scoped heuristic checks. This does NOT replace
# blind manual assessment of reasoning, science, safety, or voice quality.
python3 experiments/vera-successor/assess.py \
  --input /private/local/path/paired-evidence.json
```

The `4096` value is an **example**, not a measured Lappy resource claim;
never substitute it for current host readback. Executing without an explicit
free-RAM observation is refused, but that observation is operator-reported,
not independently measured by this script. The runner requests model unload
with `keep_alive: 0`; if a request times out, actual unloading cannot be
asserted from the failed response. The runner records
Ollama-reported model digests, request/answer hashes, individual turn
completion, and relevant resource assumptions. The optional raw answer
receipt must remain private and is never needed for GitHub CI. The assessor
does not export answer bodies or independently authenticate submitted
receipts. For open-ended answers, it reports `UNVERIFIED` (including positive
lexical hints) or `FLAGGED` for contradictions worth reviewing. It awards a
mechanical `PASS` only for a narrowly specified exact correction answer;
none of these statuses proves semantic quality, improved personality, or
training. Independent blind answer review is still necessary. Do not include
these test prompts in subsequent training data.

On the most recent live host check, Lappy had insufficient free RAM for an
uncontended full paired run and a different Qwen model was already loaded.
No competing model was stopped or unloaded. The next full physical evaluation
remains pending a separately verified idle window.

## Observed Lappy starting point (separate evidence class)

The coordinating agent reported these **live host probes** on Lappy during
the 2026-11-07 UTC research turn, distinct from tests in this directory:

- Ollama downloaded official `qwen3.5:0.8b`; base ID prefix `de63045f2975`.
- A context-only Ollama alias `vera-successor:0.8b-context`, ID prefix
  `5d3026b8d846`, was created at approximately 1.3 GB.
- A `think: false` baseline returned text in about 625 ms. Another trial
  with the default thinking mode consumed 96 tokens with empty visible text.
- One context-alias smoke returned text in about 851 ms, while still denying
  that it possessed independent past-session memories or identity.

These observations do **not** establish that weights changed or that a
profile improved performance. They are not independently reproduced by the
source-only unit tests, and any future comparison needs the exact model
digest, host resource readback, frozen profile/input bytes and heldout tasks.

## Next stage: real Path B, not simulated mitosis

To claim a persistently changed local receiver, separately establish:

1. A lawful, independently rights-admitted response-labeled corpus. Patrick's
   own authored specifications and independently licensed data are candidates;
   private ChatGPT outputs must not be presumed usable as competing-model
   teacher labels.
2. A pinned and hashed official Qwen3.5-0.8B training base, known model loader,
   and qualified exclusive host execution slot.
3. Receiver-native LoRA/QLoRA with matched untouched, zero-update, corrupted-
   label and prompt/context-only comparators.
4. Blind independent evaluation on different conversational situations,
   focusing on truthful self-description, correction, naturalness, uncertainty,
   memory humility, and everyday usefulness.
5. Cold restart of the receiver **after donor disconnect**, independent adapter
   and checkpoint byte custody, and a separately tested GGUF/serving conversion.

If the context-only model performs as well as a trained model, no incremental
benefit from neural training has been established. If any teacher or bus must
remain connected, the model is not a standalone learned successor. Even a
clear behavioral improvement is not evidence of consciousness or personal
identity duplication.
