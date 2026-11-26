# Qwen selective-association falsifier (research source only)

**Status on 2026-11-17:** source runner, synthetic experiment generator,
source tests and a real-tokenizer alignment probe. **No scored GPU training
or real-model evaluation of this selective protocol was performed.**
Lappy RTX 3050 reported only 547 MiB free, below the runner's hard
3,000 MiB free requirement. Do not evict other workloads.

This follows PR #70's real Qwen3.5-0.8B-Base LoRA proof: the adapted model
lowered the taught target loss on an **untaught** node almost as much as
on the taught one. That is an adversarial signal of answer-token bias,
not selective recall or novel knowledge transfer.

## Frozen single-machine research contract

- Base: `Qwen/Qwen3.5-0.8B-Base` revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Local WSL GPU environment from PR #70: torch 2.9.1+cu128,
  Transformers 5.14.1, PEFT 0.19.1, bitsandbytes 0.49.2.
- Seeds: **109, 110, 111**, each with two separate
  arms: `true` and `shuffled` labels, both trained from a clean base.
- NF4 4-bit base; only LoRA rank 4 on q_proj/v_proj; alpha 8,
  dropout 0, AdamW `lr=0.001`, 40 optimizer steps per arm.
- Per seed: 8 previously invented known node→code pairs, 2 training
  calibration unknowns, and 4 heldout unknown node IDs. The known
  nodes have two evaluation phrasings not present in the train prompts.
  Shuffled controls rotate known labels, preserving answer frequencies.
- Evaluation ranks 9 fixed candidate completions for every question
  by teacher-forced mean token log-probability. Tie, non-finite,
  incomplete or duplicated score tables are **rejected**, not scored.
- A local arm passes an *exploratory screen* only if exact recall,
  paraphrase recall and unknown rejection each reach >=75%,
  paraphrase recall improves over the untouched base by >=25
  percentage points, and unknown rejection does not deteriorate.
- A genuine success claim additionally requires the **correct-label
  arm to outperform the shuffled arm over multiple seeds**, plus
  independent hidden-data custody and retention checks. Those
  latter gates are NOT implemented/proven by this source package.

The manifest generator is deterministic and its source is public:
there is **no independent sealed holdout** or authenticated external
evaluator here. The source tests demonstrate that hidden unknown IDs
and heldout prompt templates are absent from the training examples;
they do NOT prove the trainer could not infer these from source code.
## Explicit bounded execution (only after the GPU is available)

From this source directory, set both environment variables to
**private absolute paths outside Git**. The training output directory
must not already exist; use a distinct path per seed and arm.

```bash
export NEUROSYNC_HF_HOME=/home/vera/.cache/bt2-neurosync-qwen-20261116/hf
export NEUROSYNC_OUTPUT_DIR=/mnt/d/VERA/.scratch/ns-selective-109-true
python run_gpu.py train --seed 109 --arm true --steps 40 --lr 0.001
# Only after the training process has exited:
python run_gpu.py evaluate --seed 109 --arm true
```

Repeat on separate scratch paths for `--arm shuffled`, and for
seeds 110 and 111. Never run both arms concurrently on a 4 GB GPU.
A training receipt includes source revision, seed, arm, parameter count,
optimizer loss history, input-manifest SHA-256 and adapter-file hashes.
The fresh-process receiver checks the manifest, exact model revision,
and allowlisted adapter filenames/hashes before loading weights.
The evaluator stores its per-candidate scores, checked weight/config digests,
and cautious screen verdict in the private output directory; model files never
enter Git. The paired diagnostic binds both digests to each training receipt
and requires identical raw config digests within each true/shuffled pair,
consistent with the frozen recipe. Older receipts lacking the evaluation config
digest are incomplete for this paired diagnostic; do not backfill or treat
matching self-reported digests as independent custody.
The script refuses GPU allocation below 3,000 MiB free, caps its
PyTorch CUDA allocation fraction at 85%, refuses preexisting output
and evaluation receipt paths, and never imports a model on `--help`
or on an environment/path validation failure. The environment and
artifact contents are local/operator-owned, not independent custody.
A successful process readback is not a host reboot or a formal
identity/continuity claim.

### Source-only verification

```bash
python3 -m unittest discover -s experiments/real-qwen-selective -p 'test_*.py' -v
```

Three-seed tokenizer preflight with the actual pinned Qwen tokenizer
checked all **570** concatenated prompt/candidate alignments:
`mismatched=0`. This was a CPU tokenizer check, not a GPU model test.
The experimental runner remains unqualified for production. PR #70
and its private adapter must remain unchanged and available for
independent replication. No extra source file here grants host authority.
