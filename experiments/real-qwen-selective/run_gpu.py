"""Experimental Qwen selective-recall arms. Never promoted as proven novelty.

Phases run in distinct processes. Only synthetic generated data. No service effects.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from selective_protocol import generate_suites, shuffled_training, compare_models

SOURCE = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def context(args):
    if "NEUROSYNC_HF_HOME" not in os.environ:
        raise ValueError("NEUROSYNC_HF_HOME required")
    if "NEUROSYNC_OUTPUT_DIR" not in os.environ:
        raise ValueError("NEUROSYNC_OUTPUT_DIR required")
    out = Path(os.environ["NEUROSYNC_OUTPUT_DIR"]).resolve()
    if out == REPO or REPO in out.parents:
        raise ValueError("REFUSE_MODEL_ARTIFACT_IN_GIT_TREE")
    os.environ["HF_HOME"] = os.environ["NEUROSYNC_HF_HOME"]
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return out

def validated_adapter_files(folder, files):
    """Never trust file paths or digest syntax from a local receipt."""
    allowed = {"adapter_model.safetensors", "adapter_config.json", "README.md"}
    required = {"adapter_model.safetensors", "adapter_config.json"}
    if type(files) is not dict or not required <= set(files) or set(files) - allowed:
        raise ValueError("INVALID_ADAPTER_FILE_MAP")
    folder = Path(folder).resolve()
    checked = {}
    for name, expected in files.items():
        if type(name) is not str or type(expected) is not str:
            raise ValueError("INVALID_ADAPTER_DIGEST")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("INVALID_ADAPTER_DIGEST")
        item = folder / name
        if item.is_symlink() or not item.is_file() or item.resolve().parent != folder:
            raise ValueError("ADAPTER_FILE_OUTSIDE_CUSTODY")
        checked[name] = item
    return checked


def load_base():
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    free, total = torch.cuda.mem_get_info()
    if free < 3000 * 1048576:
        raise RuntimeError("GPU_HEADROOM_REFUSE")
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        SOURCE, revision=REVISION, dtype=torch.float16,
        quantization_config=config, device_map={"": 0}, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, revision=REVISION,
                                               trust_remote_code=False)
    model.config.use_cache = False
    return model, tokenizer, torch

def labeled_tokens(tokenizer, prompt, answer, torch):
    ids = tokenizer(prompt + answer, return_tensors="pt", add_special_tokens=True)["input_ids"]
    prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if not torch.equal(ids[0, :len(prefix)], torch.tensor(prefix)):
        raise ValueError("TOKENIZER_PREFIX_MISMATCH")
    if len(ids[0]) <= len(prefix):
        raise ValueError("EMPTY_ANSWER_TOKENS")
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    return ids.to("cuda"), labels.to("cuda"), len(prefix)

def train(args, out):
    if out.exists():
        raise ValueError("REFUSE_EXISTING_OUTPUT_DIR")
    if args.steps < 1 or args.steps > 200:
        raise ValueError("STEPS_OUTSIDE_BOUND")
    import torch
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    suite, _unused = generate_suites(args.seed)
    if args.arm == "shuffled":
        suite = shuffled_training(suite)
    model, tokenizer, _ = load_base()
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0, bias="none",
                        target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or not all(p.ndim > 0 for p in params):
        raise RuntimeError("NO_TRAINABLE_ADAPTER")
    examples = suite["examples"]
    tokens = [labeled_tokens(tokenizer, row["prompt"], row["answer"], torch)[:2]
              for row in examples]
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train()
    losses = []
    for step in range(args.steps):
        ids, labels = tokens[step % len(tokens)]
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("NONFINITE_LOSS")
        loss.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in params):
            raise RuntimeError("MISSING_FINITE_GRADIENTS")
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            print(canonical({"step": step + 1, "loss": losses[-1]}), flush=True)
    out.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(str(out), safe_serialization=True)
    hashes = {p.name: sha_file(p) for p in out.iterdir() if p.is_file()}
    receipt = {
        "evidence": "REAL_GPU_LORA_TRAINED_UNBLINDED",
        "source": SOURCE, "revision": REVISION,
        "protocol": suite["protocol"], "arm": args.arm, "seed": args.seed,
        "steps": args.steps, "learning_rate": args.lr,
        "train_manifest_sha256": hashlib.sha256(canonical(suite).encode()).hexdigest(),
        "adapter_files_sha256": hashes,
        "losses": losses, "trainable_parameters": sum(p.numel() for p in params),
        "peak_torch_vram_mib": round(torch.cuda.max_memory_allocated() / 1048576),
        "blind_heldout": False, "novel_transfer_proven": False}
    (out / "train-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf8")
    print(canonical({"status": "TRAINED", "arm": args.arm, "steps": args.steps,
                     "last_loss": losses[-1], "adapter_hashes": hashes}), flush=True)

def candidate_mean_logprob(model, tokenizer, torch, prompt, completion):
    ids, _labels, prefix_len = labeled_tokens(tokenizer, prompt, completion, torch)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits.float()[0]
        positions = logits[prefix_len - 1:-1, :]
        targets = ids[0, prefix_len:]
        if len(targets) != len(positions):
            raise ValueError("CAUSAL_ALIGNMENT_FAILURE")
        chosen = torch.log_softmax(positions, dim=-1).gather(
            -1, targets.unsqueeze(-1)).squeeze(-1)
    result = float(chosen.mean().item())
    if not math.isfinite(result):
        raise ValueError("NONFINITE_CANDIDATE_SCORE")
    return result

def score_model(model, tokenizer, torch, suite):
    rows = []
    model.eval()
    for ex in suite["examples"]:
        scores = {answer: candidate_mean_logprob(
            model, tokenizer, torch, ex["prompt"], answer)
            for answer in suite["candidates"]}
        rows.append({"id": ex["id"], "variant": ex["variant"], "scores": scores})
    return rows

def evaluate(args, out):
    if not out.is_dir() or (out / "eval-receipt.json").exists():
        raise ValueError("MISSING_TRAIN_OR_EVALUATION_ALREADY_PRESENT")
    receipt = json.loads((out / "train-receipt.json").read_text(encoding="utf8"))
    if receipt.get("source") != SOURCE or receipt.get("revision") != REVISION:
        raise ValueError("MODEL_REVISION_MISMATCH")
    if receipt.get("seed") != args.seed or receipt.get("arm") != args.arm:
        raise ValueError("EXPERIMENT_ARM_OR_SEED_MISMATCH")
    files = receipt.get("adapter_files_sha256", {})
    if not files or "adapter_model.safetensors" not in files:
        raise ValueError("MISSING_ADAPTER_HASH")
    checked = validated_adapter_files(out, files)
    for name, path in checked.items():
        if sha_file(path) != files[name]:
            raise ValueError("ADAPTER_DIGEST_MISMATCH")
    import torch
    from peft import PeftModel
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    train_suite, eval_suite = generate_suites(args.seed)
    if args.arm == "shuffled":
        train_suite = shuffled_training(train_suite)
    if hashlib.sha256(canonical(train_suite).encode()).hexdigest() != receipt.get("train_manifest_sha256"):
        raise ValueError("TRAIN_MANIFEST_MISMATCH")
    base, tokenizer, _ = load_base()
    base_rows = score_model(base, tokenizer, torch, eval_suite)
    receiver = PeftModel.from_pretrained(base, str(out), is_trainable=False)
    adapted_rows = score_model(receiver, tokenizer, torch, eval_suite)
    result = compare_models(eval_suite, base_rows, adapted_rows)
    result["seed"] = args.seed
    result["arm"] = args.arm
    result["donor_process_attestation"] = "NOT_ATTESTED"
    result["host_reboot_proven"] = False
    result["source_revision"] = REVISION
    result["adapter_sha256"] = files["adapter_model.safetensors"]
    result["raw_rows"] = {"base": base_rows, "adapted": adapted_rows}
    for name, path in checked.items():
        if sha_file(path) != files[name]:
            raise ValueError("POST_EVAL_ADAPTER_DIGEST_MISMATCH")
    (out / "eval-receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf8")
    print(canonical({k: v for k, v in result.items() if k != "raw_rows"}), flush=True)

def main():
    parser = argparse.ArgumentParser(description="Bounded real Qwen LoRA protocol")
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("true", "shuffled"), required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("INVALID_SEED")
    if not math.isfinite(args.lr) or not 0 < args.lr <= 0.01:
        raise ValueError("INVALID_LEARNING_RATE")
    target = context(args)
    if args.phase == "train":
        train(args, target)
    else:
        evaluate(args, target)

if __name__ == "__main__":
    main()
