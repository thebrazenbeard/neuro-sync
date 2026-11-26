"""Source/CLI guards only. Never loads a GPU or fetches a model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_gpu.py"
REPO = HERE.parents[1]

class RunnerSourceGuards(unittest.TestCase):
    def _run(self, args, env=None):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            env=env, capture_output=True, text=True, timeout=15)

    def test_cli_help_does_not_require_cuda(self):
        result = self._run(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train", result.stdout)
        self.assertIn("evaluate", result.stdout)
    def test_missing_private_cache_fails_before_gpu_or_download(self):
        env = os.environ.copy()
        env.pop("NEUROSYNC_HF_HOME", None)
        env.pop("NEUROSYNC_OUTPUT_DIR", None)
        result = self._run(["train", "--seed", "109", "--arm", "true"], env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NEUROSYNC_HF_HOME", result.stderr)
        self.assertNotIn("Downloading", result.stderr)

    def test_refuses_adapter_artifact_within_repository(self):
        env = os.environ.copy()
        env.update(NEUROSYNC_HF_HOME=str(REPO), NEUROSYNC_OUTPUT_DIR=str(HERE / "unsafe"))
        result = self._run(["train", "--seed", "109", "--arm", "true"], env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REFUSE_MODEL_ARTIFACT_IN_GIT_TREE", result.stderr)
        self.assertFalse((HERE / "unsafe").exists())
    def test_bad_seed_or_lr_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as root:
            env = os.environ.copy()
            env.update(NEUROSYNC_HF_HOME=root,
                       NEUROSYNC_OUTPUT_DIR=str(Path(root) / "new-adapter"))
            for flags in (["--seed", "-1", "--arm", "true"],
                          ["--seed", "109", "--arm", "true", "--lr", "nan"],
                          ["--seed", "109", "--arm", "true", "--lr", "999"]):
                with self.subTest(flags=flags):
                    result = self._run(["train", *flags], env)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((Path(root) / "new-adapter").exists())
                    self.assertNotIn("Downloading", result.stderr)

    def test_existing_output_refused_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as root:
            env = os.environ.copy()
            env.update(NEUROSYNC_HF_HOME=root, NEUROSYNC_OUTPUT_DIR=root)
            result = self._run(["train", "--seed", "109", "--arm", "true"], env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("REFUSE_EXISTING_OUTPUT_DIR", result.stderr)

    def test_receipt_file_map_rejects_paths_and_bad_hashes_before_read(self):
        from run_gpu import validated_adapter_files
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            model = folder / "adapter_model.safetensors"
            config = folder / "adapter_config.json"
            model.write_bytes(b"fixture-weights")
            config.write_bytes(b"fixture-config")
            hashes = {
                model.name: hashlib.sha256(model.read_bytes()).hexdigest(),
                config.name: hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            self.assertEqual(set(validated_adapter_files(folder, hashes)), set(hashes))
            for corrupt in (
                {model.name: hashes[model.name]},
                {**hashes, "../outside": "a" * 64},
                {**hashes, model.name: "x" * 64},
            ):
                with self.subTest(files=corrupt):
                    with self.assertRaises(ValueError):
                        validated_adapter_files(folder, corrupt)

    def test_eval_receipt_binds_checked_config_without_model(self):
        import hashlib
        import json
        import types
        from unittest.mock import patch
        import run_gpu as runner
        from test_paired_receipts import rows
        for mutate in (False, True):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as root:
                folder=Path(root)
                config=folder/"adapter_config.json"
                config.write_bytes(b"synthetic-config")
                weights=folder/"adapter_model.safetensors"
                weights.write_bytes(b"synthetic-weights")
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (config,weights)}
                train,suite=runner.generate_suites(109)
                receipt={"source":runner.SOURCE,"revision":runner.REVISION,"seed":109,"arm":"true",
                         "train_manifest_sha256":hashlib.sha256(runner.canonical(train).encode()).hexdigest(),
                         "adapter_files_sha256":files}
                (folder/"train-receipt.json").write_text(json.dumps(receipt),encoding="utf8")
                torch=types.SimpleNamespace(manual_seed=lambda _:None,set_num_threads=lambda _:None)
                base,adapted=object(),object()
                peft=types.SimpleNamespace(PeftModel=types.SimpleNamespace(
                    from_pretrained=lambda *a,**k:adapted))
                def score(model,*unused):
                    if mutate and model is adapted:
                        config.write_bytes(b"changed-config")
                    return rows(suite,8 if model is adapted else 2,4)
                with patch.dict(sys.modules,{"torch":torch,"peft":peft}), \
                        patch.object(runner,"load_base",return_value=(base,object(),torch)), \
                        patch.object(runner,"score_model",side_effect=score):
                    if mutate:
                        with self.assertRaisesRegex(ValueError,"POST_EVAL_ADAPTER_DIGEST_MISMATCH"):
                            runner.evaluate(types.SimpleNamespace(seed=109,arm="true"),folder)
                        self.assertFalse((folder/"eval-receipt.json").exists())
                    else:
                        runner.evaluate(types.SimpleNamespace(seed=109,arm="true"),folder)
                        observed=json.loads((folder/"eval-receipt.json").read_text(encoding="utf8"))
                        self.assertEqual(observed.get("adapter_config_sha256"),files["adapter_config.json"])

if __name__ == "__main__":
    unittest.main()
