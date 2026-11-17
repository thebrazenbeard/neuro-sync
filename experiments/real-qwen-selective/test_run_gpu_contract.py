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

if __name__ == "__main__":
    unittest.main()
