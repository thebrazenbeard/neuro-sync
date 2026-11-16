import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent

class SourceGuardTests(unittest.TestCase):
    def test_scripts_compile_without_cuda_or_model_download(self):
        for name in ("train_gate.py","receiver_gate.py"):
            with self.subTest(name=name):
                ast.parse((HERE/name).read_text(encoding="utf8"))
    def test_scripts_require_explicit_private_cache_before_imports(self):
        env=os.environ.copy()
        env.pop("NEUROSYNC_HF_HOME",None)
        env.pop("NEUROSYNC_OUTPUT_DIR",None)
        for name in ("train_gate.py","receiver_gate.py"):
            with self.subTest(name=name):
                result=subprocess.run([sys.executable,str(HERE/name)],env=env,
                                      capture_output=True,text=True,timeout=15)
                self.assertNotEqual(result.returncode,0)
                self.assertIn("NEUROSYNC_HF_HOME",result.stderr)
                self.assertNotIn("Downloading",result.stderr)
    def test_refuse_to_embedded_output_path_in_source(self):
        for name in ("train_gate.py","receiver_gate.py"):
            s=(HERE/name).read_text(encoding="utf8")
            self.assertIn("REFUSE_MODEL_ARTIFACT_IN_GIT_TREE",s)
            self.assertNotIn("bt2-neurosync-qwen-adapter-20261116",s)

if __name__=="__main__":
    unittest.main()
