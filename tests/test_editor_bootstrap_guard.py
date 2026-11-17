"""No automatic editor code execution or relaxed dotenv ignores in this repo."""
import tempfile
import unittest
from pathlib import Path
from editor_bootstrap_guard import check_checkout

SAFE_IGNORE = ".env\n.env.*\n!.env.example\n*.pem\n*.key\n"

class EditorBootstrapGuardTests(unittest.TestCase):
    def fixture(self, name=None, content="x"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".gitignore").write_text(SAFE_IGNORE, encoding="utf8")
        if name:
            dst = root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf8")
        return root

    def test_safe_minimal_checkout_passes(self):
        self.assertEqual(check_checkout(self.fixture()), [])

    def test_editor_folder_open_task_rejected(self):
        root = self.fixture(".vscode/tasks.json",
            '{"version":"2.0.0","tasks":[{"label":"auto","runOptions":{"runOn":"folderOpen"},'
            '"type":"process","command":"node","args":[".vscode/ini"]}]}')
        self.assertTrue(any("tasks.json" in s for s in check_checkout(root)))

    def test_obfuscated_editor_bootstrap_rejected_without_executing(self):
        root = self.fixture(".vscode/ini", "var encrypted = 'test-inert-data';")
        self.assertTrue(any(".vscode/ini" in s for s in check_checkout(root)))

    def test_workspace_settings_rejected(self):
        root = self.fixture(".vscode/settings.json",
                            '{"task.allowAutomaticTasks":true}')
        self.assertTrue(any("settings.json" in s for s in check_checkout(root)))

    def test_inert_extensions_allowlisted(self):
        root = self.fixture(".vscode/extensions.json", '{"recommendations":[]}')
        self.assertEqual(check_checkout(root), [])

    def test_missing_dotenv_ignore_fails(self):
        root = self.fixture()
        (root / ".gitignore").write_text("*.pem\n*.key\n", encoding="utf8")
        self.assertTrue(any(".env" in s for s in check_checkout(root)))

    def test_unignore_dotenv_fails(self):
        root = self.fixture()
        (root / ".gitignore").write_text(SAFE_IGNORE + "!.env\n", encoding="utf8")
        self.assertTrue(any("unignore" in s for s in check_checkout(root)))

    def test_unknown_vscode_file_fails(self):
        root = self.fixture(".vscode/launch.json", '{"preLaunchTask":"auto"}')
        self.assertTrue(any("launch.json" in s for s in check_checkout(root)))

    def test_actual_clean_checkout_passes(self):
        self.assertEqual(check_checkout(Path(__file__).resolve().parents[1]), [])

if __name__ == "__main__":
    unittest.main()
