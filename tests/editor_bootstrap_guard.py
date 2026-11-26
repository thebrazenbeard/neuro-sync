"""Fail-closed repo-specific check for editor autorun and dotenv exposure.

Read-only and stdlib-only; never imports or executes candidate editor scripts.
"""
from pathlib import Path
import argparse
import sys

_ALLOWED_EDITOR_FILES = frozenset({"extensions.json"})
_REQUIRED_PRIVATE_PATTERNS = frozenset({".env", ".env.*"})


def check_checkout(root):
    root = Path(root).resolve()
    issues = []
    ignore = root / ".gitignore"
    try:
        rules = [line.strip() for line in
                 ignore.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeError) as exc:
        issues.append(".gitignore unreadable: " + type(exc).__name__)
        rules = []
    for required in sorted(_REQUIRED_PRIVATE_PATTERNS):
        if required not in rules:
            issues.append(".gitignore missing private pattern " + required)
    for rule in rules:
        if rule.startswith("!") and rule[1:] in _REQUIRED_PRIVATE_PATTERNS:
            issues.append(".gitignore explicit unignore forbidden: " + rule)

    vscode = root / ".vscode"
    if vscode.exists() or vscode.is_symlink():
        if vscode.is_symlink() or not vscode.is_dir():
            issues.append(".vscode is not an ordinary directory")
        else:
            for candidate in vscode.rglob("*"):
                relative = candidate.relative_to(vscode).as_posix()
                if candidate.is_dir() and not candidate.is_symlink():
                    continue
                if relative not in _ALLOWED_EDITOR_FILES or candidate.is_symlink():
                    issues.append("unreviewed .vscode/" + relative)
    return sorted(issues)


def main():
    parser = argparse.ArgumentParser(description="Reject unreviewed editor autorun")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    issues = check_checkout(args.root)
    for issue in issues:
        print("EDITOR_SECURITY_HOLD: " + issue, file=sys.stderr)
    if issues:
        return 1
    print("EDITOR_CHECKOUT_GUARD_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
