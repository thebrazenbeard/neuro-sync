"""Counterfactual source fixture: read B-visible broker paths without StateBus.

This intentionally *does not* implement the restricted receiver's policy gate.
It shows that passing raw broker and registry paths to B is not a genuine
information-isolation boundary. It is not an LLM, legitimate grant bypass,
or an operational recommendation to disregard permissions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def recover(*, broker_path: Path, registry_path: Path, x: int) -> dict:
    """Use only paths already supplied to every arm of the scripted pilot."""
    if type(x) is not int or abs(x) > 100_000:
        return {"ok": False, "reason": "invalid-task"}
    try:
        broker = json.loads(broker_path.read_text(encoding="utf8"))
        registry = json.loads(registry_path.read_text(encoding="utf8"))
        ref = broker["ref"]
        a, b = registry[ref]
        if not (type(a) is int and type(b) is int):
            raise ValueError("invalid tool coefficients")
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return {"ok": False, "reason": "missing-broker-reference"}
    return {"ok": True, "value": a * x + b, "via": "broker-path-counterfactual"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    try:
        task = json.loads(sys.stdin.read(4097))
    except json.JSONDecodeError:
        task = None
    if not isinstance(task, dict) or set(task) != {"x"}:
        result = {"ok": False, "reason": "invalid-task"}
    else:
        result = recover(broker_path=args.broker, registry_path=args.registry,
                         x=task["x"])
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
