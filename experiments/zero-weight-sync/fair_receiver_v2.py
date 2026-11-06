"""Receiver-only scripted B policy for a fair zero-weight comparison.

This executable contains NO host scenario generator, oracle, coefficient
registry, capability implementation, broker client, or StateBus read path.
It accepts only opaque handles, integer tasks, and an optional scoped ref.
Its output selects handles; host independently executes/scores each choice.
"""

from __future__ import annotations

import hashlib
import json
import sys

RECEIVER_POLICY = "scripted-receiver-v2-guess-when-uninformed"


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def receiver_decide(request: object) -> dict:
    """Same B decision logic for N/V/P/H; blind guessing is permissible."""
    if not isinstance(request, dict) or set(request) != {"handles", "tasks", "state"}:
        raise ValueError("invalid receiver request fields")
    handles, tasks, state = request["handles"], request["tasks"], request["state"]
    if (not isinstance(handles, list) or len(handles) < 3 or len(handles) > 32 or
            not all(type(h) is str and h.startswith("cap-") and len(h) <= 40 for h in handles) or
            len(set(handles)) != len(handles) or
            not isinstance(tasks, list) or len(tasks) < 1 or len(tasks) > 32 or
            not all(type(x) is int and abs(x) <= 500 for x in tasks)):
        raise ValueError("invalid public handles or task inputs")
    if state is not None and (not isinstance(state, dict) or
                              set(state) != {"ref"} or type(state["ref"]) is not str or
                              state["ref"] not in handles):
        raise ValueError("invalid selected reference")
    choices = []
    for x in tasks:
        if state is None:
            # N attempts the same candidate tool interface without a hint.
            n = int(hashlib.sha256(f"{RECEIVER_POLICY}|{x}".encode()).hexdigest(), 16)
            ref, decision = handles[n % len(handles)], "uninformed-guess"
        else:
            ref, decision = state["ref"], "supplied-reference"
        choices.append({"x": x, "ref": ref, "decision": decision})
    return {"choices": choices, "receiver_policy_sha256": hashlib.sha256(
        canonical(RECEIVER_POLICY).encode("utf8")).hexdigest()}


def main() -> None:
    raw = sys.stdin.read(8193)
    if len(raw) > 8192:
        raise ValueError("receiver payload too large")
    print(canonical(receiver_decide(json.loads(raw))))


if __name__ == "__main__":
    main()
