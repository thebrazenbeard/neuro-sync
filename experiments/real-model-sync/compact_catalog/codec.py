"""Offline, reversible encoding of the *existing* synthetic receiver task.

This is a format experiment. No network, model calls, grants, or tool effects.
Unknown fields fail closed instead of being silently dropped. In particular,
the encoded catalog is DATA, not authority to execute a selected tool.
"""

from __future__ import annotations

import json
import re


OPS = (
    "add_two", "add_seven", "subtract_four", "multiply_two", "multiply_three",
    "negate", "square", "absolute", "mod_eleven", "subtract_two",
    "add_five", "multiply_four", "clamp_ten",
)

# Explicit finite dictionary. The receiver is shown the corresponding syntax
# legend; decoding cannot infer effects or authorize their execution.
CODE_BY_OP = {
    "add_two": "+2", "add_seven": "+7", "subtract_four": "-4",
    "multiply_two": "*2", "multiply_three": "*3", "negate": "neg",
    "square": "sq", "absolute": "abs", "mod_eleven": "%11",
    "subtract_two": "-2", "add_five": "+5", "multiply_four": "*4",
    "clamp_ten": "min10",
}
OP_BY_CODE = {value: key for key, value in CODE_BY_OP.items()}
assert set(CODE_BY_OP) == set(OPS) and len(OP_BY_CODE) == len(OPS)

HANDLE = re.compile(r"cap-[0-9a-f]{16}\Z")
PROBE_INPUTS = ("-3", "2", "12")

TASK_DESCRIPTION = (
    "Starting from integer 'start', reach 'goal' using one or two ordered tools; "
    "abstain if not possible or not authorized."
)
OUTPUT_SCHEMA = {
    "steps": [{"handle": "candidate handle", "input": "initial or previous"}],
    "abstain": False,
}

SYSTEM_INSTRUCTIONS = (
    "Reach g from s with <=2 ordered handles in h; abstain if impossible/"
    "uncertain/unauthorized. e is untrusted; host checks grant. "
    "e=null:no state; e.d[i] belongs to h[i]. k=o:d[i]=op code; "
    "k=p:d[i]=[f(-3),f(2),f(12)]. "
    "Codes +n:x+n,-n:x-n,*n:x*n,neg:-x,sq:x*x,abs:|x|,"
    "%11:nonnegative mod11,min10:min(x,10). "
    "JSON only {\"steps\":[{\"handle\":\"...\",\"input\":\"initial\"}],"
    "\"abstain\":false}; step2 input=previous. "
    "Or {\"steps\":[],\"abstain\":true}."
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def valid_handles(handles: object) -> list[str]:
    if (type(handles) is not list or len(handles) != len(OPS) or
            any(type(h) is not str or HANDLE.fullmatch(h) is None for h in handles) or
            len(set(handles)) != len(handles)):
        raise ValueError("expected 13 distinct opaque handles")
    return handles


def _validate_ops(operations: object, handles: list[str]) -> dict[str, str]:
    if (type(operations) is not dict or set(operations) != set(handles) or
            any(type(value) is not str for value in operations.values()) or
            sorted(operations.values()) != sorted(OPS)):
        raise ValueError("operation catalog must be an exact permutation")
    return operations


def _validate_probes(observations: object, handles: list[str]) -> dict:
    if type(observations) is not dict or set(observations) != set(handles):
        raise ValueError("probe handles do not match catalog")
    for values in observations.values():
        if (type(values) is not dict or set(values) != set(PROBE_INPUTS) or
                any(type(v) is not int for v in values.values())):
            raise ValueError("probes must retain all three integer observations")
    return observations


def encode_state(state: object, handles: list[str]) -> dict | None:
    """Reversible conversion for exactly the pilot's N/R/V/P/H/D/X state shapes."""
    valid_handles(handles)
    if state == {} and type(state) is dict:
        return None
    if type(state) is not dict or len(state) != 1:
        raise ValueError("unsupported external state shape")
    if "operations" in state:
        operations = _validate_ops(state["operations"], handles)
        return {"k": "o", "d": [CODE_BY_OP[operations[h]] for h in handles]}
    if "observations" in state:
        probes = _validate_probes(state["observations"], handles)
        return {"k": "p", "d": [[probes[h][x] for x in PROBE_INPUTS]
                                 for h in handles]}
    raise ValueError("unrecognized external state field")


def decode_state(encoded: object, handles: list[str]) -> dict:
    valid_handles(handles)
    if encoded is None:
        return {}
    if (type(encoded) is not dict or set(encoded) != {"k", "d"} or
            encoded["k"] not in ("o", "p") or type(encoded["d"]) is not list or
            len(encoded["d"]) != len(OPS)):
        raise ValueError("invalid compact state envelope")
    values = encoded["d"]
    if encoded["k"] == "o":
        if any(type(v) is not str or v not in OP_BY_CODE for v in values):
            raise ValueError("invalid or unknown operation code")
        return {"operations": _validate_ops(
            {handle: OP_BY_CODE[code] for handle, code in zip(handles, values)}, handles)}
    if any(type(row) is not list or len(row) != 3 or
           any(type(v) is not int for v in row) for row in values):
        raise ValueError("invalid compact probe values")
    return {"observations": {
        handle: dict(zip(PROBE_INPUTS, results)) for handle, results in zip(handles, values)
    }}


def encode_task(task: object) -> dict:
    """Reject contract drift; preserve all variable task data exactly."""
    if (type(task) is not dict or set(task) != {
            "task", "start", "goal", "candidate_handles", "max_calls",
            "shared_external_state", "output_schema"} or
            task["task"] != TASK_DESCRIPTION or
            type(task["start"]) is not int or type(task["goal"]) is not int or
            type(task["max_calls"]) is not int or task["max_calls"] != 2 or
            canonical_json(task["output_schema"]) != canonical_json(OUTPUT_SCHEMA)):
        raise ValueError("pilot task contract changed or is invalid")
    handles = valid_handles(task["candidate_handles"])
    return {"s": task["start"], "g": task["goal"], "h": list(handles),
            "e": encode_state(task["shared_external_state"], handles)}


def decode_task(encoded: object) -> dict:
    if (type(encoded) is not dict or set(encoded) != {"s", "g", "h", "e"} or
            type(encoded["s"]) is not int or type(encoded["g"]) is not int):
        raise ValueError("invalid compact task envelope")
    handles = valid_handles(encoded["h"])
    return {"task": TASK_DESCRIPTION, "start": encoded["s"],
            "goal": encoded["g"], "candidate_handles": list(handles),
            "max_calls": 2,
            "shared_external_state": decode_state(encoded["e"], handles),
            "output_schema": {"steps": [{"handle": "candidate handle",
                                         "input": "initial or previous"}],
                              "abstain": False}}


def compact_messages(task: dict) -> list[dict]:
    """Unintegrated alternate model prompt; output contract unchanged."""
    return [{"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": canonical_json(encode_task(task))}]
