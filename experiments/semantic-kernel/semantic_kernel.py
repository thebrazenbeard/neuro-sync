"""Source-bound, deterministic *candidate* semantic kernel for a local decoder.

No model weights, remote fetches, authority admission, or host effects are used.
Inputs must be independently verified public bytes and explicitly typed claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SCHEMA = "mini-vera-semantic-kernel-prototype-v0"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")
_SAFE_PATH = re.compile(r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[a-zA-Z0-9_./-]{1,250}$")
_WORD = re.compile(r"[a-z0-9]{2,}")


class ContractError(ValueError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _check_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ContractError(f"invalid_{label}")
    return value


def _require_keys(obj: object, keys: set[str], required: set[str], label: str) -> dict:
    if not isinstance(obj, dict) or set(obj) - keys or required - set(obj):
        raise ContractError(f"invalid_{label}_shape")
    return obj


def compile_kernel(contract: dict) -> dict:
    """Compile explicitly typed public source assertions; never infer grants from prose."""
    _require_keys(contract, {"sources", "claims", "procedures"}, {"sources", "claims", "procedures"}, "contract")
    for field in ("sources", "claims", "procedures"):
        if not isinstance(contract[field], list) or len(contract[field]) > 256:
            raise ContractError(f"invalid_{field}")

    sources: dict[str, dict] = {}
    for source in contract["sources"]:
        _require_keys(source, {"id", "repo", "commit", "path", "sha256", "body", "privacy"},
                      {"id", "repo", "commit", "path", "sha256", "body", "privacy"}, "source")
        sid = _check_name(source["id"], "source_id")
        if sid in sources:
            raise ContractError("duplicate_source_id")
        if source["privacy"] != "PUBLIC_APPROVED":
            raise ContractError("source_privacy_not_approved")
        if not isinstance(source["repo"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repo"]):
            raise ContractError("invalid_repo")
        if not isinstance(source["commit"], str) or not _HEX40.fullmatch(source["commit"]):
            raise ContractError("unpinned_commit")
        if not isinstance(source["path"], str) or not _SAFE_PATH.fullmatch(source["path"]):
            raise ContractError("unsafe_source_path")
        if not isinstance(source["body"], str) or len(source["body"].encode("utf-8")) > 131072:
            raise ContractError("invalid_source_body")
        observed = hashlib.sha256(source["body"].encode("utf-8")).hexdigest()
        if observed != source["sha256"]:
            raise ContractError("source_byte_mismatch")
        sources[sid] = {"repo": source["repo"], "commit": source["commit"],
                        "path": source["path"], "sha256": observed, "body": source["body"]}

    claims: dict[str, dict] = {}
    for claim in contract["claims"]:
        _require_keys(claim, {"id", "source_id", "subject", "relation", "object", "exact_quote", "kind"},
                      {"id", "source_id", "subject", "relation", "object", "exact_quote", "kind"}, "claim")
        cid = _check_name(claim["id"], "claim_id")
        if cid in claims:
            raise ContractError("duplicate_claim_id")
        sid = claim["source_id"]
        if sid not in sources or claim["kind"] != "SOURCE_ASSERTION":
            raise ContractError("unbound_or_promoted_claim")
        for field in ("subject", "relation", "object"):
            _check_name(claim[field], field)
        quote = claim["exact_quote"]
        if not isinstance(quote, str) or len(quote) > 2048 or len(quote) < 8 or quote not in sources[sid]["body"]:
            raise ContractError("unsupported_claim_quote")
        # A literal quote is only weak lexical support, not semantic entailment.
        # This necessary check rejects obvious field/quote substitution; a human
        # or independently qualified evaluator must still review the meaning.
        quote_terms = set(_WORD.findall(quote.lower()))
        for field in ("subject", "relation", "object"):
            if not set(_WORD.findall(claim[field].lower())).issubset(quote_terms):
                raise ContractError("unanchored_claim_field")
        claims[cid] = {k: claim[k] for k in ("source_id", "subject", "relation", "object", "exact_quote", "kind")}

    procedures: dict[str, list] = {}
    for procedure in contract["procedures"]:
        _require_keys(procedure, {"id", "steps"}, {"id", "steps"}, "procedure")
        pid = _check_name(procedure["id"], "procedure_id")
        if pid in procedures or not isinstance(procedure["steps"], list) or len(procedure["steps"]) > 64:
            raise ContractError("invalid_procedure")
        steps = {}
        for step in procedure["steps"]:
            _require_keys(step, {"id", "kind", "requires", "capability"}, {"id", "kind", "requires"}, "step")
            stepid = _check_name(step["id"], "step_id")
            if stepid in steps or step["kind"] not in {"EVIDENCE_READ", "MODEL_DRAFT", "HOST_EFFECT"}:
                raise ContractError("invalid_step")
            requires = step["requires"]
            if not isinstance(requires, list) or any(not isinstance(x, str) for x in requires) or len(requires) != len(set(requires)):
                raise ContractError("invalid_step_dependencies")
            cap = step.get("capability")
            if step["kind"] == "HOST_EFFECT":
                _check_name(cap, "effect_capability")
            elif cap is not None:
                raise ContractError("capability_on_non_effect_step")
            steps[stepid] = {"kind": step["kind"], "requires": sorted(requires), "capability": cap}
        def visit(stepid: str, stack: set[str], seen: set[str]):
            if stepid not in steps or stepid in stack:
                raise ContractError("dangling_or_cyclic_step")
            if stepid in seen:
                return
            for dep in steps[stepid]["requires"]:
                visit(dep, stack | {stepid}, seen)
            seen.add(stepid)
        visited: set[str] = set()
        for stepid in steps:
            visit(stepid, set(), visited)
        procedures[pid] = [{"id": sid, **steps[sid]} for sid in sorted(steps)]

    conflicts: dict[tuple[str, str], set[str]] = {}
    for claim in claims.values():
        key = (claim["subject"], claim["relation"])
        conflicts.setdefault(key, set()).add(claim["object"])
    output = {
        "schema": SCHEMA,
        "sources": [{"id": sid, **{k: v for k, v in sources[sid].items() if k != "body"},
                     "pin_evidence": "CALLER_ASSERTED_PIN_UNVERIFIED"} for sid in sorted(sources)],
        "claims": [{"id": cid, **claims[cid]} for cid in sorted(claims)],
        "conflicts": [{"subject": key[0], "relation": key[1], "objects": sorted(vals)}
                      for key, vals in sorted(conflicts.items()) if len(vals) > 1],
        "procedures": [{"id": pid, "steps": procedures[pid]} for pid in sorted(procedures)],
        "authority_effect": "NONE_SOURCE_CANNOT_GRANT",
        "integrity_evidence": "LOCAL_SELF_DIGEST_ONLY",
    }
    return {**output, "sha256": digest(output)}


def verify_kernel(kernel: object) -> dict:
    """Detect in-memory alteration; self-hash is not independent authenticity."""
    _require_keys(kernel, {"schema", "sources", "claims", "conflicts", "procedures",
                           "authority_effect", "integrity_evidence", "sha256"},
                  {"schema", "sources", "claims", "conflicts", "procedures",
                   "authority_effect", "integrity_evidence", "sha256"}, "kernel")
    if (kernel["schema"] != SCHEMA or kernel["authority_effect"] != "NONE_SOURCE_CANNOT_GRANT"
            or kernel["integrity_evidence"] != "LOCAL_SELF_DIGEST_ONLY"
            or kernel["sha256"] != digest({k: v for k, v in kernel.items() if k != "sha256"})):
        raise ContractError("kernel_integrity_failure")
    return kernel


def query(kernel: dict, prompt: str, *, limit: int = 4) -> list[dict]:
    """Deterministic lexical retrieval; never semantic promotion or authority."""
    kernel = verify_kernel(kernel)
    if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 4096 or not 1 <= limit <= 16:
        raise ContractError("invalid_query")
    terms = set(_WORD.findall(prompt.lower()))
    ranked = []
    for claim in kernel["claims"]:
        text = " ".join(str(claim[x]) for x in ("subject", "relation", "object", "exact_quote"))
        score = len(terms.intersection(_WORD.findall(text.lower())))
        if score:
            ranked.append((-score, claim["id"], claim))
    return [{"claim_id": c["id"], "score": -s, "source_id": c["source_id"],
             "exact_quote": c["exact_quote"], "kind": c["kind"]} for s, _, c in sorted(ranked)[:limit]]


def propose_steps(kernel: dict, procedure_id: str) -> list[dict]:
    """Plan only. An actual trusted host must separately authorize all effects."""
    kernel = verify_kernel(kernel)
    proc = next((x for x in kernel["procedures"] if x["id"] == procedure_id), None)
    if proc is None:
        raise ContractError("unknown_procedure")
    return [{"step_id": step["id"], "kind": step["kind"],
             "decision": "BLOCKED_NEEDS_HOST_AUTHORITY" if step["kind"] == "HOST_EFFECT" else "CANDIDATE_ONLY"}
            for step in proc["steps"]]


def decoder_context(kernel: dict, question: str) -> str:
    """Structured evidence as untrusted data. Decoder output can never grant effects."""
    kernel = verify_kernel(kernel)
    payload = {"question": question, "evidence": query(kernel, question),
               "unresolved_conflicts": kernel["conflicts"],
               "effect_authority": "NONE", "compiled_kernel_sha256": kernel["sha256"]}
    return "DATA_ONLY_UNTRUSTED_SOURCE_JSON\n" + canonical(payload).decode("utf-8") + "\nEND_DATA"


def inspect_decoder_tool_request(candidate: object) -> str:
    """Fail-closed illustrative gate: no authority comes from model text."""
    if not isinstance(candidate, dict) or set(candidate) != {"action", "target"}:
        return "REJECT_UNTYPED"
    if candidate["action"] == "READ_EVIDENCE" and candidate["target"] == "COMPILED_PUBLIC_CLAIMS":
        return "CANDIDATE_READ_ONLY"
    return "DENY_EFFECT_NO_HOST_GRANT"


def append_local_observation(path: Path, event: dict) -> str:
    """Local hash chain for restart tests. Same-UID rewrite is NOT prevented."""
    _require_keys(event, {"event_id", "subject", "result"}, {"event_id", "subject", "result"}, "event")
    for field in event:
        _check_name(event[field], field)
    rows = read_local_observations(path)
    if any(row["event"]["event_id"] == event["event_id"] for row in rows):
        raise ContractError("duplicate_event")
    record = {"sequence": len(rows) + 1, "previous": rows[-1]["sha256"] if rows else "0" * 64,
              "event": event, "evidence_class": "LOCAL_UNWITNESSED"}
    record["sha256"] = digest(record)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(record).decode("utf-8") + "\n")
    return record["sha256"]


def read_local_observations(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for text in path.read_text(encoding="utf-8").splitlines():
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError("invalid_journal_json") from exc
        expected = {"sequence": len(rows) + 1, "previous": rows[-1]["sha256"] if rows else "0" * 64,
                    "event": record.get("event"), "evidence_class": "LOCAL_UNWITNESSED"}
        if set(record) != set(expected) | {"sha256"} or any(record.get(k) != v for k, v in expected.items()) or record["sha256"] != digest(expected):
            raise ContractError("journal_integrity_failure")
        rows.append(record)
    return rows
