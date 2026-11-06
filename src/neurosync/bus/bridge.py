"""Source-only client-mediated StateDelta bridge and loopback model probe.

There is NO installed ChatGPT-session integration here. A person or supported
client copies an artifact between environments and an independently authorized
StateLedger decides whether it may be proposed, validated, and committed.
An independently installed/authenticated custom MCP server is a possible
future supported transport, not a capability claimed by this source module.
Transporting state does not establish semantic or execution equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from . import StateDelta, StateLedger


CLIENT_PROTOCOL = "neuro-sync.user-mediated-state-delta.v1"
MAX_ARTIFACT_BYTES = 64 * 1024
MAX_DELTA_BYTES = 48 * 1024
_CLIENT_FIELDS = frozenset({
    "protocol", "delta_json", "delta_sha256", "content_hash",
    "source_id", "target_id", "session_id",
})


class UnsupportedHostInterface(RuntimeError):
    """No installed native session-control or automatic export/import API."""


class ArtifactRejected(ValueError):
    """A transported artifact failed an integrity or context precondition."""


class LocalModelError(RuntimeError):
    """An opt-in local model call failed; never implies a state update."""


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactRejected("duplicate JSON object field")
        result[key] = value
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _check_bound_text(value: object, name: str, length: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > length:
        raise ArtifactRejected(f"invalid {name}")
    return value


def require_host_interface(interface: str) -> None:
    """Fail closed for imagined first-party ChatGPT/hosted session APIs."""
    if interface != "user-mediated":
        raise UnsupportedHostInterface(
            "Only explicit user/tool-mediated export/import is implemented; "
            "a custom MCP route requires separate installation and authentication"
        )


def export_user_artifact(delta: "StateDelta", *, interface: str = "user-mediated") -> str:
    """Make a bounded, copyable envelope; never manufacture authorization.

    StateDelta's capability/provenance references are claims until separately
    authenticated by the authority callbacks attached to the StateLedger.
    """
    require_host_interface(interface)
    from . import StateDelta

    if not isinstance(delta, StateDelta):
        raise TypeError("a StateDelta is required")
    encoded_delta = delta.to_json()
    if not isinstance(encoded_delta, str):
        raise TypeError("StateDelta.to_json must return text")
    delta_bytes = encoded_delta.encode("utf-8")
    if len(delta_bytes) > MAX_DELTA_BYTES:
        raise ArtifactRejected("StateDelta exceeds transport size bound")
    envelope = {
        "protocol": CLIENT_PROTOCOL,
        "delta_json": encoded_delta,
        "delta_sha256": _sha256(delta_bytes),
        "content_hash": delta.content_hash,
        "source_id": delta.source_id,
        "target_id": delta.target_id,
        "session_id": delta.session_id,
    }
    artifact = _canonical(envelope)
    if len(artifact.encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise ArtifactRejected("client artifact exceeds size bound")
    return artifact


@dataclass(frozen=True)
class ImportCandidate:
    """Parsed, integrity-checked transport, *not* permission or verification."""

    delta: "StateDelta" = field(repr=False)
    artifact_sha256: str
    status: str = "untrusted-transport"


def import_user_artifact(
    artifact: str,
    *,
    expected_target_id: str,
    expected_session_id: str,
    expected_source_id: str | None = None,
    interface: str = "user-mediated",
) -> ImportCandidate:
    """Parse a manual artifact; never mutate any recipient state.

    Integrity is unkeyed and provides no origin authentication. The supplied
    target/source IDs bind only context, not authority or semantic identity.
    """
    require_host_interface(interface)
    from . import StateDelta

    _check_bound_text(expected_target_id, "expected target")
    _check_bound_text(expected_session_id, "expected session")
    if expected_source_id is not None:
        _check_bound_text(expected_source_id, "expected source")
    if not isinstance(artifact, str):
        raise TypeError("artifact must be text")
    raw = artifact.encode("utf-8")
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ArtifactRejected("client artifact exceeds size bound")
    try:
        envelope = json.loads(artifact, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ArtifactRejected("malformed client artifact") from exc
    if not isinstance(envelope, dict) or set(envelope) != _CLIENT_FIELDS:
        raise ArtifactRejected("wrong or extra client artifact fields")
    if envelope["protocol"] != CLIENT_PROTOCOL:
        raise ArtifactRejected("unsupported artifact protocol")
    delta_json = _check_bound_text(envelope["delta_json"], "encoded delta", MAX_DELTA_BYTES)
    if len(delta_json.encode("utf-8")) > MAX_DELTA_BYTES:
        raise ArtifactRejected("StateDelta exceeds transport size bound")
    _check_bound_text(envelope["delta_sha256"], "delta checksum", 64)
    _check_bound_text(envelope["content_hash"], "event content hash", 64)
    source_id = _check_bound_text(envelope["source_id"], "source identity")
    target_id = _check_bound_text(envelope["target_id"], "target identity")
    session_id = _check_bound_text(envelope["session_id"], "session identity")
    if _sha256(delta_json.encode("utf-8")) != envelope["delta_sha256"]:
        raise ArtifactRejected("StateDelta transport digest mismatch")
    if target_id != expected_target_id or (expected_source_id is not None and
                                       source_id != expected_source_id):
        raise ArtifactRejected("source or target context mismatch")
    if session_id != expected_session_id:
        raise ArtifactRejected("recipient session mismatch")
    try:
        # The nested JSON has independent parser ambiguity risks: unlike
        # outer fields, merely hashing it does not reject conflicting copies
        # of a source/session field. Require strict parse and producer's
        # canonical StateDelta serialization as well as the digest.
        json.loads(delta_json, object_pairs_hook=_strict_object)
        delta = StateDelta.from_json(delta_json)
    except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise ArtifactRejected("StateDelta schema invalid") from exc
    if delta.to_json() != delta_json:
        raise ArtifactRejected("noncanonical or ambiguous StateDelta encoding")
    if (delta.source_id != source_id or delta.target_id != target_id or
        delta.session_id != session_id or
        delta.content_hash != envelope["content_hash"]):
        raise ArtifactRejected("StateDelta identity/content mismatch")
    # This is merely a transport-expiry filter; an independent authority must
    # recheck expiry using its own trusted clock at VALIDATE and COMMIT.
    now = time.time()
    if not delta.issued_at_epoch <= now < delta.expires_at_epoch:
        raise ArtifactRejected("StateDelta outside issuance/expiry interval")
    return ImportCandidate(delta, _sha256(raw))


def propose_user_import(
    artifact: str,
    *,
    expected_target_id: str,
    expected_session_id: str,
    ledger: "StateLedger",
    authorize_proposer: Callable[..., bool],
    expected_source_id: str | None = None,
) -> object:
    """Only PROPOSE to the separately governed, durable state ledger.

    The bridge has no validate/commit primitive; user/model text never
    supplies trusted validator status. Ledger owns replay/sequence semantics.
    Caller MUST provide a genuine independent proposal-authority callback.
    """
    if ledger is None or not callable(authorize_proposer):
        raise PermissionError("independent ledger and proposer authority required")
    candidate = import_user_artifact(
        artifact, expected_target_id=expected_target_id,
        expected_session_id=expected_session_id,
        expected_source_id=expected_source_id,
    )
    return ledger.propose(candidate.delta, authorize_proposer=authorize_proposer)


@dataclass(frozen=True)
class UntrustedModelReply:
    """Output text has no grant, verifier, StateDelta, or commit authority."""

    model_id: str
    request_sha256: str
    response_sha256: str
    content: str = field(repr=False)
    trust_class: str = "MODEL_OUTPUT_UNTRUSTED"


@dataclass(frozen=True)
class LocalOpenAICompatibleAdapter:
    """Optional loopback-only HTTP model probe, not a state/weight bridge.

    Configure model_id from a separately confirmed host inventory. This class
    cannot establish that the chosen tag really denotes Qwen3.5-4B, nor can it
    prove the model's semantics, persistence, weights, or installed identity.
    """

    model_id: str
    port: int = 8000
    timeout_seconds: float = 5.0
    max_prompt_chars: int = 4096
    max_response_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not 1 <= len(self.model_id) <= 128:
            raise ValueError("explicit local model ID required")
        if (isinstance(self.port, bool) or not isinstance(self.port, int) or
                not 1 <= self.port <= 65535):
            raise ValueError("port must be in range 1..65535")
        if (isinstance(self.timeout_seconds, bool) or
                not isinstance(self.timeout_seconds, (int, float)) or
                not math.isfinite(self.timeout_seconds) or
                not 0 < self.timeout_seconds <= 30):
            raise ValueError("bounded timeout must be 0..30s")
        if (isinstance(self.max_prompt_chars, bool) or
                not isinstance(self.max_prompt_chars, int) or
                not 1 <= self.max_prompt_chars <= 16384):
            raise ValueError("prompt size limit out of range")
        if (isinstance(self.max_response_bytes, bool) or
                not isinstance(self.max_response_bytes, int) or
                not 128 <= self.max_response_bytes <= 256 * 1024):
            raise ValueError("response size limit out of range")

    def chat(self, prompt: str) -> UntrustedModelReply:
        """POST only to 127.0.0.1:port/v1/chat/completions; no credentials.

        Opt-in call; default tests monkeypatch HTTPConnection and never network.
        Secrets/prompts are not logged and not included in exception messages.
        """
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= self.max_prompt_chars:
            raise ValueError("prompt length out of range")
        body = _canonical({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode("utf-8")
        if len(body) > 4 * self.max_prompt_chars + 512:
            raise ValueError("encoded request too large")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout_seconds)
        try:
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json", "Accept": "application/json"})
            response = conn.getresponse()
            declared_length = response.getheader("Content-Length")
            if declared_length is not None:
                try:
                    length = int(declared_length)
                except ValueError as exc:
                    raise LocalModelError("invalid response length") from exc
                if length < 0 or length > self.max_response_bytes:
                    raise LocalModelError("local response exceeds byte bound")
            if response.status != 200:
                raise LocalModelError("local model HTTP request failed")
            raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise LocalModelError("local response exceeds byte bound")
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise LocalModelError("local model transport unavailable") from exc
        finally:
            conn.close()
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, dict) or decoded.get("model") != self.model_id:
                raise LocalModelError("model identity reply differs from configured ID")
            choices = decoded["choices"]
            text = choices[0]["message"]["content"]
            if (not isinstance(text, str) or
                    len(text.encode("utf-8")) > self.max_response_bytes):
                raise LocalModelError("invalid bounded model text")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LocalModelError("invalid local model response schema") from exc
        return UntrustedModelReply(self.model_id, _sha256(body), _sha256(raw), text)
