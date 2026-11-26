# Experimental JSON-schema output constraint (source-only)

Status: opt-in transport prototype; **NOT a revised v1 preregistration** and not a trained receiver.

The existing v1 runner and CLI still instantiate `LocalModelClient` with the default
`output_constraint="none"`; their request bytes, prompt profile, strict
`parse_plan` semantics, and feasibility thresholds are unchanged.

For standalone experimental client calls only, an OpenAI-compatible local
server can be passed `output_constraint="json_schema"`. The client sends
`response_format.type=json_schema` with a strict output object:
- `steps`: at most two items, each with a handle drawn only from the
  request's public candidate handles and an `initial|previous` input;
- `abstain`: boolean;
- no additional keys in the object or step items.

The schema helps enforce **format**, not correct reasoning, valid sequencing,
authorization, evidence trust, or task success. The existing independent
strict parser and host-effect gate retain those obligations.

The option is deliberately rejected for `ollama` requests rather than
silently falling back to an unconstrained request. Failures retain an
allowlisted `output_constraint=json_schema` diagnostic without model text.
Do not add this mode to v1 receipts or compare it with v1 outcomes as though
the receiver/generation configuration were unchanged.

## Preliminary host observation (2026-11-16)

On an already-running VN2 Granite 3.3 2B loopback llama-server, one tiny
JSON-object request and one tiny JSON-schema request each returned HTTP 200
and a parseable two-key abstention object in ~3.2 seconds (16 completion
tokens). The two observed text hashes were equal. These were **direct,
unblinded transport checks** with trivial prompts, not the v1 scenario suite,
not a valid nonabstaining typed plan, and not serving-process attestation.

## Required before any scored constrained-output experiment

Freeze a separate versioned protocol and new task suites before observing
task outcomes; fix endpoint, model digest, grammar, sampling, tokens, timeout,
test order, controls, stopping rule, and receipt schema. Prove stable local
request bytes and independently validate them where possible. Run the
nonabstaining plan-format feasibility gate before any N/P/V-style comparison.
The prior Granite and Qwen v1 `NOT_FEASIBLE` decisions remain unchanged.

Source tests verify the opt-in JSON body, retained default behavior,
rejection of unsupported modes and unsafe text, and typed failure receipts.
Neither these tests nor the two trivial host probes prove neural transfer.
