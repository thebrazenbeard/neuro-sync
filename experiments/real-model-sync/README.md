# Local real-receiver pilot — experimental, not yet qualified

This is a **CPU-light, standard-library-only, optional harness** for evaluating
an actual local inference engine's ability to interpret external capability
descriptions. It does not train either model or require a hosted paid API.

The donor is still a **scripted synthetic capability catalog**, not another
neural model. A live run therefore measures *receiver use of external state*,
not inter-neural knowledge transfer.

## Steps

From repository root, when an already-running local Ollama or OpenAI-compatible
server is idle and authorized to receive requests:

```bash
python experiments/real-model-sync/pilot.py freeze --suite /tmp/ns-local-pilot-suite.json
python experiments/real-model-sync/pilot.py run \
  --suite /tmp/ns-local-pilot-suite.json \
  --out-dir /tmp/ns-local-pilot-run \
  --backend openai --endpoint http://127.0.0.1:11435 \
  --model YOUR_EXACT_LOCAL_MODEL_ID \
  --timeout 30 --max-tokens 96 --prompt-profile compact
python -m unittest discover -s experiments/real-model-sync -p 'test_*.py' -v
```

For a small bounded host smoke check before committing to 84 potential model
requests, append `--max-cases 1 --arms N,V,P,H` to `run`. The receipt will say
`EXPLORATORY_INCOMPLETE`: this is not an acceptance trial and may be biased by
its first scenario. Full 12×7 comparisons require explicit workload readiness.
Only requested arms create their local test artifacts; an `--arms N` smoke
does not unnecessarily create a P file or V SQLite authority fixture.

### Bounded latency and prompt profiles

`--prompt-profile compact` (new default) sends only the puzzle start, goal,
all 13 opaque handles and the exact arm-specific external state. A common
system message still specifies a maximum of two ordered tool calls, explicit
dependency fields, and optional abstention when the solution is uncertain.
Crucially N (no state) may still guess a listed opaque handle; the compact
instruction does **not** mandate abstention simply because external state is
absent. The same
profile, token cap and timeout are applied **identically to every arm**. This
removes redundant task/schema prose but does **not** shorten any arm's data,
reveal private scenario labels or change oracle/grant execution.

`--prompt-profile legacy` retains the original longer v0 prompt as a separate
configuration for comparisons. Do not pool the two profiles as if they were
identical experimental treatments or attribute performance differences purely
to prompt bytes: the system wording and few-shot shape also differ. For one
generated scenario, prompt-message JSON was 1016 vs 778 bytes (N), 1746 vs
1508 (R), and 1549 vs 1311 (V); exact sizes vary with scenario content.
`--max-tokens` accepts 32–256 tokens and
sets OpenAI-compatible `max_tokens` or Ollama `options.num_predict`;
`--timeout` accepts up to 60 seconds per request. The default is 96 output
tokens and 30 seconds. Use a fixed profile/budget for a full trial and record
machine load. Higher timeouts can cost local CPU time, not cloud fees when the
configured endpoint is genuinely local. No service lifecycle is changed.
At the 32-token minimum, a correct two-step JSON reply may truncate; treat
that as a request-budget/invalid-reply outcome, not reasoning incapacity.

Each completed model adapter call records request byte count, prompt digest,
profile, configured token cap/timeout, adapter latency and response digest.
Even on a transport or parse failure the adapter records its prompt digest,
request bytes, latency and failure class. The harness additionally measures
`model_call_elapsed_ms` on the caller side independently of untrusted adapter
metadata. Failures execute **zero** tools and do not count as governance
success. An elapsed-time receipt does not attest the server's actual token
generation, GPU/CPU use, or model identity.
Ambiguous model JSON with duplicate keys (including nested plan-step fields)
is rejected as an invalid reply rather than silently accepting the last value.

The loader now recomputes whether each frozen goal is reachable in one or two
tool calls, rejecting mismatched `expected_possible` labels even if a modified
file carries a newly computed `.sha256` sidecar. This protects against that
specific scoring false positive, **not** a malicious but mathematically
consistent suite or third-party custody failure. The runner rejects a forged
`real_endpoint` declaration when the caller is not the exact local HTTP client
adapter; recognizing the adapter still does **not** authenticate a real neural
serving process. Arm order now follows an ex ante 7-position cyclic schedule,
with position counts differing by at most one across 12 cases.

**Known negative baseline:** A prior exploratory VN2 Granite 3.3 2B Q4_K_M
run using the legacy prompt and a 12-second per-call timeout returned four
`TimeoutError` failures across N/V/P/H and produced no plans. This is a
source-reported exploratory runtime-feasibility failure; it is neither
receiver-learning evidence nor a reason to silently omit those arms. The
compact profile and lower token cap are **source-mechanics improvements**;
a subsequent single 1,471-byte compact V request on the existing VN2 Granite
service still timed out at 20 seconds without returning a typed plan. Separate
streaming diagnostic probes against that service completed for a tiny 242-byte
prompt (1.28 seconds) and a 796-byte catalog prompt (10.65 seconds), both
with an eight-token output cap. Those probes are not comparable to the
structured-plan trial or evidence of semantic success; see the
[execution-status ledger](../../docs/EXPERIMENT_STATUS.md) for exact hashes,
source version and limitations. No general speedup is claimed.

`freeze` creates a private synthetic suite and `.sha256` self-commitment;
it will never overwrite an existing suite. Keep the private suite file out
of Git and away from the receiver model's file tools. This is **not** an
independently timestamped preregistration or separate OS custody.

Use `--model-file /path/to/unchanged-model.gguf` to hash one supplied local
file before and after the run. This proves only that the *supplied file* did
not change; it does not authenticate that the serving inference process loaded
those bytes. Serving model weight immutability remains **UNKNOWN** until host
process and file provenance are independently bound.
The runner does not start, unload, or restart the model service. Fresh requests
are **not** a verified cold model restart; separate restart evidence is needed.

The CLI accepts only existing loopback HTTP endpoints and never starts a service.
It disables inherited HTTP proxies and rejects redirects. No cloud API,
network download, or private training data is needed; the runner cannot
independently prove that the selected local server does not forward requests
to a paid remote backend, so verify the host configuration first.

## Controls and scoring

Each of 12 independently generated synthetic scenarios supplies 13 opaque tool
handles and an integer transformation goal. The model must output typed JSON
tool steps and explicit `initial`/`previous` dependencies, or abstain. The
host executes the chosen tools and compares each effect to a separately coded
oracle. **One binary primary score per scenario and arm** (not per step);
X's primary is *effect denial* at a deliberately post-decision revocation,
separately labeled from ordinary task success. Abstaining before an attempted
call does NOT count as gate enforcement, even if no effect happened:

| Arm | Extra state available to receiver |
|---|---|
| N | None; model can still submit a plan |
| R | Three discriminating raw input/output probes per handle |
| V | Verified-by-host *fixture* description map via actual StateLedger commits |
| P | The same description map as an independent ordinary JSON file |
| H | Direct host-only description map, no donor needed |
| D | A deliberately incorrect description map |
| X | The description map, but all host effect grants revoked at use time |

P is constructed without any bus read and H is prepared before V. The
X grant is an in-process `ToolGrant` test, not a test of the separate
HostOwnedEffectBroker or an OS authorization boundary. The
description map exposes tool semantics, **not a selected plan or expected
answer**. Host-fixture StateBus validation persists actual bounded tool-effect
probe receipts and a canonical SHA-256 source-file manifest; its Git HEAD OID
does not prove the working tree is clean or independently witnessed. Arm order
is deterministically counterbalanced by scenario index;
each model call has fresh isolated chat messages. Observed outputs/latency and
prompt/response hashes are retained in `receipt.json` with exact source digest.

## Claim boundaries

Local test suite uses a **scripted fake receiver**, always marked
`SCRIPTED_FIXTURE_TEST`, and MUST NOT be reported as live-model evidence. An
HTTP server run is labeled `LOCAL_HTTP_UNATTESTED` until the serving process,
model file, and request trace are independently bound; loopback alone does not
prove a particular neural model answered. It does not establish model
training, donor neural knowledge, independently enforced rights, honest ex ante
randomization, cross-model semantics, or cold restart persistence. The local
ledger's `independent-verifier` string is a schema role marker, not an
independent security principal. Same-user private suite files remain accessible
to processes with that user's filesystem privileges.

No success threshold is retrofitted to a result. Compare arms per scenario;
P/H parity is a plausible outcome, not a failure of the protocol implementation.
