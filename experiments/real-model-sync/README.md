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
  --model YOUR_EXACT_LOCAL_MODEL_ID
python -m unittest discover -s experiments/real-model-sync -p 'test_*.py' -v
```

For a small bounded host smoke check before committing to 84 potential model
requests, append `--max-cases 1 --arms N,V,P,H` to `run`. The receipt will say
`EXPLORATORY_INCOMPLETE`: this is not an acceptance trial and may be biased by
its first scenario. Full 12×7 comparisons require explicit workload readiness.

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
description map exposes tool semantics, **not a selected plan or expected
answer**. Host-fixture StateBus validation persists actual bounded tool-effect
probe receipts and a canonical SHA-256 source-file manifest; its Git HEAD OID
does not prove the working tree is clean or independently witnessed. Arm order
is deterministically counterbalanced by scenario index;
each model call has fresh isolated chat messages. Valid replies retain the
bounded parsed `selected_handles` and SHA-256 of the exact model text, but not
the raw reply; local HTTP adapter traces retain observed latency, prompt hash,
and HTTP response-envelope hash in `receipt.json` with the source digest.
Invalid replies record a failure and execute no effects; they do not retain
their raw text for later forensic reconstruction. Non-text, unencodable,
oversize, duplicate-key, and over-deep JSON replies are rejected before
execution. A matched P/H/V comparator
divergence fails explicitly even under optimized (`python -O`) execution.

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
