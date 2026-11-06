# Reversible compact catalog — offline candidate, **not integrated**

The live Granite observations are informative but sparse: a 242-byte tiny
synthetic request produced first content in ~611 ms, a 796-byte catalog request
in ~9,835 ms, and the separate 1,471-byte non-streaming typed-plan request
timed out after 20 seconds. These profiles differ in content and protocol;
the three observations do **not** establish a causal token/latency curve.

This directory explores a narrower falsifiable proposition: **can the same
13-handle typed task data be represented in fewer bytes without dropping
operation meanings, randomized handles, probe data, decoys, output schema,
or post-plan authorization?** It does not send a request or modify `pilot.py`.

## Mechanism

The original prompt repeats each randomized 20-character capability handle in
both `candidate_handles` and the operation catalog. The alternate format
stores the handles once in `h` and interprets the corresponding positional
`e.d[i]` under `h[i]`. For operation catalogs, `e.d` carries one of 13
one-to-one reversible operation codes (`+2`, `+7`, `-4`, `*2`, `*3`, `neg`,
`sq`, `abs`, `%11`, `-2`, `+5`, `*4`, `min10`). The model sees an explicit,
same-across-arms syntax legend. For probe arms, each `e.d[i]` preserves
the exact observed outputs at inputs -3, 2 and 12, even if falsified.

`encode_task` checks exact pilot-v0 keys, `max_calls==2`, the output JSON
schema, 13 distinct opaque handles, and the supported state kinds;
`decode_task(encode_task(task)) == task` for all seven pilot arms. Unknown
state fields, out-of-catalog labels, missing/duplicate handles, and malformed
probe vectors are errors, not silently dropped. This is **semantic/schema
preservation by deterministic host code**, not a proof the language model
understands positional indices or equivalent code semantics.

## Repeatable local measurement

On a deterministic 13-handle fixture, counted as compact JSON with sorted
keys and no extraneous whitespace:

| Arm | Existing external-state bytes | Encoded state bytes | Existing full prompt messages | Alternate full prompt messages |
|---|---:|---:|---:|---:|
| N (no state) | 2 | 4 (`null`) | 779 | **870** |
| R (raw probes) | 626 | 142 | 1,509 | **1,014** |
| V/P/H/D/X (full mapping or decoy) | 481 | 86 | 1,312 | **984** |

The catalog-arm messages save **328 bytes (~25%)** versus the current
compact-profile messages on this fixture. R saves 495 bytes (~33%). **N
gets 91 bytes larger** because the fixed legend remains even without state;
omitting it in N would change prompt treatment between arms. This is a
size win for catalog data, **not verified runtime speed or task accuracy**.

Run offline:

```sh
PYTHONPATH=src python3 -m unittest discover -s \
  experiments/real-model-sync/compact_catalog -p 'test_*.py' -v
```

## Hostile limitations and next falsifiers

> **Index alignment is a new failure mode.** Reordering `h` without the
> positional data changes operation associations, and the codec alone cannot
> detect who tampered with a prompt. Treat the output as untrusted model-input
> material, never as an authenticated grant or a plan-binding mechanism.

> **Byte savings do not establish token savings.** `+2`, `%11`, and short
> indices may tokenize differently by model. The model might confuse
> positional indices or integer modulo at negatives despite host-equivalent
> decode. A future bounded real-model trial must pre-register output, time,
> and invalid-plan endpoints and compare matched arms and controls.

> **No execution integration is authorized by this prototype.** The model
> still needs host-side full-plan authorization, state revision binding,
> revocation, replay/expiry controls, and independent effect custody.
> `e=null` does not itself imply permission and `X` revocation is external.

This is intentionally an opt-in offline candidate, not a new default prompt
profile or evidence of neural transfer.
