# Scenario-level pilot receipt analysis

This is a **standard-library, local-only analysis tool**, not a model runner or
an independently verified science result. It reads an existing
`../pilot.py run` receipt and produces a JSON summary. It performs no network
request, no training, no model load, and no data mutation.

From the repository root:

```bash
python3 experiments/real-model-sync/statistics/analyze.py /path/to/receipt.json
python3 -m unittest discover -s experiments/real-model-sync/statistics -p 'test_*.py' -v
```

## Rules enforced

- Validate the declared source class (`SCRIPTED_FIXTURE_TEST` or
  `LOCAL_HTTP_UNATTESTED`), experiment version, and evaluation scope. Current
  `neuro-sync-local-receiver-v1` receipts are explicitly validated; historical
  `v0` receipts remain readable as `LEGACY_V0_UNSTRATIFIED`. Relabeling an old
  v0 receipt cannot silently upgrade its evidence to v1.
- For **v1**, cross-check per-case reachability strata, the source-reported
  prepared P/H/V shared-state digests against per-call request digests, the
  empty-state N control, and distinct possible/impossible scoring totals.
  Successful *possible* cases must use nonabstaining plans; successful
  *impossible* cases must abstain. These are internal-consistency checks,
  not independent proof of oracle, suite, host, or model provenance.
- Reject absent, duplicated, unbalanced, out-of-arm, contradictory or
  re-counted result rows. Every scenario must have exactly one row for each
  arm that the receipt claims was evaluated.
- Count **one binary score per scenario and arm**. Individual tools, HTTP
  retries, and correlated arm outcomes are **not independent observations**.
- Report invalid model replies separately from unsuccessful valid plans.
- Pair V against P, H, N, R and D *only when both arms are present*.
  V1 pairs use **possible tool-solving scenarios only**; impossible-case
  correct abstention is reported separately rather than pooled into the
  planning comparison. V0 pairs retain their historical unstratified meaning.
  For each paired binary comparison, report win/loss discordance, paired
  difference, an uncorrected exact conditional two-sided binomial (McNemar)
  p-value, and a Holm familywise multiplicity-adjusted descriptive p-value.
  Zero discordances yields p=1, not evidence that methods are equivalent.
- Exclude X from task-accuracy comparisons: X scores attempted-but-denied
  permission effects, not goal completion. An abstention cannot score an X
  governance success without an attempted blocked effect.
- An incomplete arm set or fewer than 12 scenarios receives
  `EXPLORATORY_ONLY`; a full fixture receives
  `SOURCE_ONLY_FULL_LOCAL_FIXTURE` and even a full HTTP-source receipt receives
  only `FULL_LOCAL_HTTP_UNATTESTED`. A 12-case exploratory **prefix of a
  larger sealed v1 suite** remains exploratory even with all arms present.
- A separate `response_completeness` field flags invalid/missing model replies;
  a full schedule with failed requests is not a completed inference result.

The exact and Holm-adjusted p-values describe the observed paired synthetic scenarios; they
**are not** independently preregistered inference, model attestation,
or evidence for a generalized knowledge-transfer effect. Comparing several
arms and selecting scenarios after results still compromises interpretation
even if a multiplicity correction is displayed. Model identity,
independence, suite selection, and reviewer rights remain unverified.

In the current Neuro-Sync pilot, V, P and H receive *identical* model-facing
tool descriptions. Even a large V–N difference cannot isolate a unique
StateBus benefit; a V–P or V–H difference on a tiny subset may be random or
execution-order noise. The analyzer always returns
`statebus_specific_accuracy_advantage_established: false` and
`neural_transfer_established: false` because neither claim can be verified
from a self-reported local receipt alone.

In v1, pooled `observed_success_rate` may count correct abstentions on
unreachable goals; it is **not** a tool-planning success rate. Read
`arms.<task arm>.stratified_task_scores` for `possible_cases`,
`possible_plans_solved`, `possible_plan_success_rate`, `impossible_cases`,
`impossible_correct_abstentions`, and `impossible_correct_abstention_rate`.
An always-abstain scripted 12-case fixture earns zero of ten possible plans
and correctly abstains on two impossible goals. No model capability follows.

Unit tests use historical v0 JSON records and a real v1 pilot producer fixture,
with adversarial oracle-label, aggregate and digest-binding mutations and an
independently specified paired contingency calculation. They call no model.
