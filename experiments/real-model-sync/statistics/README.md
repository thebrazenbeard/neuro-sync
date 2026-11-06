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
  `LOCAL_HTTP_UNATTESTED`), experiment version, and evaluation scope.
- Reject absent, duplicated, unbalanced, out-of-arm, contradictory or
  re-counted result rows. Every scenario must have exactly one row for each
  arm that the receipt claims was evaluated.
- Count **one binary score per scenario and arm**. Individual tools, HTTP
  retries, and correlated arm outcomes are **not independent observations**.
- Report invalid model replies separately from unsuccessful valid plans.
- Pair V against P, H, N, R and D *only when both arms are present*.
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
  only `FULL_LOCAL_HTTP_UNATTESTED`.
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

The only unit tests here use fully synthetic JSON records and an independently
specified paired contingency calculation. They do not call a language model.
