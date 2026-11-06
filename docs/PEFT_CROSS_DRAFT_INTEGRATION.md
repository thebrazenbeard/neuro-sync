# Neuro-Sync Path A — source-only cross-draft integration checkpoint

This local candidate combines **unmerged** code from the following exact
sources on top of `neuro-sync/main@6db4b304d56b9e27d8d95cb9eb7bf03fe7a24c65`:

- Draft #37 `31319af108ffe5289c4f38305535e307c3b05409`:
  `trainingday_proposal.py`, its test and proposal design document.
- Draft #38 `a7468b748fddb3cd04028fefc37bf09329548397`:
  POSIX no-follow `peft_preflight.py`, its regression tests and document.
- Draft #48 `a0765eb3be9c117183207249f8eb4f63958fd17b`:
  `peft_proposal_binding.py`, its test and binder document.
- A new local `test_cross_peft_realbytes_bridge.py` verifies that actual
  synthetic file-byte SHA-256 digests flow through inspector, transfer intent,
  envelope and binder without equating a LoRA adapter with a trained full
  donor checkpoint. A coherent malicious alias is rejected.

The source-level binding is not an authenticated receipt. Its successful
result still has `physical_launch_authority=HOLD` and
`trainingday_ready=false`. There was **no** optimizer run, actual model
loading, independent data-custody check, rights approval, or retained behavior
study.

## Reproducible tests from repository root

```bash
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
PYTHONPATH=src python3 -O -B -m unittest discover -s tests -q
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_*peft*py' -v
PYTHONPATH=src python3 -B -m unittest discover -s tests -p 'test_trainingday_proposal.py' -v
```

Observed locally on this exact combined tree: **159/159 root-library tests
passed**, zero skipped, normal and `-O`. The formerly conditional cross-PR
integration test in Draft #48 actually ran. All eight experiment test
directories in the current source-mechanics CI also passed independently
(170 test cases; the toy-transfer suite reports six existing skips). CI now
includes dedicated PEFT and TRAININGDAY commands so tests outside
`test_core_*.py` are not accidentally omitted.

`python -m unittest discover -s .` from the repository root currently
collects **zero tests** because source test directories are not nested import
packages. The specified `discover -s tests` command is the working root
invocation. We do not describe the zero-test command as a pass.

## Merge and custody boundaries

This candidate is a local integration experiment, not a published PR, merge,
installation or authorization. An independent source review must reconcile
overlapping draft branches and update from exact live main before publication.
Keep the component Draft PRs open until their owners agree on one canonical
integration path. Do not merge them independently by assuming CI can resolve
semantic role conflicts. Windows secure artifact inspection remains HOLD.
