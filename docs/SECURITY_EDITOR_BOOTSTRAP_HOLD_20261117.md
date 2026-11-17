# Neuro-Sync editor-bootstrap SECURITY HOLD — 2026-11-17

Status: **SOURCE SECURITY HOLD / NOT HOST-COMPROMISE PROOF**.

## Observed, exact objects

- Safe original Neuro-Sync main: `b19186216f111fa0d27a6c23c68a1f988c4eed5c`.
- Previously verified PR #70 head: `86f9419143b144e23be0a07b00d797a181b7b46c`.
- Previously verified PR #71 head: `fcf354c161d0dd1c76fe1d61de3aea542f1fbe32`.
- Unexpected replacement #70 head: `a483505edf1c6255f379e539bbe9fc3fbc3127ac`.
- Unexpected replacement #71 head: `66cc121f06c3838d9feea5e5f79d66b7e0fa4c99`.

Live GitHub commit, PR-file and compare API readbacks show unrelated
`.vscode/ini` (large obfuscated JavaScript), `.vscode/tasks.json`
(auto-run `folderOpen` task invoking Node on the `ini` file), and
`.vscode/settings.json` enabling automatic tasks on both replacement
branches. The changed `.gitignore` no longer includes protective
`.env` and `.env.*` patterns. Git commit author strings and unsigned
status do not authenticate who caused this change.

**No one should open these branches in an editor, merge them, or run
their scripts.** Do not execute or attempt to unpack the suspicious
file on a connected workstation. The prior original safe commits are
preserved as separate Git objects; no force push or evidence erasure
has been performed.

## Actions taken

- Posted explicit SECURITY HOLD comments to PR #70 and PR #71.
- Continued from the checked, clean, previously recorded PR #71
  commit `fcf354c`, on a new independent branch. Did not fetch,
  checkout or execute the replacement heads.
- Added a repo-specific stdlib-only checkout guard that rejects all
  unreviewed `.vscode` files except inert `extensions.json`, missing
  dotenv ignores and explicit `!.env` / `!.env.*` unignores.
- Installed the guard as the first post-checkout step in the safe
  GitHub Actions source-mechanics workflow.
- Added fixture-based hostile tests including an inert copy of the
  folder-open task shape; **the actual obfuscated payload is not
  embedded, downloaded or run by these tests**.

## Remaining limitations

This detects the known repository-file pattern in **a trusted
workflow**, not arbitrary code execution or every editor extension.
An attacker who can rewrite the PR branch and its workflow can also
remove the guard; therefore green CI on the suspicious branches is
not an independent trust attestation.

Actual payload behavior, whether an editor ever opened the
replacement branch, the actor/credential behind the branch rewrite,
and any host-level impact remain **UNKNOWN**. Investigate GitHub
push/audit history and authorized access before accepting changes.
If there is evidence the editor task executed, isolate affected
workstations and perform credential/host review by an authorized
operator. Do not equate a source diff with confirmed compromise.

The clean replacement is offered for source review only.
No protected merge, branch deletion, secret rotation, host service
restart, or trust-model change was authorized or performed here.
