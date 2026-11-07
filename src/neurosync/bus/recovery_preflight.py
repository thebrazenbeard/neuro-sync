"""Read-only *structural* inspection of a paired plan-clock authority.

This is a recovery diagnostic, not an authority grant, a consistent two-file
snapshot, an anti-rollback witness, or a trusted-time source. In particular an
older but well-formed pair can pass every check below.
"""

from __future__ import annotations

import json
import math
import sqlite3
import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


# Exact SQLite master DDL emitted by Draft #56's paired consumption journal.
# Pinning this form deliberately refuses alternate conflict policies rather
# than trying to infer replay behavior from a column/PK shape alone.
_EXPECTED_JOURNAL_DDL = (
    "CREATE TABLE consumed_plan_grants ("
    "principal TEXT NOT NULL,target TEXT NOT NULL,"
    "session TEXT NOT NULL,grant_id TEXT NOT NULL,"
    "invocation_id TEXT NOT NULL,"
    "PRIMARY KEY(principal,target,session,grant_id))"
)


@dataclass(frozen=True)
class PairPreflight:
    classification: Literal["BLOCKED", "STRUCTURALLY_COMPATIBLE_UNATTESTED"]
    evidence_class: str
    authority_effect: str
    findings: tuple[str, ...]
    has_v2_grants: bool | None
    last_observed_epoch: float | None
    consumption_journal_status: Literal[
        "NOT_INSPECTED", "LEGACY_UNCHECKED", "REQUIRED_JOURNAL_MISSING",
        "INVALID_JOURNAL", "STRUCTURE_PRESENT_UNATTESTED",
    ]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)


def _assert_quiesced_rollback_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError("authority component missing")
    # A SQLite mode=ro connection can still CREATE -wal/-shm files when the
    # database header indicates WAL mode and its directory is writable.
    # Reject it *before* opening SQLite; immutable=1 would instead ignore
    # live WAL/locking and give a potentially stale inspection.
    with path.open("rb") as source:
        header = source.read(100)
    if len(header) < 100 or header[:16] != b"SQLite format 3\x00":
        raise ValueError("authority component has invalid SQLite header")
    if header[18] != 1 or header[19] != 1:
        raise ValueError("WAL or unsupported SQLite journaling mode blocked")
    # An existing hot journal or auxiliary WAL state needs a separately
    # quiesced recovery procedure. Never auto-clean up such artifacts.
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("authority component has active or residual journal artifacts")


def _read_only(path: Path) -> sqlite3.Connection:
    _assert_quiesced_rollback_file(path)
    # SQLite URI mode=ro prevents missing-file creation. Do not use immutable=1:
    # it bypasses locking and may ignore live WAL changes.
    connection = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True,
                                 isolation_level=None, timeout=2.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
            raise ValueError("authority component failed SQLite quick_check")
        return connection
    except BaseException:
        connection.close()
        raise


def _require_columns(db: sqlite3.Connection, table: str, names: set[str]) -> None:
    # Identifier names here are fixed source constants, never external input.
    actual = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
    missing = names - actual
    if missing:
        raise ValueError(f"required schema columns missing from {table}: {sorted(missing)}")


def inspect_clock_pair(authority_path: str | Path) -> PairPreflight:
    """Inspect paired files without creating, repairing or changing either.

    The result is diagnostic only: a structurally compatible older backup
    cannot be distinguished from current authoritative data using these two
    files alone. Callers must never use this result as an authorization token.
    """
    main = Path(authority_path)
    sidecar = Path(str(main) + ".clock.sqlite3")
    findings: list[str] = []
    has_grants: bool | None = None
    floor: float | None = None
    journal_status = "NOT_INSPECTED"
    try:
        with closing(_read_only(main)) as db:
            _require_columns(db, "plan_clock_binding",
                             {"singleton", "binding_uuid", "status"})
            _require_columns(db, "clock_guarded_plan_grants", {
                "grant_id", "principal", "target", "session", "state_key",
                "expires", "revoked", "plan_sha256", "planned_state_sha256",
                "plan_handler_artifact_sha256", "plan_handler_source_git_oid",
                "plan_consumed_invocation_id",
            })
            _require_columns(db, "attempts", {
                "invocation_id", "grant_id", "principal", "target", "session",
                "state_key", "status", "started", "finished", "state_sha256",
                "result_sha256", "handler_artifact_sha256", "source_git_oid",
                "receipt_sha256", "plan_sha256",
            })
            binding = db.execute(
                "SELECT binding_uuid,status FROM plan_clock_binding WHERE singleton=1"
            ).fetchone()
            if binding is None or binding["status"] != "READY":
                raise ValueError("main clock binding not READY")
            if not isinstance(binding["binding_uuid"], str) or not binding["binding_uuid"]:
                raise ValueError("main clock binding UUID malformed")
            has_grants = bool(db.execute(
                "SELECT 1 FROM clock_guarded_plan_grants LIMIT 1"
            ).fetchone())
            with closing(_read_only(sidecar)) as clock_db:
                _require_columns(clock_db, "clock_state",
                                 {"singleton", "binding_uuid", "last_observed_epoch"})
                stored = clock_db.execute(
                    "SELECT binding_uuid,last_observed_epoch FROM clock_state "
                    "WHERE singleton=1"
                ).fetchone()
                if stored is None or stored["binding_uuid"] != binding["binding_uuid"]:
                    raise ValueError("clock sidecar binding missing or mismatched")
                clock_columns = {row["name"] for row in clock_db.execute(
                    "PRAGMA table_info(clock_state)")}
                journal = clock_db.execute(
                    "SELECT type,sql FROM sqlite_master WHERE name='consumed_plan_grants'"
                ).fetchone()
                if "consumption_journal_ready" not in clock_columns:
                    journal_status = "LEGACY_UNCHECKED"
                    if journal is not None:
                        journal_status = "INVALID_JOURNAL"
                        raise ValueError("unmarked consumption journal is ambiguous")
                    findings.append("consumption_journal_unchecked_legacy_schema")
                else:
                    marker = clock_db.execute(
                        "SELECT consumption_journal_ready FROM clock_state "
                        "WHERE singleton=1").fetchone()
                    if marker is None or type(marker[0]) is not int or marker[0] != 1:
                        journal_status = "INVALID_JOURNAL"
                        raise ValueError("consumption journal marker not READY")
                    if journal is None:
                        journal_status = "REQUIRED_JOURNAL_MISSING"
                        raise ValueError("consumption_journal_missing_after_ready_marker")
                    journal_status = "INVALID_JOURNAL"
                    if journal["type"] != "table":
                        raise ValueError("consumption journal is not a table")
                    if journal["sql"] != _EXPECTED_JOURNAL_DDL:
                        raise ValueError("consumption journal DDL conflict policy unknown")
                    if clock_db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='trigger' "
                        "AND tbl_name='consumed_plan_grants' LIMIT 1"
                    ).fetchone() is not None:
                        raise ValueError("consumption journal trigger forbidden")
                    _require_columns(clock_db, "consumed_plan_grants", {
                        "principal", "target", "session", "grant_id", "invocation_id",
                    })
                    # #56's replay denial relies on this exact composite PK;
                    # column names alone would accept a duplicate-friendly
                    # replacement table and misleadingly claim compatibility.
                    journal_columns = {row["name"]: row for row in clock_db.execute(
                        "PRAGMA table_info(consumed_plan_grants)")}
                    replay_key = ("principal", "target", "session", "grant_id")
                    if any(journal_columns[name]["pk"] != position or
                           journal_columns[name]["notnull"] != 1
                           for position, name in enumerate(replay_key, 1)) or (
                               journal_columns["invocation_id"]["pk"] != 0 or
                               journal_columns["invocation_id"]["notnull"] != 1):
                        raise ValueError("consumption journal replay key malformed")
                    journal_status = "STRUCTURE_PRESENT_UNATTESTED"
                    findings.append("consumption_journal_custody_unattested")
                candidate = stored["last_observed_epoch"]
                if candidate is None:
                    if has_grants:
                        raise ValueError("clock floor missing with v2 grant history")
                    findings.append("no_clock_sample_yet")
                elif type(candidate) not in (float, int) or not math.isfinite(candidate):
                    raise ValueError("persisted clock floor malformed")
                else:
                    floor = float(candidate)
    except (sqlite3.Error, OSError, ValueError) as exc:
        findings.append(f"structural_check_failed:{type(exc).__name__}:{exc}")
        return PairPreflight(
            classification="BLOCKED",
            evidence_class="LOCAL_READ_ONLY_SOURCE_INSPECTION",
            authority_effect="NO_INTENTIONAL_AUTHORITY_DB_WRITE",
            findings=tuple(findings),
            has_v2_grants=has_grants,
            last_observed_epoch=None,
            consumption_journal_status=journal_status,
        )

    # A concurrent writer may switch journal mode after our initial checks.
    # Do not silently report compatibility if auxiliary artifacts remain.
    # This post-read check is still not an atomic quiescence guarantee.
    try:
        _assert_quiesced_rollback_file(main)
        _assert_quiesced_rollback_file(sidecar)
    except (OSError, ValueError) as exc:
        return PairPreflight(
            classification="BLOCKED",
            evidence_class="LOCAL_READ_ONLY_SOURCE_INSPECTION",
            authority_effect="NO_INTENTIONAL_AUTHORITY_DB_WRITE",
            findings=(f"postread_journal_check_failed:{type(exc).__name__}:{exc}",),
            has_v2_grants=has_grants,
            last_observed_epoch=None,
            consumption_journal_status=journal_status,
        )
    findings.extend((
        "preflight_requires_quiesced_rollback_journal_mode",
        "preopen_header_and_journal_checks_race_with_external_writers",
        "two_file_read_not_atomic_with_concurrent_writers",
        "historical_same_uuid_restore_not_detectable",
        "main_only_consumption_or_revocation_rollback_not_detectable",
        "not_authorization_or_independent_witness",
    ))
    return PairPreflight(
        classification="STRUCTURALLY_COMPATIBLE_UNATTESTED",
        evidence_class="LOCAL_READ_ONLY_SOURCE_INSPECTION",
        authority_effect="NO_INTENTIONAL_AUTHORITY_DB_WRITE",
        findings=tuple(findings),
        has_v2_grants=has_grants,
        last_observed_epoch=floor,
        consumption_journal_status=journal_status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only structural inspection; never proves backup freshness"
    )
    parser.add_argument("authority_db", help="path to the main authority SQLite file")
    args = parser.parse_args(argv)
    result = inspect_clock_pair(args.authority_db)
    print(result.to_json())
    return 0 if result.classification == "STRUCTURALLY_COMPATIBLE_UNATTESTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
