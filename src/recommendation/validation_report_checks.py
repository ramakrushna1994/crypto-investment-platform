"""
validation_report_checks.py

Post-run artifact validation helpers for walk-forward DAG reliability.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REPORT_SUFFIXES = (
    "walk_forward_summary.json",
    "walk_forward_splits.csv",
    "walk_forward_buckets.csv",
    "walk_forward_thresholds.csv",
    "walk_forward_model_compare.csv",
    "walk_forward_regimes.csv",
    "walk_forward_drift.csv",
)


def _utc(dt_value):
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)
    return None


def _summary_json_ok(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"parse_error={exc}"

    horizons = payload.get("horizons")
    if not isinstance(horizons, list):
        return False, "missing_or_invalid_horizons"
    return True, "ok"


def verify_walk_forward_reports(
    source_tables: Sequence[str],
    reports_dir: str = "/opt/airflow/files/reports",
    required_suffixes: Iterable[str] = DEFAULT_REPORT_SUFFIXES,
    max_age_hours: float = 48.0,
    **context,
):
    """
    Validate that walk-forward artifacts exist and are fresh for the current run.
    Raises ValueError if any required output is missing, stale, or invalid.
    """
    if not source_tables:
        raise ValueError("source_tables is required and cannot be empty.")

    report_root = Path(reports_dir)
    if not report_root.exists():
        raise ValueError(f"reports_dir does not exist: {report_root}")

    dag_run = context.get("dag_run")
    run_start = None
    if dag_run is not None:
        run_start = _utc(getattr(dag_run, "start_date", None))
        if run_start is None:
            run_start = _utc(getattr(dag_run, "queued_at", None))

    cutoff = run_start or (datetime.now(timezone.utc) - timedelta(hours=float(max_age_hours)))
    suffixes = tuple(required_suffixes)

    missing = []
    stale = []
    invalid = []

    for source_table in source_tables:
        safe_table = source_table.replace(".", "_")
        for suffix in suffixes:
            artifact = report_root / f"{safe_table}_{suffix}"
            if not artifact.exists():
                missing.append(str(artifact))
                continue

            mtime = datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                stale.append(f"{artifact} (mtime={mtime.isoformat()} cutoff={cutoff.isoformat()})")

            if suffix.endswith("walk_forward_summary.json"):
                ok, msg = _summary_json_ok(artifact)
                if not ok:
                    invalid.append(f"{artifact} ({msg})")

    if missing or stale or invalid:
        parts = ["Walk-forward report validation failed."]
        if missing:
            parts.append(f"missing={len(missing)}")
        if stale:
            parts.append(f"stale={len(stale)}")
        if invalid:
            parts.append(f"invalid={len(invalid)}")
        sample = []
        if missing:
            sample.append(f"missing_sample={missing[:3]}")
        if stale:
            sample.append(f"stale_sample={stale[:3]}")
        if invalid:
            sample.append(f"invalid_sample={invalid[:3]}")
        raise ValueError(" | ".join(parts + sample))

    logger.info(
        "Walk-forward report validation passed | source_tables=%s | cutoff_utc=%s | reports_dir=%s",
        list(source_tables),
        cutoff.isoformat(),
        str(report_root),
    )
    return {
        "status": "ok",
        "source_tables": list(source_tables),
        "reports_dir": str(report_root),
        "cutoff_utc": cutoff.isoformat(),
        "required_suffixes": list(suffixes),
    }

