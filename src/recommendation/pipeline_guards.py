"""
pipeline_guards.py

Runtime guardrails for daily production flow:
- freshness SLA on feature tables
- promotion gate on walk-forward validation metrics
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterable

import psycopg2

from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN check
        return None
    return out


def _utc_now():
    return datetime.now(timezone.utc)


def _as_utc(dt):
    if dt is None:
        return None
    if isinstance(dt, date) and not isinstance(dt, datetime):
        return datetime.combine(dt, time.min, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_horizons(required_horizons: Iterable[str] | str | None):
    if required_horizons is None:
        return ["1y"]
    if isinstance(required_horizons, str):
        items = [p.strip().lower() for p in required_horizons.split(",") if p.strip()]
        return items or ["1y"]
    items = [str(x).strip().lower() for x in required_horizons if str(x).strip()]
    return items or ["1y"]


def assert_source_freshness_sla(
    source_table: str,
    max_age_hours: float = 72.0,
    min_rows: int = 100,
    timestamp_column: str = "event_time",
    **context,
):
    """
    Fail-fast if source feature table is stale or too small.
    """
    conn = None
    try:
        conn = psycopg2.connect(POSTGRES.dsn)
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT COUNT(*)::bigint AS row_count,
                   MAX({timestamp_column}) AS max_event_time
            FROM {source_table}
            """
        )
        row_count, max_ts = cur.fetchone()
        cur.close()

        row_count = int(row_count or 0)
        if row_count < int(min_rows):
            raise ValueError(
                f"Freshness SLA failed for {source_table}: row_count={row_count} < min_rows={int(min_rows)}"
            )

        if max_ts is None:
            raise ValueError(f"Freshness SLA failed for {source_table}: max_{timestamp_column}=NULL")

        max_dt = _as_utc(max_ts)
        age_hours = (_utc_now() - max_dt).total_seconds() / 3600.0
        if age_hours > float(max_age_hours):
            raise ValueError(
                f"Freshness SLA failed for {source_table}: age_hours={age_hours:.2f} > max_age_hours={float(max_age_hours):.2f}"
            )

        logger.info(
            "Freshness SLA passed | table=%s | rows=%s | max_%s=%s | age_hours=%.2f | max_age_hours=%.2f",
            source_table,
            row_count,
            timestamp_column,
            max_dt.isoformat(),
            age_hours,
            float(max_age_hours),
        )
        return {
            "status": "ok",
            "source_table": source_table,
            "row_count": row_count,
            "max_event_time_utc": max_dt.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": float(max_age_hours),
        }
    finally:
        if conn is not None:
            conn.close()


def assert_model_promotion_gate(
    source_table: str,
    reports_dir: str = "/opt/airflow/files/reports",
    required_horizons: Iterable[str] | str | None = None,
    min_avg_roc_auc: float | None = None,
    min_avg_f1: float | None = None,
    max_high_drift_features: int | None = None,
    max_report_age_days: float = 10.0,
    require_drift_ok: bool | None = None,
    **context,
):
    """
    Promotion guard before publishing new signals.
    Blocks publish when latest validation report quality is below thresholds.
    """
    min_avg_roc_auc = (
        float(min_avg_roc_auc)
        if min_avg_roc_auc is not None
        else float(os.getenv("AIQ_PROMOTION_MIN_AVG_ROC_AUC", "0.54"))
    )
    min_avg_f1 = (
        float(min_avg_f1)
        if min_avg_f1 is not None
        else float(os.getenv("AIQ_PROMOTION_MIN_AVG_F1", "0.40"))
    )
    max_high_drift_features = (
        int(max_high_drift_features)
        if max_high_drift_features is not None
        else int(os.getenv("AIQ_PROMOTION_MAX_HIGH_DRIFT_FEATURES", "10"))
    )
    if require_drift_ok is None:
        require_drift_ok = os.getenv("AIQ_PROMOTION_REQUIRE_DRIFT_OK", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    horizons_required = _parse_horizons(required_horizons)
    safe_table = source_table.replace(".", "_")
    summary_path = Path(reports_dir) / f"{safe_table}_walk_forward_summary.json"
    if not summary_path.exists():
        raise ValueError(f"Promotion gate failed for {source_table}: missing report {summary_path}")

    mtime = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc)
    age_days = (_utc_now() - mtime).total_seconds() / 86400.0
    if age_days > float(max_report_age_days):
        raise ValueError(
            f"Promotion gate failed for {source_table}: report_age_days={age_days:.2f} > max_report_age_days={float(max_report_age_days):.2f}"
        )

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Promotion gate failed for {source_table}: report parse error ({exc})") from exc

    horizons = payload.get("horizons", [])
    by_horizon = {str(h.get("horizon")).lower(): h for h in horizons if isinstance(h, dict)}

    evaluation = []
    for horizon in horizons_required:
        item = by_horizon.get(horizon)
        if item is None:
            raise ValueError(f"Promotion gate failed for {source_table}: missing horizon='{horizon}' in report")

        status = str(item.get("status", "unknown")).lower()
        if status != "ok":
            raise ValueError(
                f"Promotion gate failed for {source_table} [{horizon}]: status={status} (must be ok)"
            )

        summary = item.get("summary", {}) or {}
        roc_auc = _safe_float(summary.get("avg_roc_auc"))
        f1 = _safe_float(summary.get("avg_f1"))
        if roc_auc is None or roc_auc < min_avg_roc_auc:
            raise ValueError(
                f"Promotion gate failed for {source_table} [{horizon}]: avg_roc_auc={roc_auc} < min_avg_roc_auc={min_avg_roc_auc}"
            )
        if f1 is None or f1 < min_avg_f1:
            raise ValueError(
                f"Promotion gate failed for {source_table} [{horizon}]: avg_f1={f1} < min_avg_f1={min_avg_f1}"
            )

        drift_status = str(item.get("drift_status", "unknown")).lower()
        if require_drift_ok and drift_status != "ok":
            raise ValueError(
                f"Promotion gate failed for {source_table} [{horizon}]: drift_status={drift_status} (must be ok)"
            )

        drift_summary = item.get("drift_summary", {}) or {}
        high_drift = _safe_float(drift_summary.get("high_drift_features"))
        if high_drift is not None and high_drift > float(max_high_drift_features):
            raise ValueError(
                f"Promotion gate failed for {source_table} [{horizon}]: high_drift_features={high_drift} > max_high_drift_features={max_high_drift_features}"
            )

        evaluation.append(
            {
                "horizon": horizon,
                "avg_roc_auc": roc_auc,
                "avg_f1": f1,
                "drift_status": drift_status,
                "high_drift_features": high_drift,
            }
        )

    logger.info(
        "Promotion gate passed | table=%s | report=%s | report_age_days=%.2f | thresholds={auc>=%.3f,f1>=%.3f,max_high_drift<=%s,require_drift_ok=%s} | eval=%s",
        source_table,
        str(summary_path),
        age_days,
        min_avg_roc_auc,
        min_avg_f1,
        max_high_drift_features,
        require_drift_ok,
        evaluation,
    )
    return {
        "status": "ok",
        "source_table": source_table,
        "report_path": str(summary_path),
        "report_age_days": age_days,
        "required_horizons": horizons_required,
        "evaluation": evaluation,
    }
