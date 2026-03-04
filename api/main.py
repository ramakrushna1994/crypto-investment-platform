from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
import psycopg2

from src.config.settings import POSTGRES

app = FastAPI(
    title="AI Quant Investment API",
    description="REST API for accessing AI-generated investment signals.",
    version="1.0.0"
)

VALID_ASSET_CLASSES = ["crypto", "mutual_funds", "nifty50", "nifty_midcap", "nifty_smallcap"]
VALIDATION_SOURCE_TABLES = {
    "crypto": "silver.crypto_features_daily",
    "mutual_funds": "silver.mutual_funds_features_daily",
    "nifty50": "silver.nifty50_features_daily",
    "nifty_midcap": "silver.nifty_midcap_features_daily",
    "nifty_smallcap": "silver.nifty_smallcap_features_daily",
}

def get_db_connection():
    return psycopg2.connect(POSTGRES.dsn)


def get_available_columns(conn, schema: str, table: str):
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        return {row[0] for row in cur.fetchall()}


def _to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _summarize_regimes(regime_rows):
    by_regime = {}
    for row in regime_rows or []:
        regime = str(row.get("regime", "")).lower()
        if regime not in {"bull", "bear", "sideways"}:
            continue

        samples = _to_float(row.get("samples"))
        samples = samples if samples is not None and samples >= 0 else 0.0
        bucket = by_regime.setdefault(
            regime,
            {
                "samples": 0.0,
                "hit_num": 0.0,
                "hit_den": 0.0,
                "ret_num": 0.0,
                "ret_den": 0.0,
                "bench_num": 0.0,
                "bench_den": 0.0,
            },
        )
        bucket["samples"] += samples

        hit_rate = _to_float(row.get("hit_rate"))
        if hit_rate is not None and samples > 0:
            bucket["hit_num"] += hit_rate * samples
            bucket["hit_den"] += samples

        avg_return = _to_float(row.get("avg_return"))
        if avg_return is not None and samples > 0:
            bucket["ret_num"] += avg_return * samples
            bucket["ret_den"] += samples

        avg_benchmark = _to_float(row.get("avg_benchmark_return"))
        if avg_benchmark is not None and samples > 0:
            bucket["bench_num"] += avg_benchmark * samples
            bucket["bench_den"] += samples

    out = {}
    for regime in ("bull", "bear", "sideways"):
        bucket = by_regime.get(regime)
        if not bucket:
            continue
        out[f"{regime}_samples"] = int(round(bucket["samples"]))
        out[f"{regime}_hit_rate"] = (
            bucket["hit_num"] / bucket["hit_den"] if bucket["hit_den"] > 0 else None
        )
        out[f"{regime}_avg_return"] = (
            bucket["ret_num"] / bucket["ret_den"] if bucket["ret_den"] > 0 else None
        )
        out[f"{regime}_avg_benchmark_return"] = (
            bucket["bench_num"] / bucket["bench_den"] if bucket["bench_den"] > 0 else None
        )
    return out


def _load_validation_rows(asset_class: str, source_table: str, reports_dir: str):
    safe_table = source_table.replace(".", "_")
    summary_path = Path(reports_dir) / f"{safe_table}_walk_forward_summary.json"
    if not summary_path.exists():
        return [
            {
                "asset_class": asset_class,
                "source_table": source_table,
                "horizon": "n/a",
                "status": "missing_report",
            }
        ]

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            {
                "asset_class": asset_class,
                "source_table": source_table,
                "horizon": "n/a",
                "status": "report_parse_failed",
                "error": str(exc),
            }
        ]

    generated_utc = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    horizons = payload.get("horizons", [])
    if not horizons:
        return [
            {
                "asset_class": asset_class,
                "source_table": source_table,
                "horizon": "n/a",
                "status": "no_horizons",
                "report_generated_utc": generated_utc,
            }
        ]

    rows = []
    for horizon_data in horizons:
        summary = horizon_data.get("summary", {}) or {}
        drift_summary = horizon_data.get("drift_summary", {}) or {}
        regime_summary = _summarize_regimes(horizon_data.get("regime_rows", []))
        rows.append(
            {
                "asset_class": asset_class,
                "source_table": source_table,
                "horizon": horizon_data.get("horizon", "n/a"),
                "status": horizon_data.get("status", "unknown"),
                "selected_model_type": horizon_data.get("selected_model_type"),
                "recommended_min_prob": horizon_data.get("recommended_min_prob"),
                "splits_ran": summary.get("splits_ran"),
                "avg_roc_auc": summary.get("avg_roc_auc"),
                "avg_f1": summary.get("avg_f1"),
                "avg_brier": summary.get("avg_brier"),
                "avg_cagr_approx": summary.get("avg_cagr_approx"),
                "avg_sharpe_annualized": summary.get("avg_sharpe_annualized"),
                "avg_max_drawdown": summary.get("avg_max_drawdown"),
                "avg_hit_rate": summary.get("avg_hit_rate"),
                "drift_status": horizon_data.get("drift_status"),
                "drift_avg_psi": drift_summary.get("avg_psi"),
                "drift_avg_ks": drift_summary.get("avg_ks"),
                "drift_high_features": drift_summary.get("high_drift_features"),
                "drift_medium_features": drift_summary.get("medium_drift_features"),
                "drift_recent_start": drift_summary.get("recent_start"),
                "drift_recent_end": drift_summary.get("recent_end"),
                "report_generated_utc": generated_utc,
                **regime_summary,
            }
        )
    return rows

@app.get("/")
def read_root():
    return {"status": "online", "message": "Investment Signal API is running."}

@app.get("/api/v1/signals/{asset_class}")
def get_investment_signals(
    asset_class: str,
    limit: int = 50,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Fetch the latest investment signals for a given asset class.
    Valid asset classes: 'crypto', 'mutual_funds', 'nifty50', 'nifty_midcap', 'nifty_smallcap'
    """
    if asset_class not in VALID_ASSET_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid asset class. Must be one of {VALID_ASSET_CLASSES}")

    schema_name = "gold"
    raw_table = f"{asset_class}_investment_signals"
    table_name = f"{schema_name}.{raw_table}"
    
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        available_cols = get_available_columns(conn, schema_name, raw_table)
        base_cols = [
            "symbol",
            "trade_date",
            "signal_1y",
            "confidence_1y",
            "signal_5y",
            "confidence_5y",
            "combined_confidence",
            "risk_bucket",
            "risk_score",
            "suggested_position_pct",
            "expected_return_1y",
            "risk_adjusted_score",
            "var_95_1d",
            "cvar_95_1d",
        ]
        select_cols = [c for c in base_cols if c in available_cols]
        if not select_cols:
            raise HTTPException(status_code=500, detail=f"No readable columns found in {table_name}")

        confidence_filter = []
        if "confidence_1y" in available_cols:
            confidence_filter.append("confidence_1y >= %s")
        if "confidence_5y" in available_cols:
            confidence_filter.append("confidence_5y >= %s")
        if "combined_confidence" in available_cols:
            confidence_filter.append("combined_confidence >= %s")
        if not confidence_filter:
            confidence_filter = ["1=1"]

        params = [min_confidence] * (len(confidence_filter) if confidence_filter != ["1=1"] else 0) + [limit]
        query = f"""
            SELECT DISTINCT ON (symbol)
                {', '.join(select_cols)}
            FROM {table_name}
            WHERE {' OR '.join(confidence_filter)}
            ORDER BY symbol, trade_date DESC
            LIMIT %s
        """
        cur.execute(query, params)
        
        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        return {
            "asset_class": asset_class,
            "count": len(results),
            "data": results
        }
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@app.get("/api/v1/validation/summary")
def get_validation_summary(
    asset_class: str | None = Query(default=None),
    reports_dir: str = Query(default="/opt/airflow/files/reports"),
):
    """
    Return walk-forward validation summary, including drift and regime diagnostics.
    Optional filter: asset_class=crypto|mutual_funds|nifty50|nifty_midcap|nifty_smallcap
    """
    if asset_class is not None and asset_class not in VALID_ASSET_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid asset class. Must be one of {VALID_ASSET_CLASSES}")

    selected_tables = (
        [(asset_class, VALIDATION_SOURCE_TABLES[asset_class])]
        if asset_class is not None
        else list(VALIDATION_SOURCE_TABLES.items())
    )

    rows = []
    for selected_asset, source_table in selected_tables:
        rows.extend(_load_validation_rows(selected_asset, source_table, reports_dir))
    rows.sort(key=lambda r: (str(r.get("asset_class") or ""), str(r.get("horizon") or "")))

    return {
        "asset_class": asset_class,
        "count": int(len(rows)),
        "data": rows,
    }

