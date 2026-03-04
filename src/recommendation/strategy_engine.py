"""
strategy_engine.py

Loads investor-horizon models and generates signal outputs with:
- calibrated confidence (if calibrator is available)
- risk-aware ranking signals for portfolio sizing
"""
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.config.settings import POSTGRES, validate_table_name

logger = logging.getLogger(__name__)

# Investor-grade signal thresholds for each horizon.
SIGNAL_THRESHOLDS = {
    "1y": [
        (0.72, "INVEST NOW"),
        (0.58, "ACCUMULATE"),
        (0.42, "MONITOR"),
        (0.25, "WAIT"),
        (0.00, "AVOID"),
    ],
    "5y": [
        (0.70, "STRONG HOLD"),
        (0.55, "ACCUMULATE"),
        (0.40, "MONITOR"),
        (0.20, "WAIT"),
        (0.00, "AVOID"),
    ],
}

FEATURES = [
    "rsi_14",
    "volatility_7d",
    "macd",
    "macd_signal",
    "ema_20",
    "moving_avg_7d",
    "bb_upper",
    "bb_lower",
    "sma_50",
    "sma_200",
    "atr_14",
    "stoch_k",
    "stoch_d",
]

NAV_FEATURES = [
    "rolling_return_30d",
    "rolling_return_90d",
    "sortino_30d",
    "max_drawdown_30d",
    "nav_momentum_14d",
]

# Normal-loss constants for 95% one-day downside metrics.
Z_SCORE_95 = 1.6448536269514722
ALPHA_95 = 0.05
PHI_Z95 = float(np.exp(-0.5 * (Z_SCORE_95 ** 2)) / np.sqrt(2.0 * np.pi))


def _prob_to_signal(prob, horizon):
    for threshold, label in SIGNAL_THRESHOLDS[horizon]:
        if prob >= threshold:
            return label
    return SIGNAL_THRESHOLDS[horizon][-1][1]


def _safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        if model.classes_[0] == 1:
            return probs[:, 0]
        return np.zeros(probs.shape[0], dtype=float)
    return probs[:, 1]


def _load_model_artifact(path: str) -> dict:
    payload = joblib.load(path)
    if isinstance(payload, dict) and ("model" in payload or "models" in payload):
        payload.setdefault("calibrator", None)
        payload.setdefault("metadata", {})
        payload.setdefault("model_kind", "single" if "model" in payload else "ensemble")
        return payload
    # Backward compatibility: older jobs saved the raw estimator directly.
    return {
        "model_kind": "single",
        "model": payload,
        "calibrator": None,
        "metadata": {"legacy_artifact": True},
    }


def _predict_prob_from_artifact(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    model_kind = str(artifact.get("model_kind", "single"))
    if model_kind == "ensemble":
        weighted_probs = np.zeros(len(X), dtype=float)
        total_weight = 0.0
        for component in artifact.get("models", []):
            model = component["model"]
            raw_prob = _safe_predict_proba(model, X)
            calibrator = component.get("calibrator")
            if calibrator is None:
                calibrated = np.clip(raw_prob, 0.0, 1.0)
            else:
                px = np.clip(raw_prob, 1e-6, 1 - 1e-6).reshape(-1, 1)
                calibrated = np.clip(calibrator.predict_proba(px)[:, 1], 0.0, 1.0)

            weight = float(component.get("weight", 1.0))
            weighted_probs += calibrated * weight
            total_weight += weight

        if total_weight <= 0:
            return np.zeros(len(X), dtype=float)
        return np.clip(weighted_probs / total_weight, 0.0, 1.0)

    model = artifact["model"]
    raw_prob = _safe_predict_proba(model, X)
    calibrator = artifact.get("calibrator")
    if calibrator is None:
        return np.clip(raw_prob, 0.0, 1.0)

    x = np.clip(raw_prob, 1e-6, 1 - 1e-6).reshape(-1, 1)
    calibrated = calibrator.predict_proba(x)[:, 1]
    return np.clip(calibrated, 0.0, 1.0)


def _compute_risk_fields(row: pd.Series, p1y: float, p5y: float):
    close = float(row.get("close") or 0.0)
    vol_7d = float(row.get("volatility_7d") or 0.0)
    atr_14 = float(row.get("atr_14") or 0.0)
    rsi_14 = float(row.get("rsi_14") or 50.0)
    sma_50 = float(row.get("sma_50") or 0.0)
    sma_200 = float(row.get("sma_200") or 0.0)

    vol_norm = float(np.clip(vol_7d / 0.06, 0.0, 1.0))
    atr_pct = float((atr_14 / close) if close > 0 else 1.0)
    atr_norm = float(np.clip(atr_pct / 0.08, 0.0, 1.0))
    rsi_extreme = float(np.clip(abs(rsi_14 - 50.0) / 50.0, 0.0, 1.0))

    trend_ratio = float((sma_50 - sma_200) / abs(sma_200)) if sma_200 != 0 else 0.0
    trend_penalty = float(np.clip((-trend_ratio) / 0.30, 0.0, 1.0))

    risk_raw = 0.40 * vol_norm + 0.30 * atr_norm + 0.15 * rsi_extreme + 0.15 * trend_penalty
    risk_raw = float(np.clip(risk_raw, 0.0, 1.0))
    risk_score = risk_raw * 100.0

    if risk_score < 35:
        risk_bucket = "LOW"
    elif risk_score < 65:
        risk_bucket = "MEDIUM"
    else:
        risk_bucket = "HIGH"

    combined_prob = float(np.clip(0.65 * p1y + 0.35 * p5y, 0.0, 1.0))
    conviction = float(np.clip((combined_prob - 0.50) / 0.50, 0.0, 1.0))
    suggested_position_pct = float(np.clip(0.01 + 0.11 * conviction * (1 - 0.75 * risk_raw), 0.01, 0.12))

    expected_return_1y = float(combined_prob * 0.10 + (1.0 - combined_prob) * (-0.06))
    risk_adjusted_score = float(expected_return_1y * (1.0 - risk_raw))
    expected_return_1d = float(expected_return_1y / 252.0)

    # 1-day volatility proxy combining observed volatility and ATR-based proxy.
    atr_vol_proxy = float(atr_pct / np.sqrt(14.0)) if atr_pct > 0 else 0.0
    sigma_1d = float(np.clip(max(vol_7d, atr_vol_proxy, 1e-4), 1e-4, 0.50))

    # Parametric 95% normal-loss VaR/CVaR scaled by suggested position size.
    tail_var = max(0.0, (Z_SCORE_95 * sigma_1d) - expected_return_1d)
    tail_cvar = max(0.0, ((sigma_1d * (PHI_Z95 / ALPHA_95)) - expected_return_1d))
    var_95_1d = float(np.clip(tail_var * suggested_position_pct, 0.0, 1.0))
    cvar_95_1d = float(np.clip(max(tail_cvar, tail_var) * suggested_position_pct, 0.0, 1.0))

    return {
        "combined_confidence": combined_prob,
        "risk_score": risk_score,
        "risk_bucket": risk_bucket,
        "suggested_position_pct": suggested_position_pct,
        "expected_return_1y": expected_return_1y,
        "risk_adjusted_score": risk_adjusted_score,
        "var_95_1d": var_95_1d,
        "cvar_95_1d": cvar_95_1d,
    }


def generate_signals(
    source_table="silver.mutual_funds_features_daily",
    dest_table="gold.mutual_funds_investment_signals",
):
    """Fetch latest features, run inference models, and upsert signals table."""
    source_table = validate_table_name(source_table)
    dest_table = validate_table_name(dest_table)
    
    logger.info(f"Generating signals: {source_table} -> {dest_table}")

    clean_name = source_table.split(".")[-1] if "." in source_table else source_table
    artifacts = {}
    for horizon in ["1y", "5y"]:
        path = f"/opt/airflow/models/{clean_name}_rf_{horizon}_model.joblib"
        if not os.path.exists(path):
            logger.error(f"Model missing: {path}. Run train_model.py first.")
            raise FileNotFoundError(f"Model not found: {path}")
        artifacts[horizon] = _load_model_artifact(path)
        is_calibrated = bool(artifacts[horizon].get("calibrator") is not None)
        logger.info(f"Loaded [{horizon}] artifact from {path} (calibrated={is_calibrated})")

    conn = None
    try:
        conn = psycopg2.connect(POSTGRES.dsn)

        # Determine active features from model artifact metadata
        sample_artifact = artifacts.get("1y") or artifacts.get("5y")
        trained_features = None
        if sample_artifact and isinstance(sample_artifact.get("metadata"), dict):
            trained_features = sample_artifact["metadata"].get("features")
        if isinstance(trained_features, list) and len(trained_features) > 0:
            active_features = trained_features
        else:
            # Fallback: check which NAV columns exist in the source data
            probe_query = f"SELECT column_name FROM information_schema.columns WHERE table_schema || '.' || table_name = '{source_table}' OR table_name = '{source_table.split('.')[-1]}'"
            probe_df = pd.read_sql(probe_query, conn)
            available_cols = set(probe_df["column_name"].tolist()) if not probe_df.empty else set()
            active_features = FEATURES + [f for f in NAV_FEATURES if f in available_cols]

        logger.info(f"Using {len(active_features)} features for signal generation")

        feature_cols_sql = ', '.join(active_features)
        query = f"""
            SELECT DISTINCT ON (symbol)
                symbol,
                event_time::DATE as trade_date,
                close,
                {feature_cols_sql}
            FROM {source_table}
            WHERE rsi_14 IS NOT NULL
              AND sma_50 IS NOT NULL
              AND sma_200 IS NOT NULL
            ORDER BY symbol, event_time DESC
        """
        df = pd.read_sql(query, conn)

        if df.empty:
            logger.warning(f"No feature data in {source_table}")
            return

        X = df[active_features].fillna(0)
        proba_1y = _predict_prob_from_artifact(artifacts["1y"], X)
        proba_5y = _predict_prob_from_artifact(artifacts["5y"], X)

        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {dest_table} (
                symbol                  TEXT,
                trade_date              DATE,
                signal                  TEXT,
                confidence              NUMERIC,
                signal_1y               TEXT,
                confidence_1y           NUMERIC,
                signal_5y               TEXT,
                confidence_5y           NUMERIC,
                combined_confidence     NUMERIC,
                risk_score              NUMERIC,
                risk_bucket             TEXT,
                suggested_position_pct  NUMERIC,
                expected_return_1y      NUMERIC,
                risk_adjusted_score     NUMERIC,
                var_95_1d               NUMERIC,
                cvar_95_1d              NUMERIC,
                PRIMARY KEY (symbol, trade_date)
            )
            """
        )

        for col_def in [
            "signal_1y TEXT",
            "confidence_1y NUMERIC",
            "signal_5y TEXT",
            "confidence_5y NUMERIC",
            "combined_confidence NUMERIC",
            "risk_score NUMERIC",
            "risk_bucket TEXT",
            "suggested_position_pct NUMERIC",
            "expected_return_1y NUMERIC",
            "risk_adjusted_score NUMERIC",
            "var_95_1d NUMERIC",
            "cvar_95_1d NUMERIC",
        ]:
            cur.execute(f"ALTER TABLE {dest_table} ADD COLUMN IF NOT EXISTS {col_def}")

        # ── Batch INSERT via execute_values (~10-20x faster than row-by-row) ──
        rows_to_insert = []
        for idx, row in df.iterrows():
            p1y = float(proba_1y[idx])
            p5y = float(proba_5y[idx])
            sig1y = _prob_to_signal(p1y, "1y")
            sig5y = _prob_to_signal(p5y, "5y")
            risk = _compute_risk_fields(row, p1y=p1y, p5y=p5y)

            rows_to_insert.append((
                row["symbol"],
                row["trade_date"],
                sig1y,
                p1y,
                sig1y,
                p1y,
                sig5y,
                p5y,
                risk["combined_confidence"],
                risk["risk_score"],
                risk["risk_bucket"],
                risk["suggested_position_pct"],
                risk["expected_return_1y"],
                risk["risk_adjusted_score"],
                risk["var_95_1d"],
                risk["cvar_95_1d"],
            ))

        insert_sql = f"""
            INSERT INTO {dest_table}
                (
                    symbol, trade_date, signal, confidence,
                    signal_1y, confidence_1y, signal_5y, confidence_5y,
                    combined_confidence, risk_score, risk_bucket,
                    suggested_position_pct, expected_return_1y,
                    risk_adjusted_score, var_95_1d, cvar_95_1d
                )
            VALUES %s
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                signal = EXCLUDED.signal,
                confidence = EXCLUDED.confidence,
                signal_1y = EXCLUDED.signal_1y,
                confidence_1y = EXCLUDED.confidence_1y,
                signal_5y = EXCLUDED.signal_5y,
                confidence_5y = EXCLUDED.confidence_5y,
                combined_confidence = EXCLUDED.combined_confidence,
                risk_score = EXCLUDED.risk_score,
                risk_bucket = EXCLUDED.risk_bucket,
                suggested_position_pct = EXCLUDED.suggested_position_pct,
                expected_return_1y = EXCLUDED.expected_return_1y,
                risk_adjusted_score = EXCLUDED.risk_adjusted_score,
                var_95_1d = EXCLUDED.var_95_1d,
                cvar_95_1d = EXCLUDED.cvar_95_1d
        """
        execute_values(cur, insert_sql, rows_to_insert, page_size=500)

        conn.commit()
        cur.close()
        logger.info(f"Inserted {len(rows_to_insert)} signals into {dest_table} (batched)")

    except Exception:
        logger.exception("Investor signal engine failed")
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    src_table = sys.argv[1] if len(sys.argv) > 1 else "silver.mutual_funds_features_daily"
    dst_table = sys.argv[2] if len(sys.argv) > 2 else "gold.mutual_funds_investment_signals"
    generate_signals(src_table, dst_table)
