"""
train_model.py

CPU-friendly supervised training for investor horizons (1Y/5Y):
- strict data quality gates (schema/freshness/class balance/outlier rate)
- capped rows for laptop-safe runtimes
- candidate model selection (HistGB / RF / LogisticRegression)
- optional soft-voting ensemble artifact
- probability calibration (Platt scaling)
- lightweight local experiment tracking and model registry
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from src.config.settings import POSTGRES, validate_table_name

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INVESTOR_HORIZONS = {
    "1y": (252, 1.10, ">=10% gain in 1 year"),
    "5y": (1260, 1.50, ">=50% gain in 5 years"),
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

# NAV-specific features computed by pyspark_etl for mutual fund tables.
# These are automatically included when present in the source data.
NAV_FEATURES = [
    "rolling_return_30d",
    "rolling_return_90d",
    "sortino_30d",
    "max_drawdown_30d",
    "nav_momentum_14d",
]


def _resolve_features(engine, source_table: str) -> List[str]:
    """Return the feature list: base FEATURES + any NAV_FEATURES present in the table."""
    schema, table = _source_schema_table(source_table)
    query = text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = :schema AND table_name = :table"
    )
    with engine.connect() as conn:
        cols = {r[0] for r in conn.execute(query, {"schema": schema, "table": table}).fetchall()}
    extra = [f for f in NAV_FEATURES if f in cols]
    if extra:
        logger.info(f"NAV features detected in {source_table}: {extra}")
    return FEATURES + extra

SUPPORTED_MODEL_NAMES = ["histgb", "rf", "logreg"]

# CPU/laptop-safe defaults
MAX_TRAIN_ROWS = int(os.getenv("AIQ_MAX_TRAIN_ROWS", "400000"))
CANDIDATE_SAMPLE_ROWS = int(os.getenv("AIQ_CANDIDATE_SAMPLE_ROWS", "120000"))
MIN_TRAIN_ROWS = int(os.getenv("AIQ_MIN_TRAIN_ROWS", "5000"))
MIN_SYMBOLS = int(os.getenv("AIQ_MIN_SYMBOLS", "25"))
MIN_POS_RATE = float(os.getenv("AIQ_MIN_POS_RATE", "0.01"))
MAX_POS_RATE = float(os.getenv("AIQ_MAX_POS_RATE", "0.99"))
MAX_OUTLIER_RATE = float(os.getenv("AIQ_MAX_OUTLIER_RATE", "0.20"))
MAX_SOURCE_AGE_DAYS = int(os.getenv("AIQ_MAX_SOURCE_AGE_DAYS", "10"))
CALIBRATION_FRACTION = float(os.getenv("AIQ_CALIBRATION_FRACTION", "0.15"))


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


ENABLE_ENSEMBLE = _bool_env("AIQ_ENABLE_ENSEMBLE", True)


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}@{POSTGRES.host}/{POSTGRES.db}"
    )


def _source_schema_table(source_table: str) -> Tuple[str, str]:
    if "." in source_table:
        schema, table = source_table.split(".", 1)
        return schema, table
    return "public", source_table


def _artifact_paths(source_table: str, horizon: str) -> Tuple[Path, Path]:
    model_dir = Path("/opt/airflow/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    clean_name = source_table.split(".")[-1] if "." in source_table else source_table
    model_path = model_dir / f"{clean_name}_rf_{horizon}_model.joblib"
    metrics_path = model_dir / f"{clean_name}_rf_{horizon}_metrics.json"
    return model_path, metrics_path


def _check_schema_gate(engine, source_table: str, features: List[str]) -> Tuple[bool, str]:
    schema, table = _source_schema_table(source_table)
    required_cols = {"symbol", "event_time", "close", *features}
    query = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"schema": schema, "table": table}).fetchall()
    cols = {r[0] for r in rows}
    missing = sorted(required_cols - cols)
    if missing:
        return False, f"missing_columns={','.join(missing)}"
    return True, "ok"


def _check_freshness_gate(engine, source_table: str, max_age_days: int) -> Tuple[bool, str]:
    query = text(f"SELECT MAX(event_time) AS max_event_time FROM {source_table}")
    with engine.connect() as conn:
        max_ts = conn.execute(query).scalar()
    if max_ts is None:
        return False, "no_event_time"

    max_dt = pd.Timestamp(max_ts).tz_localize(None)
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_days = (now - max_dt).total_seconds() / 86400.0
    if age_days > float(max_age_days):
        return False, f"stale_data_age_days={age_days:.2f}_gt_{max_age_days}"
    return True, f"age_days={age_days:.2f}"


def _candidate_model_names() -> List[str]:
    raw = os.getenv("AIQ_MODEL_CANDIDATES", "histgb,rf,logreg")
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    names = [p for p in parts if p in SUPPORTED_MODEL_NAMES]
    if not names:
        names = ["histgb", "rf"]
    return list(dict.fromkeys(names))


def _safe_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        return probs[:, 0] if model.classes_[0] == 1 else np.zeros(probs.shape[0], dtype=float)
    return probs[:, 1]


def _fit_platt_calibrator(raw_prob: np.ndarray, y_true: pd.Series) -> Optional[LogisticRegression]:
    if pd.Series(y_true).nunique() < 2:
        return None
    x = np.clip(raw_prob, 1e-6, 1 - 1e-6).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    calibrator.fit(x, y_true.to_numpy())
    return calibrator


def _apply_calibrator(raw_prob: np.ndarray, calibrator: Optional[LogisticRegression]) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw_prob, 0.0, 1.0)
    x = np.clip(raw_prob, 1e-6, 1 - 1e-6).reshape(-1, 1)
    calibrated = calibrator.predict_proba(x)[:, 1]
    return np.clip(calibrated, 0.0, 1.0)


def _build_model(model_name: str, horizon: str):
    if model_name == "rf":
        n_estimators = 200 if horizon == "5y" else 120
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=700,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_depth=8,
        min_samples_leaf=40,
        random_state=42,
    )


def _prepare_frame(df: pd.DataFrame, gain_threshold: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["symbol"] = out["symbol"].astype("category")
    out["event_time"] = pd.to_datetime(out["event_time"])
    out["target"] = (out["future_close"] > out["close"] * gain_threshold).astype(int)
    out = out.sort_values(["event_time", "symbol"]).reset_index(drop=True)
    if len(out) > MAX_TRAIN_ROWS:
        logger.info(
            f"Row cap applied for CPU runtime: {len(out):,} -> {MAX_TRAIN_ROWS:,} "
            "(keeping most recent rows)"
        )
        out = out.tail(MAX_TRAIN_ROWS).reset_index(drop=True)
    return out


def _winsorize_features(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    out = df.copy()
    feat_list = features or FEATURES
    for col in feat_list:
        if col not in out.columns:
            continue
        series = out[col]
        q_low = series.quantile(0.005)
        q_high = series.quantile(0.995)
        out[col] = series.clip(lower=q_low, upper=q_high)
    return out


def _estimate_outlier_rate(df: pd.DataFrame, features: List[str] = None) -> float:
    rates = []
    feat_list = features or FEATURES
    for col in feat_list:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        if series.empty:
            continue
        low = series.quantile(0.001)
        high = series.quantile(0.999)
        outlier_share = float(((series < low) | (series > high)).mean())
        rates.append(outlier_share)
    if not rates:
        return 0.0
    return float(np.mean(rates))


def _fit_candidate_with_calibration(
    frame: pd.DataFrame,
    model_name: str,
    horizon: str,
    features: List[str] = None,
) -> Tuple[object, Optional[LogisticRegression], Dict[str, object]]:
    feat_list = features or FEATURES
    model = _build_model(model_name, horizon=horizon)

    if len(frame) < 3000:
        model.fit(frame[feat_list], frame["target"])
        return model, None, {"calibration_rows": 0}

    cal_rows = int(len(frame) * CALIBRATION_FRACTION)
    if cal_rows < 800 or (len(frame) - cal_rows) < 1500:
        model.fit(frame[feat_list], frame["target"])
        return model, None, {"calibration_rows": 0}

    fit_df = frame.iloc[:-cal_rows]
    cal_df = frame.iloc[-cal_rows:]

    model.fit(fit_df[feat_list], fit_df["target"])
    raw_cal = _safe_predict_proba(model, cal_df[feat_list])
    calibrator = _fit_platt_calibrator(raw_cal, cal_df["target"])
    return model, calibrator, {"calibration_rows": int(cal_rows)}


def _evaluate_probs(y_true: pd.Series, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
    }
    if pd.Series(y_true).nunique() > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["roc_auc"] = float("nan")
    return out


def _candidate_score(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    roc_auc = metrics.get("roc_auc")
    brier = metrics.get("brier")
    f1 = metrics.get("f1")
    safe_auc = float("-inf") if roc_auc is None or np.isnan(roc_auc) else float(roc_auc)
    safe_brier = float("inf") if brier is None or np.isnan(brier) else float(brier)
    safe_f1 = float("-inf") if f1 is None or np.isnan(f1) else float(f1)
    return (safe_auc, -safe_brier, safe_f1)


def _predict_from_artifact(artifact: Dict[str, object], X: pd.DataFrame) -> np.ndarray:
    model_kind = str(artifact.get("model_kind", "single"))
    if model_kind == "ensemble":
        weighted = np.zeros(len(X), dtype=float)
        total_w = 0.0
        for comp in artifact.get("models", []):
            model = comp["model"]
            calibrator = comp.get("calibrator")
            w = float(comp.get("weight", 1.0))
            raw = _safe_predict_proba(model, X)
            p = _apply_calibrator(raw, calibrator)
            weighted += w * p
            total_w += w
        if total_w <= 0:
            return np.zeros(len(X), dtype=float)
        return np.clip(weighted / total_w, 0.0, 1.0)

    model = artifact["model"]
    calibrator = artifact.get("calibrator")
    raw = _safe_predict_proba(model, X)
    return _apply_calibrator(raw, calibrator)


def _build_fallback_artifact(reason: str, source_table: str, horizon: str, days_forward: int) -> Dict[str, object]:
    model = DummyClassifier(strategy="constant", constant=0)
    dummy_X = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
    dummy_y = pd.Series([0], name="target")
    model.fit(dummy_X, dummy_y)
    return {
        "model_kind": "single",
        "model_name": "dummy_constant_0",
        "model": model,
        "calibrator": None,
        "metadata": {
            "status": "fallback",
            "reason": reason,
            "source_table": source_table,
            "horizon": horizon,
            "days_forward": int(days_forward),
        },
    }


def _write_artifacts(
    source_table: str,
    horizon: str,
    artifact: Dict[str, object],
    metadata: Dict[str, object],
) -> Tuple[Path, Path]:
    model_path, metrics_path = _artifact_paths(source_table, horizon)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metrics_path


def _record_experiment(metadata: Dict[str, object], model_path: Path):
    model_dir = model_path.parent
    exp_dir = model_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    entry = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "model_path": str(model_path),
    }

    runs_path = exp_dir / "training_runs.jsonl"
    with runs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    registry_path = exp_dir / "model_registry.json"
    registry = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    key = f"{metadata.get('source_table')}:{metadata.get('horizon')}"
    registry[key] = {
        "run_id": run_id,
        "updated_utc": entry["timestamp_utc"],
        "model_path": str(model_path),
        "selected_model_type": metadata.get("selected_model_type"),
        "model_kind": metadata.get("model_kind"),
        "row_count": metadata.get("row_count"),
        "eval_roc_auc": metadata.get("eval_roc_auc"),
        "eval_brier": metadata.get("eval_brier"),
        "eval_f1": metadata.get("eval_f1"),
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _save_fallback(reason: str, source_table: str, horizon: str, days_forward: int):
    artifact = _build_fallback_artifact(reason, source_table, horizon, days_forward)
    metadata = artifact["metadata"]
    model_path, metrics_path = _write_artifacts(source_table, horizon, artifact, metadata)
    _record_experiment(metadata, model_path)
    logger.warning(f"[{horizon}] Saved fallback model: {model_path} | reason={reason}")
    logger.info(f"[{horizon}] Fallback metrics saved: {metrics_path}")


def train_model(source_table="silver.mutual_funds_features_daily", horizon="1y"):
    """Train horizon model(s) with calibration and persist artifact + metadata."""
    source_table = validate_table_name(source_table)
    
    if horizon not in INVESTOR_HORIZONS:
        raise ValueError(f"Invalid horizon '{horizon}'. Choose from: {list(INVESTOR_HORIZONS.keys())}")

    days_forward, gain_threshold, description = INVESTOR_HORIZONS[horizon]
    logger.info(f"Training {horizon.upper()} model on {source_table}")

    # Resolve feature set dynamically (base + NAV extras if present)

    engine = None
    try:
        engine = get_engine()

        active_features = _resolve_features(engine, source_table)
        logger.info(f"[{horizon}] Using {len(active_features)} features: {active_features}")

        schema_ok, schema_msg = _check_schema_gate(engine, source_table, active_features)
        if not schema_ok:
            _save_fallback(f"schema_gate_failed_{schema_msg}", source_table, horizon, days_forward)
            return

        fresh_ok, fresh_msg = _check_freshness_gate(engine, source_table, MAX_SOURCE_AGE_DAYS)
        if not fresh_ok:
            _save_fallback(f"freshness_gate_failed_{fresh_msg}", source_table, horizon, days_forward)
            return

        feature_cols = ", ".join(active_features)
        where_clauses = ["future_close IS NOT NULL"] + [f"{f} IS NOT NULL" for f in active_features]
        query_where = " AND ".join(where_clauses)
        query = f"""
            WITH labeled AS (
                SELECT
                    symbol,
                    event_time,
                    close,
                    LEAD(close, {days_forward}) OVER (PARTITION BY symbol ORDER BY event_time ASC) AS future_close,
                    {feature_cols}
                FROM {source_table}
            )
            SELECT *
            FROM labeled
            WHERE {query_where}
            ORDER BY event_time ASC, symbol ASC
        """

        logger.info(f"Fetching eligible labeled rows from DB for {horizon} ({description})...")
        df = pd.read_sql(query, engine)
        ml_df = _prepare_frame(df, gain_threshold=gain_threshold)

        if ml_df.empty:
            _save_fallback(f"no_rows_after_labeling_need_{days_forward}_days", source_table, horizon, days_forward)
            return

        row_count = int(len(ml_df))
        symbol_count = int(ml_df["symbol"].nunique())
        pos_rate = float(ml_df["target"].mean())

        logger.info(
            f"[{horizon}] rows={row_count:,}, symbols={symbol_count:,}, positive_rate={pos_rate * 100:.2f}%"
        )

        if row_count < MIN_TRAIN_ROWS:
            _save_fallback(
                f"insufficient_rows_{row_count}_lt_{MIN_TRAIN_ROWS}",
                source_table,
                horizon,
                days_forward,
            )
            return

        if symbol_count < MIN_SYMBOLS:
            _save_fallback(
                f"insufficient_symbols_{symbol_count}_lt_{MIN_SYMBOLS}",
                source_table,
                horizon,
                days_forward,
            )
            return

        if pos_rate <= MIN_POS_RATE or pos_rate >= MAX_POS_RATE:
            _save_fallback(
                f"class_imbalance_rate_{pos_rate:.6f}_outside_{MIN_POS_RATE}_{MAX_POS_RATE}",
                source_table,
                horizon,
                days_forward,
            )
            return

        outlier_rate_raw = _estimate_outlier_rate(ml_df)
        if outlier_rate_raw > MAX_OUTLIER_RATE:
            _save_fallback(
                f"outlier_rate_{outlier_rate_raw:.4f}_gt_{MAX_OUTLIER_RATE}",
                source_table,
                horizon,
                days_forward,
            )
            return

        ml_df = _winsorize_features(ml_df, active_features)
        outlier_rate_post = _estimate_outlier_rate(ml_df, active_features)

        split_idx = int(row_count * 0.8)
        train_df = ml_df.iloc[:split_idx].copy()
        test_df = ml_df.iloc[split_idx:].copy()

        if train_df.empty or test_df.empty or train_df["target"].nunique() < 2:
            _save_fallback("invalid_train_test_split", source_table, horizon, days_forward)
            return

        # Use active_features for all model fitting below
        fit_features = active_features

        candidate_names = _candidate_model_names()
        candidate_train_df = train_df.tail(min(len(train_df), CANDIDATE_SAMPLE_ROWS)).copy()

        logger.info(
            f"[{horizon}] candidate selection on {len(candidate_train_df):,} rows | candidates={candidate_names}"
        )

        candidate_rows = []
        fitted_candidates = {}
        for name in candidate_names:
            try:
                model, calibrator, fit_meta = _fit_candidate_with_calibration(candidate_train_df, name, horizon, fit_features)
                probs = _predict_from_artifact(
                    {
                        "model_kind": "single",
                        "model": model,
                        "calibrator": calibrator,
                    },
                    test_df[fit_features],
                )
                metrics = _evaluate_probs(test_df["target"], probs)
                candidate_rows.append({
                    "model_name": name,
                    **metrics,
                    **fit_meta,
                })
                fitted_candidates[name] = {
                    "model": model,
                    "calibrator": calibrator,
                    "metrics": metrics,
                }
            except Exception as e:
                logger.warning(f"[{horizon}] candidate={name} failed: {e}")

        if not candidate_rows:
            _save_fallback("all_candidate_models_failed", source_table, horizon, days_forward)
            return

        candidate_df = pd.DataFrame(candidate_rows)
        candidate_df["score"] = candidate_df.apply(
            lambda r: _candidate_score(
                {"roc_auc": r.get("roc_auc"), "brier": r.get("brier"), "f1": r.get("f1")}
            ),
            axis=1,
        )
        candidate_df = candidate_df.sort_values("score", ascending=False).reset_index(drop=True)

        best_name = str(candidate_df.iloc[0]["model_name"])
        selected_names = [best_name]
        if ENABLE_ENSEMBLE and len(candidate_df) >= 2:
            selected_names = [str(candidate_df.iloc[0]["model_name"]), str(candidate_df.iloc[1]["model_name"])]

        logger.info(f"[{horizon}] selected model(s): {selected_names}")

        # Refit selected model(s) on full frame (with fresh calibration split).
        deployment_components = []
        for name in selected_names:
            model, calibrator, fit_meta = _fit_candidate_with_calibration(ml_df, name, horizon, fit_features)
            match = candidate_df[candidate_df["model_name"] == name].iloc[0].to_dict()
            deployment_components.append(
                {
                    "name": name,
                    "model": model,
                    "calibrator": calibrator,
                    "weight": float(max(0.01, float(match.get("roc_auc") or 0.01))),
                    "eval_metrics": {
                        "accuracy": float(match.get("accuracy") or np.nan),
                        "f1": float(match.get("f1") or np.nan),
                        "brier": float(match.get("brier") or np.nan),
                        "roc_auc": float(match.get("roc_auc") or np.nan),
                    },
                    **fit_meta,
                }
            )

        model_kind = "ensemble" if len(deployment_components) > 1 else "single"
        if model_kind == "ensemble":
            artifact = {
                "model_kind": "ensemble",
                "models": deployment_components,
                "metadata": {},
            }
        else:
            only = deployment_components[0]
            artifact = {
                "model_kind": "single",
                "model_name": only["name"],
                "model": only["model"],
                "calibrator": only.get("calibrator"),
                "metadata": {},
            }

        test_prob = _predict_from_artifact(artifact, test_df[fit_features])
        eval_metrics = _evaluate_probs(test_df["target"], test_prob)
        test_pred = (test_prob >= 0.5).astype(int)

        logger.info(
            f"[{horizon}] Eval Accuracy={eval_metrics['accuracy'] * 100:.2f}% | "
            f"F1={eval_metrics['f1']:.4f} | Brier={eval_metrics['brier']:.5f} | "
            f"ROC-AUC={eval_metrics['roc_auc']:.5f}"
        )
        logger.info("\n" + classification_report(test_df["target"], test_pred, zero_division=0))

        metadata: Dict[str, object] = {
            "status": "ok",
            "source_table": source_table,
            "horizon": horizon,
            "days_forward": int(days_forward),
            "description": description,
            "row_count": row_count,
            "symbol_count": symbol_count,
            "positive_rate": pos_rate,
            "freshness_check": fresh_msg,
            "schema_check": schema_msg,
            "outlier_rate_pre_winsor": outlier_rate_raw,
            "outlier_rate_post_winsor": outlier_rate_post,
            "max_train_rows": int(MAX_TRAIN_ROWS),
            "candidate_sample_rows": int(CANDIDATE_SAMPLE_ROWS),
            "candidate_models": candidate_names,
            "candidate_leaderboard": candidate_df.drop(columns=["score"]).to_dict(orient="records"),
            "selected_model_type": selected_names[0],
            "selected_models": selected_names,
            "model_kind": model_kind,
            "ensemble_enabled": bool(ENABLE_ENSEMBLE),
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_f1": eval_metrics["f1"],
            "eval_brier": eval_metrics["brier"],
            "eval_roc_auc": eval_metrics["roc_auc"],
            "calibration_fraction": float(CALIBRATION_FRACTION),
        }
        artifact["metadata"] = metadata

        model_path, metrics_path = _write_artifacts(source_table, horizon, artifact, metadata)
        _record_experiment(metadata, model_path)
        logger.info(f"[{horizon}] Model artifact saved: {model_path}")
        logger.info(f"[{horizon}] Metrics saved: {metrics_path}")

    except Exception as e:
        logger.exception(f"Training failed for {source_table} [{horizon}]: {e}")
        raise
    finally:
        if engine is not None:
            engine.dispose()


def train_all_horizons(source_table="silver.mutual_funds_features_daily"):
    """Train 1Y and 5Y investor models for the given feature table. Called by Airflow DAG."""
    for horizon in INVESTOR_HORIZONS:
        train_model(source_table=source_table, horizon=horizon)


if __name__ == "__main__":
    src_table = sys.argv[1] if len(sys.argv) > 1 else "silver.mutual_funds_features_daily"
    for h in INVESTOR_HORIZONS:
        train_model(src_table, horizon=h)
