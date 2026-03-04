"""
walk_forward_evaluation.py

Phase-2 evaluator for investor horizons:
- Expanding-window walk-forward validation
- Probability calibration diagnostics (Brier score)
- Confidence bucket hit-rate/return analysis
- Simple top-N portfolio backtest metrics (CAGR, Sharpe, Max Drawdown)

Run inside the Airflow container (recommended):
    python -m src.recommendation.walk_forward_evaluation --source-table silver.crypto_features_daily
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from src.config.settings import POSTGRES
from src.recommendation.train_model import FEATURES, INVESTOR_HORIZONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_SWEEP_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
DEFAULT_MODEL_CANDIDATES = ["histgb", "rf", "logreg"]

# Regime thresholds based on equal-weight benchmark period return.
DEFAULT_BULL_THRESHOLD = 0.002
DEFAULT_BEAR_THRESHOLD = -0.002


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}@{POSTGRES.host}/{POSTGRES.db}"
    )


def fetch_labeled_rows(
    source_table: str,
    days_forward: int,
    gain_threshold: float,
) -> pd.DataFrame:
    """
    Fetch labeled rows from source features table.
    Target = 1 if future_close > close * gain_threshold else 0.
    """
    feature_cols = ", ".join(FEATURES)
    query = f"""
        WITH labeled AS (
            SELECT
                symbol,
                event_time::date AS event_date,
                close,
                LEAD(close, {days_forward}) OVER (
                    PARTITION BY symbol ORDER BY event_time ASC
                ) AS future_close,
                {feature_cols}
            FROM {source_table}
        )
        SELECT *
        FROM labeled
        WHERE future_close IS NOT NULL
          AND close > 0
          AND future_close > 0
          AND {' AND '.join([f'{f} IS NOT NULL' for f in FEATURES])}
        ORDER BY event_date ASC, symbol ASC
    """
    engine = get_engine()
    try:
        df = pd.read_sql(query, engine)
    finally:
        engine.dispose()

    if df.empty:
        return df

    # Normalize date type once so split comparisons are consistent.
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["target"] = (df["future_close"] > df["close"] * gain_threshold).astype(int)
    df["realized_return"] = (df["future_close"] / df["close"]) - 1.0
    df["realized_return"] = df["realized_return"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["realized_return"]).copy()
    return df


def build_model(model_type: str):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    if model_type == "logreg":
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
    # Default: CPU-friendly gradient boosting.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_depth=8,
        min_samples_leaf=40,
        random_state=42,
    )


def normalize_thresholds(thresholds: Optional[List[float]]) -> List[float]:
    raw = DEFAULT_SWEEP_THRESHOLDS if thresholds is None else thresholds
    cleaned = sorted({round(float(t), 4) for t in raw if 0.0 < float(t) < 1.0})
    if not cleaned:
        raise ValueError("At least one threshold in (0, 1) is required.")
    return cleaned


def parse_thresholds_arg(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return DEFAULT_SWEEP_THRESHOLDS.copy()
    return normalize_thresholds([float(p) for p in parts])


def _safe_float(value: Optional[float], fallback: float) -> float:
    if value is None:
        return fallback
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if np.isnan(v) else v


def select_best_model(model_results: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Rank candidate models using:
    1) higher ROC-AUC
    2) lower Brier score
    3) higher F1
    """
    scored = []
    for result in model_results:
        summary = result.get("summary", {})
        status = result.get("status", "unknown")
        roc_auc = _safe_float(summary.get("avg_roc_auc"), float("-inf"))
        brier = _safe_float(summary.get("avg_brier"), float("inf"))
        f1 = _safe_float(summary.get("avg_f1"), float("-inf"))
        if status != "ok":
            roc_auc = float("-inf")
            brier = float("inf")
            f1 = float("-inf")
        scored.append(
            {
                "model_type": result.get("model_type"),
                "status": status,
                "summary": summary,
                "score": [roc_auc, -brier, f1],
            }
        )

    if not scored:
        return "histgb", []

    best = max(scored, key=lambda x: tuple(x["score"]))
    # Strip internal score from serialized output.
    model_comparison = [{k: v for k, v in row.items() if k != "score"} for row in scored]
    return best["model_type"], model_comparison


def walk_forward_splits(
    dates: pd.Series,
    min_train_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Expanding-window splits:
    train: [0 .. train_end]
    test:  [test_start .. test_end]
    """
    unique_dates = pd.Series(sorted(pd.to_datetime(dates.unique())))
    splits = []
    idx = min_train_days
    while idx + test_days <= len(unique_dates):
        train_end = unique_dates.iloc[idx - 1]
        test_start = unique_dates.iloc[idx]
        test_end = unique_dates.iloc[idx + test_days - 1]
        splits.append((train_end, test_start, test_end))
        idx += step_days
    return splits


def leakage_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, object]:
    """
    Basic leakage checks:
    - train end date must be strictly before test start date
    - no (symbol, event_date) overlap between train and test
    """
    if train_df.empty or test_df.empty:
        return {
            "date_violation": True,
            "overlap_rows": 0,
            "train_max_date": None,
            "test_min_date": None,
        }

    train_max = pd.to_datetime(train_df["event_date"]).max()
    test_min = pd.to_datetime(test_df["event_date"]).min()
    date_violation = bool(train_max >= test_min)

    train_keys = train_df[["symbol", "event_date"]].drop_duplicates()
    test_keys = test_df[["symbol", "event_date"]].drop_duplicates()
    overlap_rows = int(train_keys.merge(test_keys, on=["symbol", "event_date"], how="inner").shape[0])

    return {
        "date_violation": date_violation,
        "overlap_rows": overlap_rows,
        "train_max_date": str(train_max.date()),
        "test_min_date": str(test_min.date()),
    }


def _classify_regime(benchmark_return: float) -> str:
    if benchmark_return >= DEFAULT_BULL_THRESHOLD:
        return "bull"
    if benchmark_return <= DEFAULT_BEAR_THRESHOLD:
        return "bear"
    return "sideways"


def regime_performance(
    enriched_df: pd.DataFrame,
    top_n: int,
    min_prob: float,
) -> pd.DataFrame:
    """
    Aggregate strategy hit-rate/return by market regime computed from
    equal-weight benchmark return per event_date.
    """
    if enriched_df.empty:
        return pd.DataFrame(columns=["regime", "samples", "avg_return", "hit_rate", "avg_benchmark_return"])

    rows = []
    grouped = enriched_df.sort_values(["event_date", "prob"], ascending=[True, False]).groupby("event_date")
    for event_date, frame in grouped:
        benchmark_return = float(frame["realized_return"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        picks = frame[frame["prob"] >= min_prob].head(top_n)
        if picks.empty:
            continue
        strategy_return = float(picks["realized_return"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        rows.append(
            {
                "event_date": pd.to_datetime(event_date),
                "regime": _classify_regime(benchmark_return),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "hit": 1 if strategy_return > 0 else 0,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["regime", "samples", "avg_return", "hit_rate", "avg_benchmark_return"])

    rdf = pd.DataFrame(rows)
    return (
        rdf.groupby("regime", as_index=False)
        .agg(
            samples=("strategy_return", "count"),
            avg_return=("strategy_return", "mean"),
            hit_rate=("hit", "mean"),
            avg_benchmark_return=("benchmark_return", "mean"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    if len(expected) < 2 or len(actual) < 2:
        return float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        edges = np.array([np.min(expected), np.mean(expected), np.max(expected)])
    expected_hist, _ = np.histogram(expected, bins=edges)
    actual_hist, _ = np.histogram(actual, bins=edges)
    expected_pct = np.clip(expected_hist / max(expected_hist.sum(), 1), 1e-6, 1.0)
    actual_pct = np.clip(actual_hist / max(actual_hist.sum(), 1), 1e-6, 1.0)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    values = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
    cdf_a = np.searchsorted(a_sorted, values, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, values, side="right") / len(b_sorted)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def compute_feature_drift(
    df: pd.DataFrame,
    feature_cols: List[str],
    recent_days: int = 63,
    baseline_days: int = 252,
    min_samples: int = 300,
) -> Dict[str, object]:
    """
    Compare recent vs baseline windows and compute feature PSI/KS drift diagnostics.
    """
    if df.empty:
        return {"status": "no_data", "summary": {}, "rows": []}

    tmp = df.copy()
    tmp["event_date"] = pd.to_datetime(tmp["event_date"])
    max_date = tmp["event_date"].max()
    recent_start = max_date - pd.Timedelta(days=max(recent_days - 1, 1))
    baseline_end = recent_start - pd.Timedelta(days=1)
    baseline_start = baseline_end - pd.Timedelta(days=max(baseline_days - 1, 1))

    recent_df = tmp[tmp["event_date"] >= recent_start]
    baseline_df = tmp[(tmp["event_date"] >= baseline_start) & (tmp["event_date"] <= baseline_end)]
    if len(recent_df) < min_samples or len(baseline_df) < min_samples:
        return {
            "status": "insufficient_samples",
            "summary": {
                "recent_rows": int(len(recent_df)),
                "baseline_rows": int(len(baseline_df)),
                "min_samples": int(min_samples),
            },
            "rows": [],
        }

    rows = []
    for col in feature_cols:
        if col not in tmp.columns:
            continue
        base = baseline_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        rec = recent_df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(base) < min_samples or len(rec) < min_samples:
            continue
        psi_v = _psi(base, rec, bins=10)
        ks_v = _ks_statistic(base, rec)
        drift_level = "low"
        if np.isnan(psi_v) or np.isnan(ks_v):
            drift_level = "unknown"
        elif psi_v >= 0.25 or ks_v >= 0.20:
            drift_level = "high"
        elif psi_v >= 0.10 or ks_v >= 0.10:
            drift_level = "medium"
        rows.append(
            {
                "feature": col,
                "psi": float(psi_v),
                "ks": float(ks_v),
                "drift_level": drift_level,
                "baseline_mean": float(np.nanmean(base)),
                "recent_mean": float(np.nanmean(rec)),
            }
        )

    if not rows:
        return {"status": "no_valid_features", "summary": {}, "rows": []}

    drift_df = pd.DataFrame(rows)
    high_features = int((drift_df["drift_level"] == "high").sum())
    medium_features = int((drift_df["drift_level"] == "medium").sum())
    summary = {
        "baseline_start": str(pd.to_datetime(baseline_start).date()),
        "baseline_end": str(pd.to_datetime(baseline_end).date()),
        "recent_start": str(pd.to_datetime(recent_start).date()),
        "recent_end": str(pd.to_datetime(max_date).date()),
        "baseline_rows": int(len(baseline_df)),
        "recent_rows": int(len(recent_df)),
        "avg_psi": float(drift_df["psi"].mean(skipna=True)),
        "avg_ks": float(drift_df["ks"].mean(skipna=True)),
        "high_drift_features": high_features,
        "medium_drift_features": medium_features,
    }
    return {"status": "ok", "summary": summary, "rows": drift_df.to_dict(orient="records")}


def portfolio_backtest(
    test_df: pd.DataFrame,
    holding_period_days: int,
    top_n: int = 10,
    min_prob: float = 0.55,
    trading_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    brokerage_bps: float = 2.0,
    tax_bps: float = 3.0,
) -> Dict[str, float]:
    """
    Equal-weight top-N strategy with transaction frictions, turnover, and
    benchmark comparison.
    """
    if test_df.empty:
        return {
            "periods": 0,
            "avg_period_return": np.nan,
            "benchmark_avg_period_return": np.nan,
            "alpha_avg_period_return": np.nan,
            "cagr_approx": np.nan,
            "benchmark_cagr_approx": np.nan,
            "alpha_cagr": np.nan,
            "sharpe_annualized": np.nan,
            "benchmark_sharpe_annualized": np.nan,
            "info_ratio": np.nan,
            "max_drawdown": np.nan,
            "benchmark_max_drawdown": np.nan,
            "hit_rate": np.nan,
            "avg_turnover": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
        }

    base_cost = (trading_cost_bps + slippage_bps + brokerage_bps + tax_bps) / 10000.0
    period_returns = []
    benchmark_returns = []
    turnovers = []
    prev_picks = set()

    grouped = test_df.sort_values(["event_date", "prob"], ascending=[True, False]).groupby("event_date")
    for _, g in grouped:
        benchmark_clean = g["realized_return"].replace([np.inf, -np.inf], np.nan).dropna()
        benchmark_returns.append(float(benchmark_clean.mean()) if not benchmark_clean.empty else 0.0)

        picks = g[g["prob"] >= min_prob].head(top_n)
        if picks.empty:
            period_returns.append(0.0)
            turnovers.append(0.0)
            prev_picks = set()
        else:
            clean_returns = picks["realized_return"].replace([np.inf, -np.inf], np.nan).dropna()
            if clean_returns.empty:
                period_returns.append(0.0)
                turnovers.append(0.0)
                prev_picks = set()
            else:
                curr_picks = set(picks["symbol"].astype(str).tolist())
                if not prev_picks:
                    turnover = 1.0
                else:
                    changed = len(curr_picks.symmetric_difference(prev_picks))
                    turnover = float(changed / max(2 * top_n, 1))
                prev_picks = curr_picks
                turnovers.append(turnover)
                period_cost = base_cost * turnover
                period_returns.append(float(clean_returns.mean() - period_cost))

    r = pd.Series(period_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(b) != len(r):
        n = min(len(b), len(r))
        r = r.iloc[:n].reset_index(drop=True)
        b = b.iloc[:n].reset_index(drop=True)

    if r.empty:
        return {
            "periods": 0,
            "avg_period_return": np.nan,
            "benchmark_avg_period_return": np.nan,
            "alpha_avg_period_return": np.nan,
            "cagr_approx": np.nan,
            "benchmark_cagr_approx": np.nan,
            "alpha_cagr": np.nan,
            "sharpe_annualized": np.nan,
            "benchmark_sharpe_annualized": np.nan,
            "info_ratio": np.nan,
            "max_drawdown": np.nan,
            "benchmark_max_drawdown": np.nan,
            "hit_rate": np.nan,
            "avg_turnover": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
        }

    equity = (1.0 + r).cumprod()
    benchmark_equity = (1.0 + b).cumprod() if not b.empty else pd.Series(dtype=float)
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    if not benchmark_equity.empty:
        benchmark_running_max = benchmark_equity.cummax()
        benchmark_drawdown = (benchmark_equity / benchmark_running_max) - 1.0
    else:
        benchmark_drawdown = pd.Series([np.nan], dtype=float)

    # Approximate annualization assuming daily rebalance.
    total_years = (len(r) * max(holding_period_days, 1)) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / total_years) - 1.0) if total_years > 0 else np.nan
    benchmark_cagr = (
        float(benchmark_equity.iloc[-1] ** (1.0 / total_years) - 1.0)
        if total_years > 0 and not benchmark_equity.empty
        else np.nan
    )
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    bstd = float(b.std(ddof=1)) if len(b) > 1 else 0.0
    periods_per_year = 252.0 / max(holding_period_days, 1)
    sharpe = float((r.mean() / std) * np.sqrt(periods_per_year)) if std > 0 else np.nan
    benchmark_sharpe = float((b.mean() / bstd) * np.sqrt(periods_per_year)) if bstd > 0 else np.nan

    excess = r - b
    excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    info_ratio = float((excess.mean() / excess_std) * np.sqrt(periods_per_year)) if excess_std > 0 else np.nan
    var_95 = float(np.nanpercentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

    return {
        "periods": int(len(r)),
        "holding_period_days": int(holding_period_days),
        "avg_period_return": float(r.mean()),
        "benchmark_avg_period_return": float(b.mean()) if not b.empty else np.nan,
        "alpha_avg_period_return": float(r.mean() - b.mean()) if not b.empty else np.nan,
        "cagr_approx": cagr,
        "benchmark_cagr_approx": benchmark_cagr,
        "alpha_cagr": float(cagr - benchmark_cagr) if not np.isnan(benchmark_cagr) else np.nan,
        "sharpe_annualized": sharpe,
        "benchmark_sharpe_annualized": benchmark_sharpe,
        "info_ratio": info_ratio,
        "max_drawdown": float(drawdown.min()),
        "benchmark_max_drawdown": float(benchmark_drawdown.min()),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(pd.Series(turnovers, dtype=float).mean()) if turnovers else np.nan,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }


def confidence_buckets(test_df: pd.DataFrame) -> pd.DataFrame:
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    labels = ["0.00-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-1.00"]
    bdf = test_df.copy()
    bdf["bucket"] = pd.cut(bdf["prob"], bins=bins, labels=labels, include_lowest=True, right=False)
    agg = (
        bdf.groupby("bucket", observed=False)
        .agg(
            samples=("target", "count"),
            hit_rate=("target", "mean"),
            avg_realized_return=("realized_return", "mean"),
        )
        .reset_index()
    )
    return agg


def evaluate_threshold_sweep(
    split_prediction_frames: List[pd.DataFrame],
    horizon: str,
    model_type: str,
    holding_period_days: int,
    thresholds: List[float],
    top_n: int,
    trading_cost_bps: float,
    slippage_bps: float,
    brokerage_bps: float,
    tax_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    rows = []
    for split_id, frame in enumerate(split_prediction_frames, start=1):
        for threshold in thresholds:
            bt = portfolio_backtest(
                frame,
                holding_period_days=holding_period_days,
                top_n=top_n,
                min_prob=threshold,
                trading_cost_bps=trading_cost_bps,
                slippage_bps=slippage_bps,
                brokerage_bps=brokerage_bps,
                tax_bps=tax_bps,
            )
            rows.append(
                {
                    "horizon": horizon,
                    "model_type": model_type,
                    "split": split_id,
                    "threshold": float(threshold),
                    **bt,
                }
            )

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), None

    split_df = pd.DataFrame(rows)
    summary_df = (
        split_df.groupby("threshold", as_index=False)
        .agg(
            splits=("split", "nunique"),
            avg_periods=("periods", "mean"),
            avg_period_return=("avg_period_return", "mean"),
            avg_benchmark_period_return=("benchmark_avg_period_return", "mean"),
            avg_alpha_period_return=("alpha_avg_period_return", "mean"),
            avg_cagr_approx=("cagr_approx", "mean"),
            avg_benchmark_cagr_approx=("benchmark_cagr_approx", "mean"),
            avg_alpha_cagr=("alpha_cagr", "mean"),
            avg_sharpe_annualized=("sharpe_annualized", "mean"),
            avg_benchmark_sharpe_annualized=("benchmark_sharpe_annualized", "mean"),
            avg_information_ratio=("info_ratio", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_benchmark_max_drawdown=("benchmark_max_drawdown", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
            avg_turnover=("avg_turnover", "mean"),
            avg_var_95=("var_95", "mean"),
            avg_cvar_95=("cvar_95", "mean"),
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )
    summary_df.insert(0, "model_type", model_type)
    summary_df.insert(0, "horizon", horizon)

    rank_df = summary_df.copy()
    rank_df["score_sharpe"] = rank_df["avg_sharpe_annualized"].replace([np.inf, -np.inf], np.nan).fillna(float("-inf"))
    rank_df["score_cagr"] = rank_df["avg_cagr_approx"].replace([np.inf, -np.inf], np.nan).fillna(float("-inf"))
    rank_df["score_hit_rate"] = rank_df["avg_hit_rate"].replace([np.inf, -np.inf], np.nan).fillna(float("-inf"))
    best_row = rank_df.sort_values(
        ["score_sharpe", "score_cagr", "score_hit_rate", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return split_df, summary_df, float(best_row["threshold"])


def evaluate_horizon(
    df: pd.DataFrame,
    horizon: str,
    days_forward: int,
    model_type: str,
    min_train_days: int,
    test_days: int,
    step_days: int,
    top_n: int,
    min_prob: float,
    trading_cost_bps: float,
    slippage_bps: float,
    brokerage_bps: float,
    tax_bps: float,
    sweep_thresholds: Optional[List[float]] = None,
) -> Dict:
    if df.empty:
        return {
            "horizon": horizon,
            "model_type": model_type,
            "status": "no_data",
            "splits": [],
            "bucket_rows": [],
            "regime_rows": [],
            "threshold_sweep": [],
            "summary": {},
        }

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    splits = walk_forward_splits(df["event_date"], min_train_days, test_days, step_days)
    if not splits:
        return {
            "horizon": horizon,
            "model_type": model_type,
            "status": "insufficient_history",
            "splits": [],
            "bucket_rows": [],
            "regime_rows": [],
            "threshold_sweep": [],
            "summary": {
                "rows": int(len(df)),
                "unique_dates": int(df["event_date"].nunique()),
                "message": "Not enough dates for requested walk-forward config.",
            },
        }

    split_rows = []
    bucket_rows = []
    regime_rows = []
    split_prediction_frames = []
    thresholds = normalize_thresholds(sweep_thresholds)

    for i, (train_end, test_start, test_end) in enumerate(splits, start=1):
        train_df = df[df["event_date"] <= train_end]
        test_df = df[(df["event_date"] >= test_start) & (df["event_date"] <= test_end)]
        if train_df.empty or test_df.empty:
            continue

        leakage = leakage_checks(train_df, test_df)
        if leakage["date_violation"] or leakage["overlap_rows"] > 0:
            logger.error(
                f"[{horizon}] split {i}: leakage detected "
                f"(date_violation={leakage['date_violation']}, overlap_rows={leakage['overlap_rows']})"
            )
            continue

        y_train = train_df["target"]
        if y_train.nunique() < 2:
            logger.warning(f"[{horizon}] split {i}: train has single class, skipping.")
            continue

        model = build_model(model_type)
        model.fit(train_df[FEATURES], y_train)
        prob = model.predict_proba(test_df[FEATURES])[:, 1]
        pred = (prob >= 0.5).astype(int)

        y_test = test_df["target"].to_numpy()
        metrics = {
            "horizon": horizon,
            "model_type": model_type,
            "split": i,
            "train_end": str(pd.to_datetime(train_end).date()),
            "test_start": str(pd.to_datetime(test_start).date()),
            "test_end": str(pd.to_datetime(test_end).date()),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "leakage_date_violation": bool(leakage["date_violation"]),
            "leakage_overlap_rows": int(leakage["overlap_rows"]),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "brier": float(brier_score_loss(y_test, prob)),
        }
        if len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
        else:
            metrics["roc_auc"] = np.nan

        enriched = test_df.copy()
        enriched["prob"] = prob
        bt = portfolio_backtest(
            enriched,
            holding_period_days=days_forward,
            top_n=top_n,
            min_prob=min_prob,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
            brokerage_bps=brokerage_bps,
            tax_bps=tax_bps,
        )
        metrics.update(bt)
        split_rows.append(metrics)
        split_prediction_frames.append(enriched)

        b = confidence_buckets(enriched)
        b["horizon"] = horizon
        b["model_type"] = model_type
        b["split"] = i
        bucket_rows.append(b)

        r = regime_performance(enriched, top_n=top_n, min_prob=min_prob)
        if not r.empty:
            r["horizon"] = horizon
            r["model_type"] = model_type
            r["split"] = i
            regime_rows.append(r)

    if not split_rows:
        return {
            "horizon": horizon,
            "model_type": model_type,
            "status": "no_valid_splits",
            "splits": [],
            "bucket_rows": [],
            "regime_rows": [],
            "threshold_sweep": [],
            "summary": {"rows": int(len(df))},
        }

    split_df = pd.DataFrame(split_rows)
    split_df = split_df.replace([np.inf, -np.inf], np.nan)
    bucket_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()
    threshold_split_df, threshold_summary_df, recommended_threshold = evaluate_threshold_sweep(
        split_prediction_frames=split_prediction_frames,
        horizon=horizon,
        model_type=model_type,
        holding_period_days=days_forward,
        thresholds=thresholds,
        top_n=top_n,
        trading_cost_bps=trading_cost_bps,
        slippage_bps=slippage_bps,
        brokerage_bps=brokerage_bps,
        tax_bps=tax_bps,
    )
    regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()

    summary = {
        "splits_ran": int(len(split_df)),
        "avg_accuracy": float(split_df["accuracy"].mean()),
        "avg_f1": float(split_df["f1"].mean()),
        "avg_brier": float(split_df["brier"].mean()),
        "avg_roc_auc": float(split_df["roc_auc"].mean(skipna=True)),
        "avg_cagr_approx": float(split_df["cagr_approx"].mean(skipna=True)),
        "avg_benchmark_cagr_approx": float(split_df["benchmark_cagr_approx"].mean(skipna=True)),
        "avg_alpha_cagr": float(split_df["alpha_cagr"].mean(skipna=True)),
        "avg_sharpe_annualized": float(split_df["sharpe_annualized"].mean(skipna=True)),
        "avg_benchmark_sharpe_annualized": float(split_df["benchmark_sharpe_annualized"].mean(skipna=True)),
        "avg_information_ratio": float(split_df["info_ratio"].mean(skipna=True)),
        "avg_max_drawdown": float(split_df["max_drawdown"].mean(skipna=True)),
        "avg_benchmark_max_drawdown": float(split_df["benchmark_max_drawdown"].mean(skipna=True)),
        "avg_hit_rate": float(split_df["hit_rate"].mean(skipna=True)),
        "avg_turnover": float(split_df["avg_turnover"].mean(skipna=True)),
        "avg_var_95": float(split_df["var_95"].mean(skipna=True)),
        "avg_cvar_95": float(split_df["cvar_95"].mean(skipna=True)),
    }

    return {
        "horizon": horizon,
        "model_type": model_type,
        "status": "ok",
        "splits": split_df.to_dict(orient="records"),
        "bucket_rows": bucket_df.to_dict(orient="records"),
        "regime_rows": regime_df.to_dict(orient="records"),
        "threshold_sweep": threshold_summary_df.to_dict(orient="records"),
        "threshold_sweep_splits": threshold_split_df.to_dict(orient="records"),
        "recommended_min_prob": recommended_threshold,
        "summary": summary,
    }


def save_outputs(
    output_dir: Path,
    source_table: str,
    results: Dict,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_table = source_table.replace(".", "_")

    json_path = output_dir / f"{safe_table}_walk_forward_summary.json"
    splits_path = output_dir / f"{safe_table}_walk_forward_splits.csv"
    buckets_path = output_dir / f"{safe_table}_walk_forward_buckets.csv"
    regimes_path = output_dir / f"{safe_table}_walk_forward_regimes.csv"
    thresholds_path = output_dir / f"{safe_table}_walk_forward_thresholds.csv"
    model_compare_path = output_dir / f"{safe_table}_walk_forward_model_compare.csv"
    drift_path = output_dir / f"{safe_table}_walk_forward_drift.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    all_splits = []
    all_buckets = []
    all_regimes = []
    all_thresholds = []
    all_model_compare = []
    all_drift = []
    for horizon_result in results["horizons"]:
        if horizon_result.get("splits"):
            all_splits.extend(horizon_result["splits"])
        if horizon_result.get("bucket_rows"):
            all_buckets.extend(horizon_result["bucket_rows"])
        if horizon_result.get("regime_rows"):
            all_regimes.extend(horizon_result["regime_rows"])
        if horizon_result.get("threshold_sweep"):
            all_thresholds.extend(horizon_result["threshold_sweep"])
        if horizon_result.get("drift_rows"):
            for row in horizon_result["drift_rows"]:
                all_drift.append(
                    {
                        "horizon": horizon_result.get("horizon"),
                        **row,
                    }
                )
        if horizon_result.get("model_comparison"):
            for row in horizon_result["model_comparison"]:
                summary = row.get("summary", {})
                all_model_compare.append(
                    {
                        "horizon": horizon_result.get("horizon"),
                        "selected_model_type": horizon_result.get("selected_model_type"),
                        "candidate_model_type": row.get("model_type"),
                        "candidate_status": row.get("status"),
                        **summary,
                    }
                )

    pd.DataFrame(all_splits).to_csv(splits_path, index=False)
    pd.DataFrame(all_buckets).to_csv(buckets_path, index=False)
    pd.DataFrame(all_regimes).to_csv(regimes_path, index=False)
    pd.DataFrame(all_thresholds).to_csv(thresholds_path, index=False)
    pd.DataFrame(all_model_compare).to_csv(model_compare_path, index=False)
    pd.DataFrame(all_drift).to_csv(drift_path, index=False)

    return {
        "summary_json": str(json_path),
        "splits_csv": str(splits_path),
        "buckets_csv": str(buckets_path),
        "regimes_csv": str(regimes_path),
        "thresholds_csv": str(thresholds_path),
        "model_compare_csv": str(model_compare_path),
        "drift_csv": str(drift_path),
    }


def run_walk_forward_evaluation(
    source_table: str,
    model_type: str = "histgb",
    min_train_days: int = 504,  # ~2 trading years
    test_days: int = 126,       # ~6 trading months
    step_days: int = 63,        # ~quarterly step
    top_n: int = 10,
    min_prob: float = 0.55,
    trading_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    brokerage_bps: float = 2.0,
    tax_bps: float = 3.0,
    drift_recent_days: int = 63,
    drift_baseline_days: int = 252,
    drift_min_samples: int = 300,
    thresholds: Optional[List[float]] = None,
    output_dir: str = "/opt/airflow/files/reports",
    return_results: bool = True,
) -> Dict:
    logger.info(f"Walk-forward evaluation started for {source_table}")
    horizons_out = []
    sweep_thresholds = normalize_thresholds(thresholds)

    for horizon, (days_forward, gain_threshold, desc) in INVESTOR_HORIZONS.items():
        logger.info(f"[{horizon}] Loading labeled rows ({desc})")
        try:
            df = fetch_labeled_rows(source_table, days_forward, gain_threshold)
        except Exception as e:
            logger.error(f"[{horizon}] data load failed: {e}")
            horizons_out.append(
                {"horizon": horizon, "status": "load_failed", "error": str(e), "splits": [], "summary": {}}
            )
            continue

        logger.info(f"[{horizon}] rows={len(df):,}, symbols={df['symbol'].nunique() if not df.empty else 0:,}")
        candidate_models = DEFAULT_MODEL_CANDIDATES if model_type in ("both", "auto") else [model_type]
        model_results = []
        for candidate_model in candidate_models:
            logger.info(f"[{horizon}] Evaluating model={candidate_model}")
            model_results.append(
                evaluate_horizon(
                    df=df,
                    horizon=horizon,
                    days_forward=days_forward,
                    model_type=candidate_model,
                    min_train_days=min_train_days,
                    test_days=test_days,
                    step_days=step_days,
                    top_n=top_n,
                    min_prob=min_prob,
                    trading_cost_bps=trading_cost_bps,
                    slippage_bps=slippage_bps,
                    brokerage_bps=brokerage_bps,
                    tax_bps=tax_bps,
                    sweep_thresholds=sweep_thresholds,
                )
            )

        if model_type in ("both", "auto"):
            selected_model, model_comparison = select_best_model(model_results)
            selected = next(
                (r for r in model_results if r.get("model_type") == selected_model),
                model_results[0],
            )
            horizon_result = {
                **selected,
                "selected_model_type": selected_model,
                "model_comparison": model_comparison,
            }
        else:
            selected = model_results[0]
            horizon_result = {
                **selected,
                "selected_model_type": selected.get("model_type"),
                "model_comparison": [
                    {
                        "model_type": selected.get("model_type"),
                        "status": selected.get("status"),
                        "summary": selected.get("summary", {}),
                    }
                ],
            }

        drift = compute_feature_drift(
            df=df,
            feature_cols=FEATURES,
            recent_days=drift_recent_days,
            baseline_days=drift_baseline_days,
            min_samples=drift_min_samples,
        )
        horizon_result["drift_status"] = drift.get("status")
        horizon_result["drift_summary"] = drift.get("summary", {})
        horizon_result["drift_rows"] = drift.get("rows", [])
        horizons_out.append(horizon_result)

    final = {
        "source_table": source_table,
        "model_type_request": model_type,
        "config": {
            "min_train_days": min_train_days,
            "test_days": test_days,
            "step_days": step_days,
            "top_n": top_n,
            "min_prob": min_prob,
            "trading_cost_bps": trading_cost_bps,
            "slippage_bps": slippage_bps,
            "brokerage_bps": brokerage_bps,
            "tax_bps": tax_bps,
            "drift_recent_days": drift_recent_days,
            "drift_baseline_days": drift_baseline_days,
            "drift_min_samples": drift_min_samples,
            "sweep_thresholds": sweep_thresholds,
        },
        "horizons": horizons_out,
    }

    paths = save_outputs(Path(output_dir), source_table, final)
    logger.info("Walk-forward outputs:")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")

    if return_results:
        return {"results": final, "paths": paths}
    return {"paths": paths}


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward and backtest evaluator.")
    parser.add_argument("--source-table", default="silver.crypto_features_daily")
    parser.add_argument("--model-type", default="histgb", choices=["histgb", "rf", "logreg", "both", "auto"])
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--step-days", type=int, default=63)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-prob", type=float, default=0.55)
    parser.add_argument("--trading-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--brokerage-bps", type=float, default=2.0)
    parser.add_argument("--tax-bps", type=float, default=3.0)
    parser.add_argument("--drift-recent-days", type=int, default=63)
    parser.add_argument("--drift-baseline-days", type=int, default=252)
    parser.add_argument("--drift-min-samples", type=int, default=300)
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70",
        help="Comma-separated confidence thresholds for sweep report, e.g. 0.50,0.55,0.60",
    )
    parser.add_argument("--output-dir", default="/opt/airflow/files/reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_walk_forward_evaluation(
        source_table=args.source_table,
        model_type=args.model_type,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        top_n=args.top_n,
        min_prob=args.min_prob,
        trading_cost_bps=args.trading_cost_bps,
        slippage_bps=args.slippage_bps,
        brokerage_bps=args.brokerage_bps,
        tax_bps=args.tax_bps,
        drift_recent_days=args.drift_recent_days,
        drift_baseline_days=args.drift_baseline_days,
        drift_min_samples=args.drift_min_samples,
        thresholds=parse_thresholds_arg(args.thresholds),
        output_dir=args.output_dir,
    )
