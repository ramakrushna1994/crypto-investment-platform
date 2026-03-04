"""
Shared pytest fixtures for AI Quant Investment Engine tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_feature_row():
    """Return a single-row pd.Series simulating a latest feature snapshot."""
    return pd.Series({
        "symbol": "RELIANCE.NS",
        "trade_date": pd.Timestamp("2026-03-01"),
        "close": 2500.0,
        "rsi_14": 55.0,
        "volatility_7d": 0.025,
        "macd": 12.5,
        "macd_signal": 11.0,
        "ema_20": 2480.0,
        "moving_avg_7d": 2490.0,
        "bb_upper": 2550.0,
        "bb_lower": 2420.0,
        "sma_50": 2450.0,
        "sma_200": 2300.0,
        "atr_14": 45.0,
        "stoch_k": 65.0,
        "stoch_d": 60.0,
    })


@pytest.fixture
def sample_features_df():
    """Return a multi-row DataFrame of synthetic features for training tests."""
    np.random.seed(42)
    n = 6000
    symbols = [f"SYM{i}" for i in range(30)]
    rows = []
    for sym in symbols:
        for day in range(n // len(symbols)):
            rows.append({
                "symbol": sym,
                "event_time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "close": 100 + np.random.randn() * 10,
                "future_close": 100 + np.random.randn() * 15,
                "rsi_14": np.clip(50 + np.random.randn() * 15, 5, 95),
                "volatility_7d": abs(np.random.randn() * 0.03),
                "macd": np.random.randn() * 2,
                "macd_signal": np.random.randn() * 2,
                "ema_20": 100 + np.random.randn() * 8,
                "moving_avg_7d": 100 + np.random.randn() * 8,
                "bb_upper": 110 + np.random.randn() * 5,
                "bb_lower": 90 + np.random.randn() * 5,
                "sma_50": 100 + np.random.randn() * 6,
                "sma_200": 100 + np.random.randn() * 4,
                "atr_14": abs(np.random.randn() * 3),
                "stoch_k": np.clip(50 + np.random.randn() * 20, 0, 100),
                "stoch_d": np.clip(50 + np.random.randn() * 20, 0, 100),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def walk_forward_summary_payload():
    """Return a realistic walk-forward summary JSON payload."""
    return {
        "horizons": [
            {
                "horizon": "1y",
                "status": "ok",
                "selected_model_type": "histgb",
                "recommended_min_prob": 0.55,
                "summary": {
                    "splits_ran": 4,
                    "avg_roc_auc": 0.62,
                    "avg_f1": 0.48,
                    "avg_brier": 0.22,
                    "avg_cagr_approx": 0.08,
                    "avg_sharpe_annualized": 1.1,
                    "avg_max_drawdown": -0.15,
                    "avg_hit_rate": 0.58,
                },
                "drift_status": "ok",
                "drift_summary": {
                    "avg_psi": 0.03,
                    "avg_ks": 0.05,
                    "high_drift_features": 0,
                    "medium_drift_features": 2,
                    "recent_start": "2025-06-01",
                    "recent_end": "2026-03-01",
                },
                "regime_rows": [
                    {"regime": "bull", "samples": 500, "hit_rate": 0.65, "avg_return": 0.12, "avg_benchmark_return": 0.10},
                    {"regime": "bear", "samples": 200, "hit_rate": 0.42, "avg_return": -0.05, "avg_benchmark_return": -0.08},
                    {"regime": "sideways", "samples": 300, "hit_rate": 0.55, "avg_return": 0.03, "avg_benchmark_return": 0.02},
                ],
            }
        ]
    }
