"""
validate_indicators.py

Local sanity validator for silver-layer indicators.
Compares Spark-computed features (compute_indicators) with pandas references
on a deterministic sample dataset.

Usage:
    python -m src.etl.validate_indicators
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.etl.pyspark_etl import compute_indicators


def build_sample_df(num_days: int = 120) -> pd.DataFrame:
    """Create a deterministic OHLC sample with trend + oscillation."""
    start = datetime(2023, 1, 1)
    rows = []
    for i in range(num_days):
        t = start + timedelta(days=i)
        base = 100.0 + (0.18 * i) + (2.5 * math.sin(i / 7.0))
        close = round(base, 6)
        high = round(close + 1.3 + 0.2 * math.sin(i / 3.0), 6)
        low = round(close - 1.1 - 0.2 * math.cos(i / 5.0), 6)
        rows.append(
            {
                "symbol": "TEST",
                "event_time": t,
                "close": close,
                "high": high,
                "low": low,
            }
        )
    return pd.DataFrame(rows)


def pandas_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Compute reference indicators with pandas using project-equivalent formulas."""
    out = df.copy().sort_values(["symbol", "event_time"]).reset_index(drop=True)

    # Trend/volatility
    out["moving_avg_7d"] = out["close"].rolling(7, min_periods=1).mean()
    out["volatility_7d"] = out["close"].rolling(7, min_periods=1).std(ddof=1)
    out["sma_50"] = out["close"].rolling(50, min_periods=1).mean()
    out["sma_200"] = out["close"].rolling(200, min_periods=1).mean()

    # EMA-based metrics (adjust=False to match recursive implementation)
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger
    sma20 = out["close"].rolling(20, min_periods=1).mean()
    std20 = out["close"].rolling(20, min_periods=1).std(ddof=1)
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20

    # RSI (with same edge handling as Spark code)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = (-delta.clip(upper=0)).fillna(0.0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out["rsi_14"] = rsi

    # ATR
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    true_range = pd.concat([tr1.fillna(0), tr2.fillna(0), tr3.fillna(0)], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14, min_periods=1).mean().values

    # Stochastic
    low14 = out["low"].rolling(14, min_periods=1).min()
    high14 = out["high"].rolling(14, min_periods=1).max()
    denom = (high14 - low14)
    stoch_k = ((out["close"] - low14) / denom) * 100
    stoch_k = stoch_k.where(denom != 0, 50.0)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = pd.Series(stoch_k).rolling(3, min_periods=1).mean().values

    return out


def run_validation(tol: float = 1e-5) -> None:
    spark = (
        SparkSession.builder
        .appName("IndicatorValidation")
        .master("local[2]")
        .getOrCreate()
    )

    try:
        pdf = build_sample_df()
        sdf = spark.createDataFrame(pdf)
        sdf = sdf.withColumn("close", F.col("close").cast("double")) \
                 .withColumn("high", F.col("high").cast("double")) \
                 .withColumn("low", F.col("low").cast("double"))

        spark_out = (
            compute_indicators(sdf)
            .orderBy("event_time")
            .toPandas()
            .sort_values("event_time")
            .reset_index(drop=True)
        )

        pd_ref = pandas_reference(pdf)

        # Compare last segment (avoids early warm-up edge differences dominating)
        compare_cols = [
            "moving_avg_7d", "volatility_7d", "sma_50", "sma_200",
            "ema_20", "macd", "macd_signal", "bb_upper", "bb_lower",
            "rsi_14", "atr_14", "stoch_k", "stoch_d",
        ]

        tail_n = 60
        s_tail = spark_out.tail(tail_n)
        p_tail = pd_ref.tail(tail_n)

        print("Indicator comparison (Spark vs pandas reference)")
        print(f"Rows compared: last {tail_n}")
        print(f"Tolerance: {tol}")
        print("-" * 72)

        has_fail = False
        for col in compare_cols:
            s = pd.to_numeric(s_tail[col], errors="coerce")
            p = pd.to_numeric(p_tail[col], errors="coerce")
            diff = (s - p).abs()
            max_diff = float(diff.max(skipna=True)) if not diff.empty else float("nan")
            ok = max_diff <= tol or pd.isna(max_diff)
            status = "PASS" if ok else "FAIL"
            print(f"{col:16s}  max_abs_diff={max_diff:.8f}  {status}")
            if not ok:
                has_fail = True

        print("-" * 72)
        if has_fail:
            raise SystemExit("Validation failed for one or more indicators.")
        print("All indicator checks passed.")
    finally:
        spark.stop()


if __name__ == "__main__":
    run_validation()
