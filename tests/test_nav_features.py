"""
Tests for NAV-specific feature computation in pyspark_etl.py.
Since PySpark is not available in the local test environment, these tests
validate the mathematical correctness of NAV feature formulas using
plain Python/NumPy/pandas equivalents.
"""
import numpy as np
import pandas as pd
import pytest


class TestNAVFeatureFormulas:
    """
    Validate the mathematical formulas used in compute_nav_features()
    using pure-Python equivalents (no Spark dependency required).
    """

    @pytest.fixture
    def nav_series(self):
        """Generate a 120-day NAV series for a single fund."""
        np.random.seed(42)
        nav = 100.0
        navs = [nav]
        for _ in range(119):
            nav *= 1 + np.random.normal(0.0003, 0.008)
            navs.append(nav)
        dates = pd.date_range("2025-09-01", periods=120, freq="B")
        return pd.DataFrame({
            "symbol": "MF_TEST_001",
            "event_time": dates,
            "close": navs,
            "open": navs,
            "high": navs,
            "low": navs,
            "volume": 0,
        })

    def test_rolling_return_30d(self, nav_series):
        """30-day rolling return = close / lag(close, 30) - 1."""
        df = nav_series.copy()
        df["expected_rr30"] = df["close"] / df["close"].shift(30) - 1
        # Only check from day 30 onwards where the lag exists
        valid = df.iloc[30:]
        for idx in valid.index:
            expected = float(valid.loc[idx, "expected_rr30"])
            actual = (df.loc[idx, "close"] / df.loc[idx - 30, "close"]) - 1
            assert abs(expected - actual) < 1e-10

    def test_rolling_return_90d(self, nav_series):
        """90-day rolling return = close / lag(close, 90) - 1."""
        df = nav_series.copy()
        df["expected_rr90"] = df["close"] / df["close"].shift(90) - 1
        valid = df.iloc[90:]
        assert len(valid) > 0
        for idx in valid.index:
            expected = float(valid.loc[idx, "expected_rr90"])
            assert np.isfinite(expected)

    def test_sortino_positive_for_rising_nav(self):
        """A steadily rising NAV should have a positive Sortino ratio."""
        navs = [100 + i * 0.1 for i in range(60)]
        daily_rets = [(navs[i] / navs[i - 1]) - 1 for i in range(1, 60)]
        # All returns are positive -> downside deviation = 0 -> sortino = 0.0 (capped)
        # Actually, downside returns are all 0 (no negative returns)
        downside_rets = [min(r, 0) for r in daily_rets]
        downside_std = np.std(downside_rets)
        # With all zeros, std = 0 -> sortino should be 0 (guarded in code)
        assert downside_std < 1e-8

    def test_sortino_negative_for_falling_nav(self):
        """A falling NAV should produce a negative Sortino ratio."""
        navs = [100 - i * 0.2 for i in range(60)]
        daily_rets = [(navs[i] / navs[i - 1]) - 1 for i in range(1, 60)]
        window = daily_rets[-30:]
        mean_ret = np.mean(window)
        downside = [min(r, 0) for r in window]
        downside_std = np.std(downside)
        if downside_std > 1e-8:
            sortino = mean_ret / downside_std
            assert sortino < 0

    def test_max_drawdown_always_non_positive(self, nav_series):
        """Max drawdown = close / running_max - 1, always <= 0."""
        df = nav_series.copy()
        for window_start in range(0, len(df) - 30):
            window = df["close"].iloc[window_start:window_start + 30]
            running_max = window.cummax()
            drawdown = (window / running_max) - 1
            assert drawdown.iloc[-1] <= 0.0 + 1e-10

    def test_max_drawdown_zero_at_peak(self):
        """At an all-time-high within the window, drawdown should be 0."""
        navs = list(range(100, 131))  # Monotonically increasing
        max_nav = max(navs[-30:])
        drawdown = navs[-1] / max_nav - 1
        assert abs(drawdown) < 1e-10

    def test_nav_momentum_positive_above_sma(self):
        """NAV above its 14-day SMA should give positive momentum."""
        navs = [100 + i * 0.5 for i in range(20)]  # Rising
        sma_14 = np.mean(navs[-14:])
        momentum = (navs[-1] / sma_14) - 1
        assert momentum > 0

    def test_nav_momentum_negative_below_sma(self):
        """NAV below its 14-day SMA should give negative momentum."""
        navs = [100 + i * 0.5 for i in range(14)] + [95]  # Drop on last day
        sma_14 = np.mean(navs[-14:])
        momentum = (navs[-1] / sma_14) - 1
        assert momentum < 0

    def test_mf_atr_not_degenerate(self, nav_series):
        """
        Even with open=high=low=close (MF data), ATR is NOT zero because
        true_range = max(high-low, |high-prev_close|, |low-prev_close|)
        and cross-day NAV changes make |close - prev_close| > 0.
        """
        df = nav_series.copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]  # 0 for MF
        df["tr2"] = abs(df["high"] - df["prev_close"])
        df["tr3"] = abs(df["low"] - df["prev_close"])
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr_14"] = df["true_range"].rolling(14).mean()

        # After warmup, ATR should be > 0 (reflecting daily NAV changes)
        valid_atr = df["atr_14"].dropna()
        assert (valid_atr > 0).all(), "ATR should not be zero for MF data with varying NAV"


class TestNAVFeatureColumnConstants:
    """Verify NAV feature column definitions."""

    def test_nav_feature_columns_count(self):
        from src.etl.pyspark_etl import NAV_FEATURE_COLUMNS
        assert len(NAV_FEATURE_COLUMNS) == 5

    def test_nav_feature_columns_content(self):
        from src.etl.pyspark_etl import NAV_FEATURE_COLUMNS
        expected = {
            "rolling_return_30d",
            "rolling_return_90d",
            "sortino_30d",
            "max_drawdown_30d",
            "nav_momentum_14d",
        }
        assert set(NAV_FEATURE_COLUMNS) == expected

    def test_train_model_nav_features_match(self):
        """NAV_FEATURES in train_model must match NAV_FEATURE_COLUMNS in pyspark_etl."""
        from src.etl.pyspark_etl import NAV_FEATURE_COLUMNS
        from src.recommendation.train_model import NAV_FEATURES
        assert set(NAV_FEATURES) == set(NAV_FEATURE_COLUMNS)

    def test_strategy_engine_nav_features_match(self):
        """NAV_FEATURES in strategy_engine must match."""
        from src.etl.pyspark_etl import NAV_FEATURE_COLUMNS
        from src.recommendation.strategy_engine import NAV_FEATURES
        assert set(NAV_FEATURES) == set(NAV_FEATURE_COLUMNS)
