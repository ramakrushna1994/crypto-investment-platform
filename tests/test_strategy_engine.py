"""
Tests for strategy_engine.py — signal mapping, risk computation, and model artifact loading.
"""
import numpy as np
import pandas as pd
import pytest

from src.recommendation.strategy_engine import (
    _prob_to_signal,
    _compute_risk_fields,
    _load_model_artifact,
    _predict_prob_from_artifact,
    SIGNAL_THRESHOLDS,
)


# ── Signal Mapping Tests ─────────────────────────────────────────────────────

class TestProbToSignal:
    """Tests for _prob_to_signal threshold mapping."""

    def test_1y_invest_now(self):
        assert _prob_to_signal(0.80, "1y") == "INVEST NOW"

    def test_1y_accumulate(self):
        assert _prob_to_signal(0.60, "1y") == "ACCUMULATE"

    def test_1y_monitor(self):
        assert _prob_to_signal(0.50, "1y") == "MONITOR"

    def test_1y_wait(self):
        assert _prob_to_signal(0.30, "1y") == "WAIT"

    def test_1y_avoid(self):
        assert _prob_to_signal(0.10, "1y") == "AVOID"

    def test_1y_zero_probability(self):
        assert _prob_to_signal(0.0, "1y") == "AVOID"

    def test_1y_max_probability(self):
        assert _prob_to_signal(1.0, "1y") == "INVEST NOW"

    def test_5y_strong_hold(self):
        assert _prob_to_signal(0.75, "5y") == "STRONG HOLD"

    def test_5y_accumulate(self):
        assert _prob_to_signal(0.58, "5y") == "ACCUMULATE"

    def test_5y_avoid(self):
        assert _prob_to_signal(0.05, "5y") == "AVOID"

    def test_1y_boundary_invest_now(self):
        """Exact boundary: 0.72 should yield INVEST NOW."""
        assert _prob_to_signal(0.72, "1y") == "INVEST NOW"

    def test_1y_boundary_just_below_invest_now(self):
        """Just below 0.72 threshold."""
        assert _prob_to_signal(0.719, "1y") == "ACCUMULATE"

    def test_5y_boundary_strong_hold(self):
        assert _prob_to_signal(0.70, "5y") == "STRONG HOLD"

    def test_all_thresholds_descending(self):
        """Verify thresholds are defined in descending order."""
        for horizon, thresholds in SIGNAL_THRESHOLDS.items():
            values = [t for t, _ in thresholds]
            assert values == sorted(values, reverse=True), (
                f"Thresholds for {horizon} must be in descending order"
            )


# ── Risk Field Computation Tests ─────────────────────────────────────────────

class TestComputeRiskFields:
    """Tests for _compute_risk_fields."""

    def test_output_keys(self, sample_feature_row):
        result = _compute_risk_fields(sample_feature_row, p1y=0.7, p5y=0.6)
        expected_keys = {
            "combined_confidence",
            "risk_score",
            "risk_bucket",
            "suggested_position_pct",
            "expected_return_1y",
            "risk_adjusted_score",
            "var_95_1d",
            "cvar_95_1d",
        }
        assert set(result.keys()) == expected_keys

    def test_combined_confidence_weighted(self, sample_feature_row):
        result = _compute_risk_fields(sample_feature_row, p1y=0.80, p5y=0.60)
        # combined = 0.65 * 0.80 + 0.35 * 0.60 = 0.52 + 0.21 = 0.73
        assert abs(result["combined_confidence"] - 0.73) < 1e-6

    def test_risk_bucket_low(self, sample_feature_row):
        """Low volatility + strong trend → LOW risk."""
        row = sample_feature_row.copy()
        row["volatility_7d"] = 0.005
        row["atr_14"] = 5.0
        row["rsi_14"] = 50.0
        row["sma_50"] = 2500.0
        row["sma_200"] = 2400.0
        result = _compute_risk_fields(row, p1y=0.7, p5y=0.6)
        assert result["risk_bucket"] == "LOW"

    def test_risk_bucket_high(self, sample_feature_row):
        """High volatility + bearish trend + extreme RSI → HIGH risk."""
        row = sample_feature_row.copy()
        row["volatility_7d"] = 0.10
        row["atr_14"] = 200.0
        row["rsi_14"] = 85.0
        row["sma_50"] = 2000.0
        row["sma_200"] = 2500.0
        result = _compute_risk_fields(row, p1y=0.3, p5y=0.2)
        assert result["risk_bucket"] == "HIGH"

    def test_position_pct_bounds(self, sample_feature_row):
        """Suggested position must be between 1% and 12%."""
        for p1y in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for p5y in [0.0, 0.5, 1.0]:
                result = _compute_risk_fields(sample_feature_row, p1y=p1y, p5y=p5y)
                assert 0.01 <= result["suggested_position_pct"] <= 0.12, (
                    f"Position pct {result['suggested_position_pct']} out of bounds "
                    f"for p1y={p1y}, p5y={p5y}"
                )

    def test_var_cvar_non_negative(self, sample_feature_row):
        """VaR and CVaR must be >= 0."""
        result = _compute_risk_fields(sample_feature_row, p1y=0.5, p5y=0.5)
        assert result["var_95_1d"] >= 0.0
        assert result["cvar_95_1d"] >= 0.0

    def test_cvar_gte_var(self, sample_feature_row):
        """CVaR must always be >= VaR (expected shortfall >= value at risk)."""
        for p1y in [0.2, 0.5, 0.8]:
            result = _compute_risk_fields(sample_feature_row, p1y=p1y, p5y=0.5)
            assert result["cvar_95_1d"] >= result["var_95_1d"] - 1e-10, (
                f"CVaR {result['cvar_95_1d']} < VaR {result['var_95_1d']} for p1y={p1y}"
            )

    def test_zero_close_price(self, sample_feature_row):
        """Should handle close=0 without division errors."""
        row = sample_feature_row.copy()
        row["close"] = 0.0
        result = _compute_risk_fields(row, p1y=0.5, p5y=0.5)
        assert np.isfinite(result["risk_score"])

    def test_zero_sma200(self, sample_feature_row):
        """Should handle sma_200=0 without division errors."""
        row = sample_feature_row.copy()
        row["sma_200"] = 0.0
        result = _compute_risk_fields(row, p1y=0.5, p5y=0.5)
        assert np.isfinite(result["risk_score"])

    def test_risk_score_range(self, sample_feature_row):
        """Risk score should be 0-100."""
        result = _compute_risk_fields(sample_feature_row, p1y=0.5, p5y=0.5)
        assert 0.0 <= result["risk_score"] <= 100.0


# ── Model Artifact Loading Tests ─────────────────────────────────────────────

class TestLoadModelArtifact:
    """Tests for _load_model_artifact and _predict_prob_from_artifact."""

    def test_loads_single_artifact(self, tmp_path):
        """A dict with 'model' key is treated as single artifact."""
        import joblib
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="constant", constant=1)
        model.fit([[0]], [1])

        artifact = {"model": model, "calibrator": None, "metadata": {"test": True}}
        path = tmp_path / "test_model.joblib"
        joblib.dump(artifact, path)

        loaded = _load_model_artifact(str(path))
        assert loaded["model_kind"] == "single"
        assert loaded["model"] is not None
        assert loaded["calibrator"] is None

    def test_loads_legacy_artifact(self, tmp_path):
        """A raw estimator (not a dict) gets wrapped as legacy single artifact."""
        import joblib
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="constant", constant=0)
        model.fit([[0]], [0])

        path = tmp_path / "legacy_model.joblib"
        joblib.dump(model, path)

        loaded = _load_model_artifact(str(path))
        assert loaded["model_kind"] == "single"
        assert loaded["metadata"].get("legacy_artifact") is True

    def test_predict_prob_single(self, tmp_path):
        """Single artifact produces probability array matching input length."""
        import joblib
        from sklearn.dummy import DummyClassifier

        model = DummyClassifier(strategy="constant", constant=1)
        model.fit([[0], [1]], [0, 1])

        artifact = {"model_kind": "single", "model": model, "calibrator": None}
        X = pd.DataFrame({"rsi_14": [50, 60, 70]})
        probs = _predict_prob_from_artifact(artifact, X)
        assert len(probs) == 3
        assert all(0.0 <= p <= 1.0 for p in probs)
