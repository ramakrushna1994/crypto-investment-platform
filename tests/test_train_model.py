"""
Tests for train_model.py — data quality gates, winsorization, candidate scoring, and fallback logic.
"""
import numpy as np
import pandas as pd
import pytest

from src.recommendation.train_model import (
    FEATURES,
    INVESTOR_HORIZONS,
    _winsorize_features,
    _estimate_outlier_rate,
    _candidate_score,
    _evaluate_probs,
    _prepare_frame,
    _build_fallback_artifact,
)


# ── Winsorization Tests ──────────────────────────────────────────────────────

class TestWinsorize:
    """Tests for _winsorize_features."""

    def test_clips_extreme_values(self):
        """Values beyond 0.5/99.5 percentile should be clipped."""
        df = pd.DataFrame({
            "rsi_14": [0.0] * 50 + [50.0] * 900 + [100.0] * 50,
            "volatility_7d": [0.01] * 1000,
        })
        result = _winsorize_features(df)
        # The extreme 0.0 and 100.0 values should be clipped
        assert result["rsi_14"].min() >= df["rsi_14"].quantile(0.005)
        assert result["rsi_14"].max() <= df["rsi_14"].quantile(0.995)

    def test_does_not_modify_original(self, sample_features_df):
        """Winsorization should return a copy, not modify in place."""
        original_vals = sample_features_df["rsi_14"].copy()
        _winsorize_features(sample_features_df)
        pd.testing.assert_series_equal(sample_features_df["rsi_14"], original_vals)

    def test_handles_missing_feature_columns(self):
        """If a FEATURES column doesn't exist, skip it gracefully."""
        df = pd.DataFrame({"rsi_14": [50.0, 60.0, 70.0]})
        result = _winsorize_features(df)
        assert "rsi_14" in result.columns
        assert len(result) == 3

    def test_empty_dataframe(self):
        """Should handle an empty DataFrame without errors."""
        df = pd.DataFrame(columns=FEATURES)
        result = _winsorize_features(df)
        assert result.empty


# ── Outlier Rate Estimation Tests ─────────────────────────────────────────────

class TestEstimateOutlierRate:
    """Tests for _estimate_outlier_rate."""

    def test_uniform_data_low_outlier_rate(self):
        """Normally distributed data should have very low outlier rate."""
        np.random.seed(42)
        df = pd.DataFrame({
            feat: np.random.normal(50, 10, 10000) for feat in FEATURES
        })
        rate = _estimate_outlier_rate(df)
        assert rate < 0.01  # Less than 1% outliers

    def test_all_constant_yields_zero(self):
        """Constant columns have no outliers."""
        df = pd.DataFrame({feat: [42.0] * 100 for feat in FEATURES})
        rate = _estimate_outlier_rate(df)
        assert rate == 0.0

    def test_empty_dataframe(self):
        """Empty df should return 0.0."""
        df = pd.DataFrame(columns=FEATURES)
        rate = _estimate_outlier_rate(df)
        assert rate == 0.0

    def test_positive_rate(self):
        """Injected outliers should raise the rate."""
        np.random.seed(42)
        data = {feat: np.random.normal(50, 5, 1000) for feat in FEATURES}
        # Inject heavy outlier for one feature
        data["rsi_14"][0] = 99999.0
        data["rsi_14"][1] = -99999.0
        df = pd.DataFrame(data)
        rate = _estimate_outlier_rate(df)
        assert rate > 0.0


# ── Candidate Scoring Tests ──────────────────────────────────────────────────

class TestCandidateScore:
    """Tests for _candidate_score ranking logic."""

    def test_higher_auc_wins(self):
        """Model with higher ROC-AUC should score higher."""
        score_high = _candidate_score({"roc_auc": 0.70, "brier": 0.20, "f1": 0.50})
        score_low = _candidate_score({"roc_auc": 0.55, "brier": 0.20, "f1": 0.50})
        assert score_high > score_low

    def test_lower_brier_wins(self):
        """Model with lower Brier loss should score higher (same AUC/F1)."""
        score_good = _candidate_score({"roc_auc": 0.65, "brier": 0.15, "f1": 0.50})
        score_bad = _candidate_score({"roc_auc": 0.65, "brier": 0.30, "f1": 0.50})
        assert score_good > score_bad

    def test_nan_auc_ranks_last(self):
        """NaN ROC-AUC should yield the worst possible score."""
        score_valid = _candidate_score({"roc_auc": 0.50, "brier": 0.25, "f1": 0.40})
        score_nan = _candidate_score({"roc_auc": float("nan"), "brier": 0.25, "f1": 0.40})
        assert score_valid > score_nan

    def test_none_values_handled(self):
        """None metrics should not crash."""
        score = _candidate_score({"roc_auc": None, "brier": None, "f1": None})
        assert isinstance(score, tuple) and len(score) == 3


# ── Evaluate Probabilities Tests ─────────────────────────────────────────────

class TestEvaluateProbs:
    """Tests for _evaluate_probs metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1, brier=0."""
        y_true = pd.Series([0, 0, 1, 1, 1])
        probs = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        metrics = _evaluate_probs(y_true, probs)
        assert metrics["accuracy"] == 1.0
        assert metrics["brier"] == 0.0
        assert metrics["roc_auc"] == 1.0

    def test_random_predictions(self):
        """Random predictions — metrics should be in valid ranges."""
        np.random.seed(42)
        y_true = pd.Series(np.random.randint(0, 2, 500))
        probs = np.random.rand(500)
        metrics = _evaluate_probs(y_true, probs)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["brier"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_single_class(self):
        """Single-class target should return nan ROC-AUC without crashing."""
        y_true = pd.Series([0, 0, 0])
        probs = np.array([0.1, 0.2, 0.3])
        metrics = _evaluate_probs(y_true, probs)
        assert np.isnan(metrics["roc_auc"])


# ── Prepare Frame Tests ──────────────────────────────────────────────────────

class TestPrepareFrame:
    """Tests for _prepare_frame label creation and row capping."""

    def test_creates_target_column(self, sample_features_df):
        result = _prepare_frame(sample_features_df, gain_threshold=1.10)
        assert "target" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})

    def test_sorted_by_event_time(self, sample_features_df):
        result = _prepare_frame(sample_features_df, gain_threshold=1.10)
        assert result["event_time"].is_monotonic_increasing or len(result) == 0

    def test_empty_input(self):
        df = pd.DataFrame()
        result = _prepare_frame(df, gain_threshold=1.10)
        assert result.empty


# ── Fallback Artifact Tests ──────────────────────────────────────────────────

class TestBuildFallbackArtifact:
    """Tests for _build_fallback_artifact."""

    def test_fallback_structure(self):
        artifact = _build_fallback_artifact(
            reason="test_reason",
            source_table="silver.test_table",
            horizon="1y",
            days_forward=252,
        )
        assert artifact["model_kind"] == "single"
        assert artifact["model_name"] == "dummy_constant_0"
        assert artifact["calibrator"] is None
        assert artifact["metadata"]["status"] == "fallback"
        assert artifact["metadata"]["reason"] == "test_reason"

    def test_fallback_predicts_zero(self):
        """Fallback model should always predict class 0 (AVOID)."""
        artifact = _build_fallback_artifact("test", "t", "1y", 252)
        model = artifact["model"]
        X = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
        pred = model.predict(X)
        assert pred[0] == 0

    def test_all_horizons_supported(self):
        for horizon, (days, _, _) in INVESTOR_HORIZONS.items():
            artifact = _build_fallback_artifact("test", "t", horizon, days)
            assert artifact["metadata"]["horizon"] == horizon
