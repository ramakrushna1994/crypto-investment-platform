"""
Tests for email_alert.py — validation summary formatting for email body.
"""
import numpy as np
import pandas as pd
import pytest

from src.recommendation.email_alert import build_validation_email_section


class TestBuildValidationEmailSection:
    """Tests for build_validation_email_section."""

    def test_empty_df_returns_not_available(self):
        result = build_validation_email_section(pd.DataFrame())
        assert "Not available" in result

    def test_ok_status_includes_metrics(self):
        df = pd.DataFrame([{
            "asset_class": "crypto",
            "horizon": "1y",
            "status": "ok",
            "selected_model_type": "histgb",
            "recommended_min_prob": 0.55,
            "avg_roc_auc": 0.62,
            "avg_f1": 0.48,
            "splits_ran": 4,
            "drift_status": "ok",
            "drift_high_features": 0,
            "drift_medium_features": 2,
            "bull_hit_rate": 0.65,
            "bear_hit_rate": 0.42,
            "sideways_hit_rate": 0.55,
        }])
        result = build_validation_email_section(df)
        assert "crypto" in result
        assert "histgb" in result
        assert "ROC-AUC" in result
        assert "F1" in result
        assert "drift" in result.lower()
        assert "bull" in result.lower() or "hit-rate" in result.lower()

    def test_non_ok_status_shows_status(self):
        df = pd.DataFrame([{
            "asset_class": "mutual_funds",
            "horizon": "1y",
            "status": "missing_report",
        }])
        result = build_validation_email_section(df)
        assert "mutual_funds" in result
        assert "missing_report" in result

    def test_multiple_assets(self):
        df = pd.DataFrame([
            {"asset_class": "crypto", "horizon": "1y", "status": "ok",
             "selected_model_type": "rf", "recommended_min_prob": 0.55,
             "avg_roc_auc": 0.60, "avg_f1": 0.45, "splits_ran": 3,
             "drift_status": "ok", "drift_high_features": 0, "drift_medium_features": 1,
             "bull_hit_rate": 0.62, "bear_hit_rate": 0.40, "sideways_hit_rate": 0.50},
            {"asset_class": "nifty50", "horizon": "5y", "status": "ok",
             "selected_model_type": "histgb", "recommended_min_prob": 0.60,
             "avg_roc_auc": 0.68, "avg_f1": 0.52, "splits_ran": 5,
             "drift_status": "ok", "drift_high_features": 1, "drift_medium_features": 3,
             "bull_hit_rate": 0.70, "bear_hit_rate": 0.38, "sideways_hit_rate": 0.52},
        ])
        result = build_validation_email_section(df)
        assert "crypto" in result
        assert "nifty50" in result

    def test_nan_metrics_show_na(self):
        df = pd.DataFrame([{
            "asset_class": "test",
            "horizon": "1y",
            "status": "ok",
            "selected_model_type": "rf",
            "recommended_min_prob": None,
            "avg_roc_auc": None,
            "avg_f1": None,
            "splits_ran": None,
            "drift_status": "n/a",
            "drift_high_features": None,
            "drift_medium_features": None,
            "bull_hit_rate": None,
            "bear_hit_rate": None,
            "sideways_hit_rate": None,
        }])
        result = build_validation_email_section(df)
        assert "n/a" in result
