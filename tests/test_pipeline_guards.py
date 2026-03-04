"""
Tests for pipeline_guards.py — promotion gate with file-system based validation reports.
"""
import json
import time
from pathlib import Path

import pytest

from src.recommendation.pipeline_guards import (
    _parse_horizons,
    _safe_float,
    assert_model_promotion_gate,
)


# ── Parse Horizons Tests ─────────────────────────────────────────────────────

class TestParseHorizons:
    """Tests for _parse_horizons."""

    def test_none_defaults_to_1y(self):
        assert _parse_horizons(None) == ["1y"]

    def test_single_string(self):
        assert _parse_horizons("5y") == ["5y"]

    def test_comma_separated(self):
        assert _parse_horizons("1y,5y") == ["1y", "5y"]

    def test_list_input(self):
        assert _parse_horizons(["1y", "5y"]) == ["1y", "5y"]

    def test_strips_whitespace(self):
        assert _parse_horizons(" 1y , 5y ") == ["1y", "5y"]

    def test_empty_string_defaults(self):
        assert _parse_horizons("") == ["1y"]

    def test_empty_list_defaults(self):
        assert _parse_horizons([]) == ["1y"]

    def test_case_normalization(self):
        assert _parse_horizons("1Y,5Y") == ["1y", "5y"]


# ── Safe Float Tests ─────────────────────────────────────────────────────────

class TestSafeFloat:
    """Tests for _safe_float."""

    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_invalid_string(self):
        assert _safe_float("not_a_number") is None

    def test_integer(self):
        assert _safe_float(42) == 42.0


# ── Promotion Gate Tests ─────────────────────────────────────────────────────

class TestPromotionGate:
    """Tests for assert_model_promotion_gate with temp report files."""

    def _write_summary(self, path: Path, payload: dict):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_passes_with_valid_report(self, tmp_path, walk_forward_summary_payload):
        summary_path = tmp_path / "silver_test_features_daily_walk_forward_summary.json"
        self._write_summary(summary_path, walk_forward_summary_payload)

        result = assert_model_promotion_gate(
            source_table="silver.test_features_daily",
            reports_dir=str(tmp_path),
            required_horizons="1y",
            min_avg_roc_auc=0.55,
            min_avg_f1=0.40,
            max_report_age_days=1.0,
        )
        assert result["status"] == "ok"

    def test_fails_on_missing_report(self, tmp_path):
        with pytest.raises(ValueError, match="missing report"):
            assert_model_promotion_gate(
                source_table="silver.nonexistent_table",
                reports_dir=str(tmp_path),
                required_horizons="1y",
            )

    def test_fails_on_stale_report(self, tmp_path, walk_forward_summary_payload):
        summary_path = tmp_path / "silver_test_features_daily_walk_forward_summary.json"
        self._write_summary(summary_path, walk_forward_summary_payload)
        # backdate the file modification time by 30 days
        import os
        old_time = time.time() - (30 * 86400)
        os.utime(summary_path, (old_time, old_time))

        with pytest.raises(ValueError, match="report_age_days"):
            assert_model_promotion_gate(
                source_table="silver.test_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="1y",
                max_report_age_days=7.0,
            )

    def test_fails_on_low_roc_auc(self, tmp_path):
        payload = {
            "horizons": [{
                "horizon": "1y",
                "status": "ok",
                "summary": {"avg_roc_auc": 0.45, "avg_f1": 0.50},
                "drift_status": "ok",
                "drift_summary": {},
            }]
        }
        path = tmp_path / "silver_weak_features_daily_walk_forward_summary.json"
        self._write_summary(path, payload)

        with pytest.raises(ValueError, match="avg_roc_auc"):
            assert_model_promotion_gate(
                source_table="silver.weak_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="1y",
                min_avg_roc_auc=0.55,
            )

    def test_fails_on_low_f1(self, tmp_path):
        payload = {
            "horizons": [{
                "horizon": "1y",
                "status": "ok",
                "summary": {"avg_roc_auc": 0.65, "avg_f1": 0.20},
                "drift_status": "ok",
                "drift_summary": {},
            }]
        }
        path = tmp_path / "silver_lowf1_features_daily_walk_forward_summary.json"
        self._write_summary(path, payload)

        with pytest.raises(ValueError, match="avg_f1"):
            assert_model_promotion_gate(
                source_table="silver.lowf1_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="1y",
                min_avg_f1=0.40,
            )

    def test_fails_on_bad_status(self, tmp_path):
        payload = {
            "horizons": [{
                "horizon": "1y",
                "status": "degraded",
                "summary": {"avg_roc_auc": 0.70, "avg_f1": 0.60},
                "drift_status": "ok",
                "drift_summary": {},
            }]
        }
        path = tmp_path / "silver_bad_features_daily_walk_forward_summary.json"
        self._write_summary(path, payload)

        with pytest.raises(ValueError, match="status=degraded"):
            assert_model_promotion_gate(
                source_table="silver.bad_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="1y",
            )

    def test_fails_on_missing_horizon(self, tmp_path, walk_forward_summary_payload):
        path = tmp_path / "silver_missing_features_daily_walk_forward_summary.json"
        self._write_summary(path, walk_forward_summary_payload)

        with pytest.raises(ValueError, match="missing horizon='5y'"):
            assert_model_promotion_gate(
                source_table="silver.missing_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="5y",  # report only has 1y
            )

    def test_fails_on_high_drift(self, tmp_path):
        payload = {
            "horizons": [{
                "horizon": "1y",
                "status": "ok",
                "summary": {"avg_roc_auc": 0.65, "avg_f1": 0.50},
                "drift_status": "warn",
                "drift_summary": {"high_drift_features": 15},
            }]
        }
        path = tmp_path / "silver_drift_features_daily_walk_forward_summary.json"
        self._write_summary(path, payload)

        with pytest.raises(ValueError, match="high_drift_features"):
            assert_model_promotion_gate(
                source_table="silver.drift_features_daily",
                reports_dir=str(tmp_path),
                required_horizons="1y",
                max_high_drift_features=5,
                require_drift_ok=False,  # don't fail on drift_status, only on count
            )
