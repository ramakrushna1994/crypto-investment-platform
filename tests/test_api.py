"""
Tests for api/main.py — helper functions and endpoint logic.
"""
import json
import sys
from pathlib import Path

import pytest

# api/ is not a package under src/, so we need to ensure its parent is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from api.main import (
    _to_float,
    _summarize_regimes,
    _load_validation_rows,
)


# ── _to_float Tests ──────────────────────────────────────────────────────────

class TestToFloat:
    """Tests for _to_float safe conversion."""

    def test_valid_float(self):
        assert _to_float(3.14) == 3.14

    def test_valid_int(self):
        assert _to_float(42) == 42.0

    def test_valid_string(self):
        assert _to_float("2.5") == 2.5

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_nan_returns_none(self):
        assert _to_float(float("nan")) is None

    def test_invalid_string(self):
        assert _to_float("abc") is None

    def test_empty_string(self):
        assert _to_float("") is None


# ── Regime Summarization Tests ───────────────────────────────────────────────

class TestSummarizeRegimes:
    """Tests for _summarize_regimes."""

    def test_basic_regime_summary(self):
        rows = [
            {"regime": "bull", "samples": 100, "hit_rate": 0.60, "avg_return": 0.10, "avg_benchmark_return": 0.08},
            {"regime": "bear", "samples": 50, "hit_rate": 0.40, "avg_return": -0.05, "avg_benchmark_return": -0.07},
        ]
        result = _summarize_regimes(rows)
        assert result["bull_samples"] == 100
        assert abs(result["bull_hit_rate"] - 0.60) < 1e-6
        assert result["bear_samples"] == 50

    def test_empty_rows(self):
        assert _summarize_regimes([]) == {}
        assert _summarize_regimes(None) == {}

    def test_ignores_invalid_regimes(self):
        rows = [
            {"regime": "unknown", "samples": 100, "hit_rate": 0.50, "avg_return": 0.00, "avg_benchmark_return": 0.00},
        ]
        result = _summarize_regimes(rows)
        assert result == {}

    def test_weighted_average_across_splits(self):
        """Multiple rows for the same regime should be sample-weighted."""
        rows = [
            {"regime": "bull", "samples": 100, "hit_rate": 0.70, "avg_return": 0.15, "avg_benchmark_return": 0.10},
            {"regime": "bull", "samples": 300, "hit_rate": 0.50, "avg_return": 0.05, "avg_benchmark_return": 0.04},
        ]
        result = _summarize_regimes(rows)
        # weighted hit_rate = (100*0.70 + 300*0.50) / 400 = 220/400 = 0.55
        assert abs(result["bull_hit_rate"] - 0.55) < 1e-6
        assert result["bull_samples"] == 400

    def test_all_three_regimes(self):
        rows = [
            {"regime": "bull", "samples": 100, "hit_rate": 0.60, "avg_return": 0.10, "avg_benchmark_return": 0.08},
            {"regime": "bear", "samples": 50, "hit_rate": 0.35, "avg_return": -0.05, "avg_benchmark_return": -0.07},
            {"regime": "sideways", "samples": 80, "hit_rate": 0.50, "avg_return": 0.02, "avg_benchmark_return": 0.01},
        ]
        result = _summarize_regimes(rows)
        assert "bull_samples" in result
        assert "bear_samples" in result
        assert "sideways_samples" in result


# ── Validation Rows Loading Tests ────────────────────────────────────────────

class TestLoadValidationRows:
    """Tests for _load_validation_rows."""

    def test_missing_report_returns_status(self, tmp_path):
        rows = _load_validation_rows("crypto", "silver.crypto_features_daily", str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["status"] == "missing_report"
        assert rows[0]["asset_class"] == "crypto"

    def test_valid_report_parsed(self, tmp_path, walk_forward_summary_payload):
        safe_table = "silver_crypto_features_daily"
        path = tmp_path / f"{safe_table}_walk_forward_summary.json"
        path.write_text(json.dumps(walk_forward_summary_payload), encoding="utf-8")

        rows = _load_validation_rows("crypto", "silver.crypto_features_daily", str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        assert rows[0]["horizon"] == "1y"
        assert rows[0]["avg_roc_auc"] == 0.62
        assert rows[0]["drift_status"] == "ok"

    def test_corrupt_report(self, tmp_path):
        safe_table = "silver_bad_features_daily"
        path = tmp_path / f"{safe_table}_walk_forward_summary.json"
        path.write_text("{broken json", encoding="utf-8")

        rows = _load_validation_rows("bad", "silver.bad_features_daily", str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["status"] == "report_parse_failed"

    def test_empty_horizons(self, tmp_path):
        safe_table = "silver_empty_features_daily"
        path = tmp_path / f"{safe_table}_walk_forward_summary.json"
        path.write_text(json.dumps({"horizons": []}), encoding="utf-8")

        rows = _load_validation_rows("empty", "silver.empty_features_daily", str(tmp_path))
        assert len(rows) == 1
        assert rows[0]["status"] == "no_horizons"

    def test_regime_fields_included(self, tmp_path, walk_forward_summary_payload):
        safe_table = "silver_regime_features_daily"
        path = tmp_path / f"{safe_table}_walk_forward_summary.json"
        path.write_text(json.dumps(walk_forward_summary_payload), encoding="utf-8")

        rows = _load_validation_rows("regime", "silver.regime_features_daily", str(tmp_path))
        row = rows[0]
        assert "bull_samples" in row
        assert "bear_hit_rate" in row
        assert "sideways_avg_return" in row
