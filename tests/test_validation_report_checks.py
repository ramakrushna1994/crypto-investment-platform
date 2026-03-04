"""
Tests for validation_report_checks.py — walk-forward artifact freshness validation.
"""
import json
import os
import time
from pathlib import Path

import pytest

from src.recommendation.validation_report_checks import (
    verify_walk_forward_reports,
    DEFAULT_REPORT_SUFFIXES,
)


class TestVerifyWalkForwardReports:
    """Tests for verify_walk_forward_reports."""

    def _create_fresh_artifacts(self, reports_dir: Path, source_table: str, summary_payload: dict):
        """Create all required report artifacts as fresh files."""
        safe_table = source_table.replace(".", "_")
        for suffix in DEFAULT_REPORT_SUFFIXES:
            path = reports_dir / f"{safe_table}_{suffix}"
            if suffix.endswith(".json"):
                path.write_text(json.dumps(summary_payload), encoding="utf-8")
            else:
                path.write_text("col1,col2\n1,2\n", encoding="utf-8")

    def test_passes_with_all_fresh_artifacts(self, tmp_path, walk_forward_summary_payload):
        source = "silver.crypto_features_daily"
        self._create_fresh_artifacts(tmp_path, source, walk_forward_summary_payload)

        result = verify_walk_forward_reports(
            source_tables=[source],
            reports_dir=str(tmp_path),
            max_age_hours=1.0,
        )
        assert result["status"] == "ok"

    def test_fails_on_missing_artifacts(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            verify_walk_forward_reports(
                source_tables=["silver.nonexistent_table"],
                reports_dir=str(tmp_path),
                max_age_hours=48.0,
            )

    def test_fails_on_stale_artifacts(self, tmp_path, walk_forward_summary_payload):
        source = "silver.stale_features_daily"
        self._create_fresh_artifacts(tmp_path, source, walk_forward_summary_payload)

        # Backdate all files by 7 days
        safe_table = source.replace(".", "_")
        old_time = time.time() - (7 * 86400)
        for suffix in DEFAULT_REPORT_SUFFIXES:
            path = tmp_path / f"{safe_table}_{suffix}"
            os.utime(path, (old_time, old_time))

        with pytest.raises(ValueError, match="stale"):
            verify_walk_forward_reports(
                source_tables=[source],
                reports_dir=str(tmp_path),
                max_age_hours=24.0,
            )

    def test_fails_on_invalid_summary_json(self, tmp_path):
        source = "silver.invalid_features_daily"
        safe_table = source.replace(".", "_")
        for suffix in DEFAULT_REPORT_SUFFIXES:
            path = tmp_path / f"{safe_table}_{suffix}"
            if suffix.endswith(".json"):
                path.write_text("{invalid json", encoding="utf-8")
            else:
                path.write_text("col1\n1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid"):
            verify_walk_forward_reports(
                source_tables=[source],
                reports_dir=str(tmp_path),
                max_age_hours=48.0,
            )

    def test_fails_on_empty_source_tables(self, tmp_path):
        with pytest.raises(ValueError, match="source_tables"):
            verify_walk_forward_reports(
                source_tables=[],
                reports_dir=str(tmp_path),
            )

    def test_fails_on_missing_reports_dir(self):
        with pytest.raises(ValueError, match="does not exist"):
            verify_walk_forward_reports(
                source_tables=["silver.test"],
                reports_dir="/nonexistent/path",
            )

    def test_multiple_source_tables(self, tmp_path, walk_forward_summary_payload):
        sources = ["silver.crypto_features_daily", "silver.nifty50_features_daily"]
        for source in sources:
            self._create_fresh_artifacts(tmp_path, source, walk_forward_summary_payload)

        result = verify_walk_forward_reports(
            source_tables=sources,
            reports_dir=str(tmp_path),
            max_age_hours=1.0,
        )
        assert result["status"] == "ok"
        assert len(result["source_tables"]) == 2
