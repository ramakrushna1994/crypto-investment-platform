"""
audit_tasks.py

Airflow task functions for audit logging, data quality, and reconciliation.
These tasks are executed as part of the DAG to validate ETL operations.
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
from src.audit.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


def _extract_training_gate_failure(reason: str):
    schema_prefix = "schema_gate_failed_"
    freshness_prefix = "freshness_gate_failed_"
    if reason.startswith(schema_prefix):
        return "schema", reason[len(schema_prefix):]
    if reason.startswith(freshness_prefix):
        return "freshness", reason[len(freshness_prefix):]
    return None, None


def _load_training_gate_failures(model_dir: str = "/opt/airflow/models"):
    root = Path(model_dir)
    if not root.exists():
        return {
            "status": "missing_model_dir",
            "model_dir": str(root),
            "scanned_files": 0,
            "tracked_models": 0,
            "failures": [],
        }

    latest_by_model = {}
    scanned_files = 0
    for metrics_path in root.glob("*_metrics.json"):
        scanned_files += 1
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        source_table = payload.get("source_table", metrics_path.stem)
        horizon = payload.get("horizon", "unknown")
        key = f"{source_table}:{horizon}"
        mtime = metrics_path.stat().st_mtime
        current = latest_by_model.get(key)
        if current is None or mtime > current["mtime"]:
            latest_by_model[key] = {
                "path": metrics_path,
                "mtime": mtime,
                "payload": payload,
            }

    failures = []
    for item in latest_by_model.values():
        payload = item["payload"]
        status = str(payload.get("status", "")).lower()
        reason = str(payload.get("reason", ""))
        gate_type, gate_detail = _extract_training_gate_failure(reason)
        if status != "fallback" or gate_type is None:
            continue

        failures.append(
            {
                "gate_type": gate_type,
                "gate_detail": gate_detail,
                "reason": reason,
                "source_table": payload.get("source_table"),
                "horizon": payload.get("horizon"),
                "metrics_path": str(item["path"]),
                "updated_utc": datetime.fromtimestamp(item["mtime"], tz=timezone.utc).isoformat(),
            }
        )

    failures = sorted(
        failures,
        key=lambda x: (
            str(x.get("source_table") or ""),
            str(x.get("horizon") or ""),
            str(x.get("gate_type") or ""),
        ),
    )
    return {
        "status": "ok",
        "model_dir": str(root),
        "scanned_files": int(scanned_files),
        "tracked_models": int(len(latest_by_model)),
        "failures": failures,
    }


def audit_etl_task(asset_class: str, source_table: str, target_table: str, **context):
    """
    Audit ETL operation: logs the completion of signal generation task.
    
    Args:
        asset_class: Asset class name (crypto, nifty50, etc.)
        source_table: Feature table (source of signals)
        target_table: Signals table (target)
        **context: Airflow context dict
    """
    dag_id = context["dag"].dag_id
    task_id = context["task"].task_id
    dag_run = context["dag_run"]
    
    try:
        audit = AuditLogger()
        
        # Log ETL operation
        audit_id = audit.log_etl_start(
            dag_id=dag_id,
            task_id=task_id,
            source_table=source_table,
            target_table=target_table,
            dag_run_id=dag_run.run_id,
            operation_type="SIGNAL_GENERATION"
        )
        
        # In a real scenario, get actual row counts from the database
        # For now, we'll log success
        audit.log_etl_success(
            audit_id=audit_id,
            rows_source=None,  # Would query source table
            rows_target=None   # Would query target table
        )
        
        logger.info(f"✓ AUDIT_ETL: {asset_class} - audit_id={audit_id}")
        
        return {
            "status": "success",
            "asset_class": asset_class,
            "audit_id": audit_id
        }
        
    except Exception as e:
        logger.error(f"✗ AUDIT_ETL FAILED: {asset_class} - {str(e)}")
        raise


def quality_check_task(asset_class: str, table_name: str, 
                       completeness_threshold: float = 95.0, **context):
    """
    Data quality check: validates completeness and uniqueness of data.
    
    Args:
        asset_class: Asset class name
        table_name: Table to check
        completeness_threshold: Minimum completeness % required
        **context: Airflow context
    """
    try:
        audit = AuditLogger()
        
        # Check data quality
        quality_result = audit.check_data_quality(
            table_name=table_name,
            completeness_threshold=completeness_threshold
        )
        
        if quality_result.get("status") == "NO_DATA":
            logger.warning(f"⚠ QUALITY_CHECK: {asset_class} - No quality metrics available")
            return {
                "status": "no_data",
                "asset_class": asset_class,
                "table": table_name
            }
        
        overall_pass = quality_result.get("overall_pass", False)
        completeness = quality_result.get("completeness_percent", 0)
        uniqueness = quality_result.get("uniqueness_percent", 0)
        quality_score = quality_result.get("quality_score", 0)
        
        if overall_pass:
            logger.info(f"✓ QUALITY_CHECK PASS: {asset_class}")
            logger.info(f"  Completeness: {completeness:.1f}% | Uniqueness: {uniqueness:.1f}% | Score: {quality_score:.1f}")
        else:
            logger.warning(f"⚠ QUALITY_CHECK WARNING: {asset_class}")
            logger.warning(f"  Completeness: {completeness:.1f}% (threshold: {completeness_threshold}%)")
            logger.warning(f"  Uniqueness: {uniqueness:.1f}% | Score: {quality_score:.1f}")
        
        return {
            "status": "pass" if overall_pass else "warn",
            "asset_class": asset_class,
            "table": table_name,
            "completeness_percent": completeness,
            "uniqueness_percent": uniqueness,
            "quality_score": quality_score,
            "quality_status": quality_result.get("quality_status", "UNKNOWN")
        }
        
    except Exception as e:
        logger.error(f"✗ QUALITY_CHECK FAILED: {asset_class} - {str(e)}")
        raise


def reconcile_task(asset_class: str, source_table: str, target_table: str, **context):
    """
    Reconciliation check: validates that row counts match between source and target.
    
    Args:
        asset_class: Asset class name
        source_table: Source table to reconcile from
        target_table: Target table to reconcile to
        **context: Airflow context
    """
    try:
        audit = AuditLogger()
        
        # For reconciliation, we need an audit_id from a prior ETL operation
        # In production, this would link to the signal generation task's audit_id
        # For now, we'll create a new audit context
        audit_id = audit.log_etl_start(
            dag_id=context["dag"].dag_id,
            task_id=context["task"].task_id,
            source_table=source_table,
            target_table=target_table,
            dag_run_id=context["dag_run"].run_id,
            operation_type="RECONCILIATION"
        )
        
        # Perform reconciliation
        recon_result = audit.reconcile_tables(
            audit_id=audit_id,
            source_table=source_table,
            target_table=target_table
        )
        
        status = recon_result.get("status", "UNKNOWN")
        match = recon_result.get("match", False)
        variance = recon_result.get("variance_percent", 0)
        source_count = recon_result.get("source_count", 0)
        target_count = recon_result.get("target_count", 0)
        
        if match:
            logger.info(f"✓ RECONCILE PASS: {asset_class}")
            logger.info(f"  Source: {source_count} rows | Target: {target_count} rows")
        elif variance < 5:
            logger.warning(f"⚠ RECONCILE WARNING: {asset_class}")
            logger.warning(f"  Source: {source_count} rows | Target: {target_count} rows | Variance: {variance:.1f}%")
        else:
            logger.error(f"✗ RECONCILE FAIL: {asset_class}")
            logger.error(f"  Source: {source_count} rows | Target: {target_count} rows | Variance: {variance:.1f}%")
            raise Exception(f"Reconciliation failed for {asset_class}: {variance:.1f}% variance")
        
        return {
            "status": status.lower(),
            "asset_class": asset_class,
            "source_count": source_count,
            "target_count": target_count,
            "variance_percent": variance,
            "match": match
        }
        
    except Exception as e:
        logger.error(f"✗ RECONCILE FAILED: {asset_class} - {str(e)}")
        raise


def generate_daily_audit_report(**context):
    """
    Generate daily audit report summarizing all ETL operations.
    Creates a comprehensive report including:
    - Task execution status
    - Data quality metrics
    - Reconciliation results
    - Performance statistics
    
    Args:
        **context: Airflow context
    """
    try:
        audit = AuditLogger()
        
        # Get audit summary for today
        summary = audit.get_audit_summary(days=1)
        
        total_runs = summary.get("total_runs", 0)
        successful_runs = summary.get("successful_runs", 0)
        failed_runs = summary.get("failed_runs", 0)
        
        logger.info("=" * 80)
        logger.info("DAILY AUDIT REPORT")
        logger.info("=" * 80)
        logger.info(f"Total Runs: {total_runs}")
        logger.info(f"Successful: {successful_runs} ✓")
        logger.info(f"Failed: {failed_runs} ✗")
        
        if failed_runs > 0:
            logger.warning("\nFailed Tasks:")
            failed_tasks = audit.get_failed_tasks(days=1)
            for task in failed_tasks:
                logger.warning(f"  - {task.get('dag_id')}/{task.get('task_id')}: {task.get('error_message')}")
        else:
            failed_tasks = []

        training_gate_summary = _load_training_gate_failures()
        gate_failures = training_gate_summary.get("failures", [])
        if gate_failures:
            logger.warning("\nTraining Gate Failures (latest artifacts):")
            for fail in gate_failures:
                logger.warning(
                    "  - %s [%s] gate=%s detail=%s path=%s",
                    fail.get("source_table"),
                    fail.get("horizon"),
                    fail.get("gate_type"),
                    fail.get("gate_detail"),
                    fail.get("metrics_path"),
                )
        else:
            logger.info("\nTraining Gate Failures: none detected in latest model artifacts.")
        
        report_status = "PASS"
        if failed_runs > 0:
            report_status = "FAIL"
        elif gate_failures:
            report_status = "WARN"

        audit_results = {
            "summary": summary,
            "failed_tasks": failed_tasks,
            "training_gate_summary": training_gate_summary,
            "report_status": report_status,
        }
        
        logger.info("=" * 80)
        
        return audit_results
        
    except Exception as e:
        logger.error(f"✗ AUDIT REPORT GENERATION FAILED - {str(e)}")
        raise
