"""
audit_logger.py

Audit and Reconciliation logging for ETL pipelines.
Integrates with PostgreSQL audit schema to track all transformations,
validate data integrity, and enable rollback/recovery.

Usage:
    from src.audit.audit_logger import AuditLogger
    
    logger = AuditLogger()
    audit_id = logger.log_etl_start(
        dag_id="my_dag",
        task_id="my_task",
        source_table="bronze.raw_data",
        target_table="silver.features"
    )
    
    try:
        # Do ETL work
        rows_inserted = 1000
        rows_target = 1000
        
        logger.log_etl_success(
            audit_id=audit_id,
            rows_inserted=rows_inserted,
            rows_target=rows_target
        )
    except Exception as e:
        logger.log_etl_failure(audit_id, str(e))
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Manages audit logging for ETL operations.
    Logs to PostgreSQL audit schema.
    """
    
    def __init__(self, host: str = "postgres", user: str = "ai_quant", 
                 password: str = "ai_quant", db: str = "ai_quant"):
        """Initialize audit logger with database connection."""
        self.conn_params = {
            "host": host,
            "user": user,
            "password": password,
            "database": db
        }
    
    def _get_connection(self):
        """Get database connection."""
        try:
            conn = psycopg2.connect(**self.conn_params)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def log_etl_start(self, dag_id: str, task_id: str, source_table: str, 
                     target_table: str, dag_run_id: str = None, 
                     operation_type: str = "TRANSFORM") -> int:
        """
        Log the start of an ETL operation.
        
        Args:
            dag_id: Airflow DAG ID
            task_id: Airflow Task ID
            source_table: Source table name
            target_table: Target table name
            dag_run_id: Airflow DAG run ID
            operation_type: Type of operation (TRANSFORM, INGESTION, etc.)
        
        Returns:
            audit_id: Unique identifier for this ETL operation
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT audit.sp_log_etl_start(
                        %s, %s, %s, %s, %s, %s, %s
                    )
                """, (dag_id, task_id, datetime.now(), dag_run_id, 
                      source_table, target_table, operation_type))
                
                audit_id = cur.fetchone()[0]
                conn.commit()
                
                logger.info(f"ETL START: audit_id={audit_id}, task={task_id}, "
                           f"{source_table} -> {target_table}")
                return audit_id
        finally:
            conn.close()
    
    def log_etl_success(self, audit_id: int, rows_inserted: int = None,
                       rows_updated: int = None, rows_deleted: int = None,
                       rows_source: int = None, rows_target: int = None,
                       data_volume_mb: float = None):
        """
        Log successful completion of ETL task.
        
        Args:
            audit_id: Audit ID from log_etl_start
            rows_inserted: Number of rows inserted
            rows_updated: Number of rows updated
            rows_deleted: Number of rows deleted
            rows_source: Total rows read from source
            rows_target: Total rows in target after operation
            data_volume_mb: Data volume processed in MB
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT audit.sp_log_etl_end(
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (audit_id, "SUCCESS", rows_inserted, rows_updated,
                      rows_deleted, rows_source, rows_target, None, None))
                
                conn.commit()
                
                logger.info(f"ETL SUCCESS: audit_id={audit_id}, "
                           f"inserted={rows_inserted}, target_rows={rows_target}")
        finally:
            conn.close()
    
    def log_etl_failure(self, audit_id: int, error_message: str, 
                       error_stacktrace: str = None):
        """
        Log failure of ETL task.
        
        Args:
            audit_id: Audit ID from log_etl_start
            error_message: Error message
            error_stacktrace: Full error stacktrace
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT audit.sp_log_etl_end(
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (audit_id, "FAILED", None, None, None, None, None,
                      error_message, error_stacktrace))
                
                conn.commit()
                
                logger.error(f"ETL FAILED: audit_id={audit_id}, error={error_message}")
        finally:
            conn.close()
    
    def reconcile_tables(self, audit_id: int, source_table: str, 
                        target_table: str, 
                        source_key_columns: list = None) -> Dict:
        """
        Reconcile source and target tables.
        Validates that row counts match within threshold.
        
        Args:
            audit_id: Audit ID to link reconciliation
            source_table: Source table name
            target_table: Target table name
            source_key_columns: Primary key columns for validation
        
        Returns:
            Reconciliation result with match status
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # check existence of source/target prior to calling proc
                cur.execute("SELECT to_regclass(%s) AS exists_name", (source_table,))
                row = cur.fetchone()
                if row is None or row.get("exists_name") is None:
                    logger.warning(f"Source table {source_table} does not exist")
                cur.execute("SELECT to_regclass(%s) AS exists_name", (target_table,))
                row = cur.fetchone()
                if row is None or row.get("exists_name") is None:
                    logger.warning(f"Target table {target_table} does not exist")

                key_columns_sql = f"ARRAY{source_key_columns}::TEXT[]" if source_key_columns else "NULL"
                
                cur.execute(f"""
                    SELECT source_count, target_count, row_difference, 
                           match, variance_percent
                    FROM audit.sp_reconcile_tables(
                        %s, %s, {key_columns_sql}, %s
                    )
                """, (source_table, target_table, audit_id))
                
                result = cur.fetchone()
                conn.commit()
                
                status = "PASS" if result["match"] else "WARN" if result["variance_percent"] < 5 else "FAIL"
                
                logger.info(f"RECONCILIATION: {source_table} -> {target_table}, "
                           f"Status={status}, Variance={result['variance_percent']}%")
                
                return {
                    "source_count": result["source_count"],
                    "target_count": result["target_count"],
                    "difference": result["row_difference"],
                    "variance_percent": float(result["variance_percent"]),
                    "status": status,
                    "match": result["match"]
                }
        finally:
            conn.close()
    
    def check_data_quality(self, table_name: str, 
                          completeness_threshold: float = 95.0,
                          uniqueness_threshold: float = 99.0) -> Dict:
        """
        Check data quality metrics for a table.
        
        Args:
            table_name: Table to check
            completeness_threshold: Minimum % non-null values required
            uniqueness_threshold: Minimum % unique rows required
        
        Returns:
            Data quality metrics and pass/fail status
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get latest quality metrics
                cur.execute("""
                    SELECT 
                        total_rows, 
                        null_rows,
                        duplicate_rows,
                        data_completeness_percent,
                        data_uniqueness_percent,
                        quality_score,
                        quality_status,
                        issue_count,
                        issue_details
                    FROM audit.data_quality_metrics
                    WHERE table_name = %s
                    ORDER BY check_date DESC, check_timestamp DESC
                    LIMIT 1
                """, (table_name,))
                
                result = cur.fetchone()
                
                if result:
                    completeness_pass = result["data_completeness_percent"] >= completeness_threshold
                    uniqueness_pass = result["data_uniqueness_percent"] >= uniqueness_threshold
                    overall_pass = completeness_pass and uniqueness_pass
                    
                    logger.info(f"DATA QUALITY: {table_name}, "
                               f"Score={result['quality_score']}, Status={result['quality_status']}")
                    
                    return {
                        "table": table_name,
                        "total_rows": result["total_rows"],
                        "null_rows": result["null_rows"],
                        "duplicate_rows": result["duplicate_rows"],
                        "completeness_percent": float(result["data_completeness_percent"]),
                        "uniqueness_percent": float(result["data_uniqueness_percent"]),
                        "quality_score": float(result["quality_score"]),
                        "quality_status": result["quality_status"],
                        "issue_count": result["issue_count"],
                        "overall_pass": overall_pass,
                        "completeness_pass": completeness_pass,
                        "uniqueness_pass": uniqueness_pass
                    }
                else:
                    logger.warning(f"No quality metrics found for {table_name}")
                    return {"table": table_name, "status": "NO_DATA"}
        finally:
            conn.close()
    
    def get_audit_summary(self, days: int = 7) -> Dict:
        """
        Get audit summary for last N days.
        
        Args:
            days: Number of days to summarize
        
        Returns:
            Summary of ETL operations
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        run_date,
                        total_tasks,
                        successful_tasks,
                        failed_tasks,
                        partial_tasks,
                        total_rows_inserted,
                        total_rows_updated,
                        total_rows_deleted,
                        avg_duration_seconds,
                        max_duration_seconds
                    FROM audit.v_audit_summary
                    WHERE run_date >= CURRENT_DATE - %s
                    ORDER BY run_date DESC
                """, (days,))
                
                results = cur.fetchall()
                
                summary = {
                    "period_days": days,
                    "total_runs": len(results),
                    "successful_runs": sum(1 for r in results if r["failed_tasks"] == 0),
                    "failed_runs": sum(1 for r in results if r["failed_tasks"] > 0),
                    "details": [dict(r) for r in results]
                }
                
                logger.info(f"AUDIT SUMMARY: {summary['successful_runs']}/{summary['total_runs']} successful runs")
                return summary
        finally:
            conn.close()
    
    def get_failed_tasks(self, days: int = 7) -> list:
        """
        Get list of failed tasks in last N days.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of failed tasks
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        dag_id, dag_run_id, task_id, source_table, target_table,
                        start_time, error_message, retry_count, status
                    FROM audit.v_failed_tasks
                    WHERE start_time >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                    ORDER BY start_time DESC
                """, (days,))
                
                results = [dict(r) for r in cur.fetchall()]
                
                logger.info(f"FAILED TASKS: {len(results)} failures in last {days} days")
                return results
        finally:
            conn.close()


def audit_etl_operation(source_table: str, target_table: str, 
                       operation_type: str = "TRANSFORM"):
    """
    Decorator for auditing ETL operations.
    Automatically logs start/end and captures success/failure.
    
    Usage:
        @audit_etl_operation("bronze.raw_data", "silver.features")
        def my_etl_task(**context):
            # ETL logic here
            return {"rows_inserted": 1000}
    """
    def decorator(func):
        def wrapper(*args, **context):
            dag_id = context.get("dag").dag_id if "dag" in context else "unknown"
            task_id = context.get("task").task_id if "task" in context else func.__name__
            dag_run_id = context.get("dag_run").run_id if "dag_run" in context else None
            
            audit = AuditLogger()
            audit_id = audit.log_etl_start(
                dag_id=dag_id,
                task_id=task_id,
                source_table=source_table,
                target_table=target_table,
                dag_run_id=dag_run_id,
                operation_type=operation_type
            )
            
            try:
                result = func(*args, **context)
                
                rows_inserted = result.get("rows_inserted", 0) if isinstance(result, dict) else None
                rows_target = result.get("rows_target", 0) if isinstance(result, dict) else None
                
                audit.log_etl_success(
                    audit_id=audit_id,
                    rows_inserted=rows_inserted,
                    rows_target=rows_target
                )
                
                # Reconcile tables
                audit.reconcile_tables(audit_id, source_table, target_table)
                
                # Check data quality
                audit.check_data_quality(target_table)
                
                return result
            except Exception as e:
                audit.log_etl_failure(audit_id, str(e), repr(e.__traceback__))
                raise
        
        return wrapper
    return decorator
