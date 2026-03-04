/*
================================================================================
AUDIT & RECONCILIATION FRAMEWORK
AI Quant Investment Engine - Database Audit Layer
================================================================================
Purpose: Track all data transformations, validate data integrity, and enable
         rollback/recovery of failed ETL processes.

Tables:
  - etl_audit_log       → Track all ETL operations (start/end/status)
  - etl_reconciliation  → Source-target row count validation
  - data_quality_checks → Quality metrics per table
  - table_lineage       → Data flow from bronze → silver → gold
  - change_log          → Track INSERT/UPDATE/DELETE operations
  - process_audit       → DAG/task execution tracking
  
Views:
  - v_audit_summary     → Daily audit summary
  - v_reconciliation    → Data validation status
  - v_data_quality      → Quality metrics dashboard
  
Procedures:
  - sp_log_etl_start    → Log ETL start
  - sp_log_etl_end      → Log ETL completion
  - sp_reconcile_tables → Validate source-target match
  - sp_check_data_quality → Run quality checks
  
================================================================================
*/

-- Create audit schema
CREATE SCHEMA IF NOT EXISTS audit;
ALTER SCHEMA audit OWNER TO ai_quant;

-- ============================================================================
-- 1. ETL AUDIT LOG TABLE
-- ============================================================================
-- Tracks all ETL operations: what ran, when, how long, success/failure
CREATE TABLE IF NOT EXISTS audit.etl_audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    dag_id VARCHAR(255),
    task_id VARCHAR(255),
    execution_date TIMESTAMP WITH TIME ZONE,
    dag_run_id VARCHAR(255),
    
    source_table VARCHAR(255) NOT NULL,
    target_table VARCHAR(255) NOT NULL,
    operation_type VARCHAR(50), -- INSERT, UPDATE, DELETE, MERGE, TRANSFORM
    
    start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds NUMERIC(10, 2),
    
    status VARCHAR(20), -- RUNNING, SUCCESS, FAILED, PARTIAL
    error_message TEXT,
    error_stacktrace TEXT,
    
    rows_inserted BIGINT,
    rows_updated BIGINT,
    rows_deleted BIGINT,
    rows_source BIGINT,
    rows_target_before BIGINT,
    rows_target_after BIGINT,
    
    data_volume_mb NUMERIC(12, 2),
    execution_host VARCHAR(255),
    executor_memory_gb NUMERIC(5, 2),
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    job_parameters JSONB,
    configuration JSONB,
    
    CONSTRAINT etl_audit_log_source_target CHECK (source_table != target_table)
);

CREATE INDEX idx_etl_audit_task_date ON audit.etl_audit_log(dag_run_id, start_time DESC);
CREATE INDEX idx_etl_audit_tables ON audit.etl_audit_log(source_table, target_table, start_time DESC);
CREATE INDEX idx_etl_audit_status ON audit.etl_audit_log(status, start_time DESC);

-- ============================================================================
-- 2. RECONCILIATION TABLE
-- ============================================================================
-- Validates that source row count matches target after transformation
CREATE TABLE IF NOT EXISTS audit.etl_reconciliation (
    reconciliation_id BIGSERIAL PRIMARY KEY,
    audit_id BIGINT REFERENCES audit.etl_audit_log(audit_id) ON DELETE CASCADE,
    
    source_table VARCHAR(255) NOT NULL,
    target_table VARCHAR(255) NOT NULL,
    reconciliation_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    source_row_count BIGINT NOT NULL,
    target_row_count BIGINT NOT NULL,
    expected_row_count BIGINT, -- If different from source due to filtering
    
    row_count_match BOOLEAN,
    row_count_difference BIGINT,
    row_count_variance_percent NUMERIC(5, 2),
    
    -- Column-level validation
    source_columns BIGINT,
    target_columns BIGINT,
    columns_match BOOLEAN,
    
    -- Key validation
    source_key_columns TEXT[], -- ['id', 'symbol', 'event_time']
    orphaned_rows BIGINT, -- Rows in target without source key
    missing_rows BIGINT,  -- Rows in source not in target
    
    reconciliation_status VARCHAR(20), -- PASS, FAIL, WARN, PENDING
    check_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    checked_by VARCHAR(255), -- System or user
    resolution_notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_reconciliation_tables ON audit.etl_reconciliation(source_table, target_table, reconciliation_date DESC);
CREATE INDEX idx_reconciliation_status ON audit.etl_reconciliation(reconciliation_status, check_timestamp DESC);

-- ============================================================================
-- 3. DATA QUALITY CHECKS TABLE
-- ============================================================================
-- Tracks data quality metrics: null rates, duplicates, outliers, etc.
CREATE TABLE IF NOT EXISTS audit.data_quality_metrics (
    quality_id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    check_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Row-level metrics
    total_rows BIGINT,
    unique_rows BIGINT,
    duplicate_rows BIGINT,
    null_rows BIGINT,
    
    -- Column-level metrics (JSON for flexibility)
    column_quality JSONB, -- {col_name: {null_count, null_percent, unique_count, min, max}}
    
    -- Statistical checks
    data_completeness_percent NUMERIC(5, 2), -- % non-null values
    data_uniqueness_percent NUMERIC(5, 2),  -- % unique rows
    data_validity_percent NUMERIC(5, 2),    -- % passing validation rules
    timeliness_percent NUMERIC(5, 2),       -- % within expected time range
    
    -- Data profile
    min_value NUMERIC,
    max_value NUMERIC,
    mean_value NUMERIC(12, 2),
    stddev_value NUMERIC(12, 2),
    
    -- Issues found
    issues_found BOOLEAN,
    issue_count BIGINT DEFAULT 0,
    issue_details JSONB, -- [{"issue": "null_values", "count": 100, "percent": 0.5}]
    
    quality_score NUMERIC(5, 2), -- 0-100
    quality_status VARCHAR(20), -- EXCELLENT, GOOD, FAIR, POOR
    
    check_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_quality_table_date ON audit.data_quality_metrics(table_name, check_date DESC);
CREATE INDEX idx_quality_status ON audit.data_quality_metrics(quality_status, check_date DESC);

-- ============================================================================
-- 4. TABLE LINEAGE / DATA FLOW TRACKING
-- ============================================================================
-- Tracks data transformation lineage: bronze → silver → gold
CREATE TABLE IF NOT EXISTS audit.table_lineage (
    lineage_id BIGSERIAL PRIMARY KEY,
    source_schema VARCHAR(100),
    source_table VARCHAR(255),
    target_schema VARCHAR(100),
    target_table VARCHAR(255),
    
    transformation_type VARCHAR(50), -- INGESTION, TRANSFORMATION, AGGREGATION, ML_FEATURE
    
    join_keys TEXT[], -- Column names used to join source to target
    filter_criteria TEXT, -- WHERE clause applied
    aggregation_columns TEXT[], -- GROUP BY columns
    
    -- Data lineage
    input_row_estimate BIGINT,
    output_row_estimate BIGINT,
    data_loss_percent NUMERIC(5, 2), -- If rows filtered out
    
    dag_id VARCHAR(255),
    task_id VARCHAR(255),
    created_date DATE NOT NULL DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_lineage_source_target ON audit.table_lineage(source_table, target_table);
CREATE INDEX idx_lineage_active ON audit.table_lineage(is_active, created_date DESC);

-- ============================================================================
-- 5. CHANGE LOG TABLE
-- ============================================================================
-- CDC-style change tracking for audit trail
CREATE TABLE IF NOT EXISTS audit.change_log (
    change_id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    record_id JSONB, -- {'id': 123, 'symbol': 'AAPL'} - the key
    
    change_type VARCHAR(20), -- INSERT, UPDATE, DELETE
    old_values JSONB, -- Full row before change
    new_values JSONB, -- Full row after change
    changed_columns TEXT[], -- Only columns that changed
    
    change_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    batch_id BIGINT, -- ETL batch that caused the change
    changed_by VARCHAR(255), -- User or system (e.g., 'airflow-spark_etl_mutual_funds')
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_change_log_table_time ON audit.change_log(table_name, change_timestamp DESC);
CREATE INDEX idx_change_log_batch ON audit.change_log(batch_id);
CREATE INDEX idx_change_log_type ON audit.change_log(change_type, change_timestamp DESC);

-- ============================================================================
-- 6. PROCESS AUDIT TABLE
-- ============================================================================
-- Track DAG/task execution with retry counts, dependencies, etc.
CREATE TABLE IF NOT EXISTS audit.process_audit (
    process_id BIGSERIAL PRIMARY KEY,
    dag_id VARCHAR(255) NOT NULL,
    dag_run_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    
    -- Execution details
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_seconds NUMERIC(10, 2),
    
    -- Status tracking
    status VARCHAR(20), -- SCHEDULED, RUNNING, SUCCESS, FAILED, RETRYING, SKIPPED
    retry_count SMALLINT DEFAULT 0,
    max_retries SMALLINT,
    
    -- Resource usage
    cpu_percent NUMERIC(5, 2),
    memory_mb NUMERIC(10, 2),
    memory_peak_mb NUMERIC(10, 2),
    disk_read_mb NUMERIC(12, 2),
    disk_write_mb NUMERIC(12, 2),
    
    -- Upstream/downstream
    upstream_tasks TEXT[],
    downstream_tasks TEXT[],
    
    -- Failure info
    error_code VARCHAR(100),
    error_message TEXT,
    log_location VARCHAR(500),
    
    -- Metadata
    executor_id VARCHAR(255),
    task_params JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_process_audit_dag_run ON audit.process_audit(dag_id, dag_run_id, task_id);
CREATE INDEX idx_process_audit_status ON audit.process_audit(status, start_time DESC);
CREATE INDEX idx_process_audit_failures ON audit.process_audit(status, end_time DESC) WHERE status IN ('FAILED', 'RETRYING');

-- ============================================================================
-- 7. AUDIT VIEWS
-- ============================================================================

-- Summary of recent ETL runs
CREATE OR REPLACE VIEW audit.v_audit_summary AS
SELECT 
    DATE(start_time) as run_date,
    COUNT(*) as total_tasks,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_tasks,
    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_tasks,
    SUM(CASE WHEN status = 'PARTIAL' THEN 1 ELSE 0 END) as partial_tasks,
    SUM(rows_inserted) as total_rows_inserted,
    SUM(rows_updated) as total_rows_updated,
    SUM(rows_deleted) as total_rows_deleted,
    AVG(duration_seconds) as avg_duration_seconds,
    MAX(duration_seconds) as max_duration_seconds
FROM audit.etl_audit_log
WHERE start_time >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(start_time)
ORDER BY run_date DESC;

-- Reconciliation dashboard
CREATE OR REPLACE VIEW audit.v_reconciliation_status AS
SELECT 
    source_table,
    target_table,
    reconciliation_date,
    source_row_count,
    target_row_count,
    row_count_difference,
    row_count_variance_percent,
    reconciliation_status,
    CASE 
        WHEN row_count_match THEN 'PASS' 
        WHEN row_count_variance_percent < 1 THEN 'WARN'
        ELSE 'FAIL'
    END as data_integrity_status,
    check_timestamp,
    resolution_notes
FROM audit.etl_reconciliation
WHERE reconciliation_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY reconciliation_date DESC, source_table;

-- Data quality dashboard
CREATE OR REPLACE VIEW audit.v_data_quality_summary AS
SELECT 
    table_name,
    check_date,
    total_rows,
    data_completeness_percent,
    data_uniqueness_percent,
    data_validity_percent,
    quality_score,
    quality_status,
    issue_count,
    issue_details,
    check_timestamp
FROM audit.data_quality_metrics
WHERE check_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY check_date DESC, quality_score ASC;

-- Failed tasks summary
CREATE OR REPLACE VIEW audit.v_failed_tasks AS
SELECT 
    dag_id,
    dag_run_id,
    task_id,
    start_time,
    end_time,
    duration_seconds,
    error_message,
    retry_count,
    status
FROM audit.process_audit
WHERE status IN ('FAILED', 'RETRYING')
  AND start_time >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY start_time DESC;

-- ============================================================================
-- 8. STORED PROCEDURES
-- ============================================================================

-- Log ETL task start
CREATE OR REPLACE FUNCTION audit.sp_log_etl_start(
    p_dag_id VARCHAR,
    p_task_id VARCHAR,
    p_execution_date TIMESTAMP,
    p_dag_run_id VARCHAR,
    p_source_table VARCHAR,
    p_target_table VARCHAR,
    p_operation_type VARCHAR DEFAULT 'TRANSFORM'
)
RETURNS BIGINT AS $$
DECLARE
    v_audit_id BIGINT;
BEGIN
    INSERT INTO audit.etl_audit_log (
        dag_id, task_id, execution_date, dag_run_id,
        source_table, target_table, operation_type,
        status, start_time
    ) VALUES (
        p_dag_id, p_task_id, p_execution_date, p_dag_run_id,
        p_source_table, p_target_table, p_operation_type,
        'RUNNING', CURRENT_TIMESTAMP
    )
    RETURNING audit_id INTO v_audit_id;
    
    RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql;

-- Log ETL task end with metrics
CREATE OR REPLACE FUNCTION audit.sp_log_etl_end(
    p_audit_id BIGINT,
    p_status VARCHAR,
    p_rows_inserted BIGINT DEFAULT NULL,
    p_rows_updated BIGINT DEFAULT NULL,
    p_rows_deleted BIGINT DEFAULT NULL,
    p_rows_source BIGINT DEFAULT NULL,
    p_rows_target_after BIGINT DEFAULT NULL,
    p_error_message TEXT DEFAULT NULL,
    p_error_stacktrace TEXT DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_start_time TIMESTAMP;
    v_duration NUMERIC;
BEGIN
    SELECT start_time INTO v_start_time 
    FROM audit.etl_audit_log 
    WHERE audit_id = p_audit_id;
    
    v_duration := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - v_start_time)) / 60.0;
    
    UPDATE audit.etl_audit_log
    SET
        end_time = CURRENT_TIMESTAMP,
        duration_seconds = v_duration,
        status = p_status,
        rows_inserted = p_rows_inserted,
        rows_updated = p_rows_updated,
        rows_deleted = p_rows_deleted,
        rows_source = p_rows_source,
        rows_target_after = p_rows_target_after,
        error_message = p_error_message,
        error_stacktrace = p_error_stacktrace,
        updated_at = CURRENT_TIMESTAMP
    WHERE audit_id = p_audit_id;
    
END;
$$ LANGUAGE plpgsql;

-- Reconcile source and target tables
CREATE OR REPLACE FUNCTION audit.sp_reconcile_tables(
    p_source_table VARCHAR,
    p_target_table VARCHAR,
    p_source_key_columns TEXT[] DEFAULT NULL,
    p_audit_id BIGINT DEFAULT NULL
)
RETURNS TABLE(
    source_count BIGINT,
    target_count BIGINT,
    row_difference BIGINT,
    match BOOLEAN,
    variance_percent NUMERIC
) AS $$
DECLARE
    v_source_count BIGINT;
    v_target_count BIGINT;
    v_target_count_before BIGINT;
    v_difference BIGINT;
    v_variance NUMERIC;
BEGIN
    -- Get source row count: catch undefined table errors
    BEGIN
        EXECUTE format('SELECT COUNT(*) FROM %I', p_source_table) INTO v_source_count;
    EXCEPTION WHEN undefined_table THEN
        v_source_count := 0;
    END;
    
    -- Get target row count: catch undefined table errors
    BEGIN
        EXECUTE format('SELECT COUNT(*) FROM %I', p_target_table) INTO v_target_count;
    EXCEPTION WHEN undefined_table THEN
        v_target_count := 0;
    END;
    
    v_difference := v_target_count - v_source_count;
    v_variance := CASE 
        WHEN v_source_count = 0 THEN 0
        ELSE (ABS(v_difference)::NUMERIC / v_source_count) * 100
    END;
    
    -- Log reconciliation
    INSERT INTO audit.etl_reconciliation (
        audit_id, source_table, target_table, reconciliation_date,
        source_row_count, target_row_count, expected_row_count,
        row_count_match, row_count_difference, row_count_variance_percent,
        reconciliation_status, source_key_columns
    ) VALUES (
        p_audit_id, p_source_table, p_target_table, CURRENT_DATE,
        v_source_count, v_target_count, v_source_count,
        (v_variance < 1), v_difference, v_variance,
        CASE WHEN v_variance < 0.5 THEN 'PASS'
             WHEN v_variance < 5 THEN 'WARN'
             ELSE 'FAIL' END,
        p_source_key_columns
    );
    
    RETURN QUERY SELECT v_source_count, v_target_count, v_difference, 
                        (v_variance < 1), v_variance;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 9. UTILITY FUNCTIONS
-- ============================================================================

-- Generate daily audit report
CREATE OR REPLACE FUNCTION audit.fn_generate_daily_report(p_report_date DATE DEFAULT CURRENT_DATE)
RETURNS TABLE(
    report_section VARCHAR,
    metric_name VARCHAR,
    metric_value VARCHAR
) AS $$
BEGIN
    -- ETL Summary
    RETURN QUERY
    SELECT 'ETL Summary'::VARCHAR, 'Total Tasks Run'::VARCHAR, 
           COUNT(*)::VARCHAR FROM audit.etl_audit_log 
    WHERE DATE(start_time) = p_report_date;
    
    RETURN QUERY
    SELECT 'ETL Summary'::VARCHAR, 'Successful Tasks'::VARCHAR,
           COUNT(*)::VARCHAR FROM audit.etl_audit_log
    WHERE DATE(start_time) = p_report_date AND status = 'SUCCESS';
    
    RETURN QUERY
    SELECT 'ETL Summary'::VARCHAR, 'Failed Tasks'::VARCHAR,
           COUNT(*)::VARCHAR FROM audit.etl_audit_log
    WHERE DATE(start_time) = p_report_date AND status = 'FAILED';
    
    -- Data Quality Summary
    RETURN QUERY
    SELECT 'Data Quality'::VARCHAR, 'Tables Checked'::VARCHAR,
           COUNT(DISTINCT table_name)::VARCHAR FROM audit.data_quality_metrics
    WHERE check_date = p_report_date;
    
    RETURN QUERY
    SELECT 'Data Quality'::VARCHAR, 'Tables with Issues'::VARCHAR,
           COUNT(*)::VARCHAR FROM audit.data_quality_metrics
    WHERE check_date = p_report_date AND issues_found = TRUE;
    
    -- Reconciliation Summary
    RETURN QUERY
    SELECT 'Reconciliation'::VARCHAR, 'Passed Checks'::VARCHAR,
           COUNT(*)::VARCHAR FROM audit.etl_reconciliation
    WHERE reconciliation_date = p_report_date AND reconciliation_status = 'PASS';
    
    RETURN QUERY
    SELECT 'Reconciliation'::VARCHAR, 'Failed Checks'::VARCHAR,
           COUNT(*)::VARCHAR FROM audit.etl_reconciliation
    WHERE reconciliation_date = p_report_date AND reconciliation_status = 'FAIL';
    
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 10. PERMISSIONS
-- ============================================================================

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA audit TO ai_quant;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO ai_quant;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA audit TO ai_quant;

-- Create read-only role for auditing (if not exists)
DO $$
BEGIN
    CREATE ROLE audit_viewer NOLOGIN;
    EXCEPTION WHEN duplicate_object THEN NULL;
END
$$;
GRANT USAGE ON SCHEMA audit TO audit_viewer;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO audit_viewer;

-- ============================================================================
-- 11. INITIAL DOCUMENTATION
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit.schema_documentation (
    doc_id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    data_type VARCHAR(100),
    description TEXT,
    business_purpose TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO audit.schema_documentation VALUES
(DEFAULT, 'etl_audit_log', 'audit_id', 'BIGINT', 'Unique audit identifier', 'Primary key for ETL operations'),
(DEFAULT, 'etl_audit_log', 'status', 'VARCHAR(20)', 'SUCCESS, FAILED, RUNNING, PARTIAL', 'Status of ETL operation'),
(DEFAULT, 'etl_audit_log', 'rows_inserted', 'BIGINT', 'Count of rows inserted', 'Rows added to target'),
(DEFAULT, 'etl_reconciliation', 'row_count_match', 'BOOLEAN', 'True if source=target rows', 'Data integrity validation'),
(DEFAULT, 'data_quality_metrics', 'quality_score', 'NUMERIC(5,2)', '0-100 score', 'Overall data quality rating'),
(DEFAULT, 'process_audit', 'status', 'VARCHAR(20)', 'Task execution status', 'SCHEDULED, RUNNING, SUCCESS, FAILED')
ON CONFLICT DO NOTHING;

COMMIT;
