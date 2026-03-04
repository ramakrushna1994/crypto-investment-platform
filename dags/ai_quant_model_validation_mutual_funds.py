from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import timedelta
import pendulum

from src.recommendation.walk_forward_evaluation import run_walk_forward_evaluation
from src.recommendation.validation_report_checks import verify_walk_forward_reports


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=20),
    "execution_timeout": timedelta(hours=4),
}

MUTUAL_FUNDS_LOCK_POOL = "mf_validation_single_pool"

MUTUAL_FUNDS_EVAL_KWARGS = {
    # Resource-aware profile for large local runs.
    "model_type": "histgb",
    "min_train_days": 756,  # ~3 trading years
    "test_days": 126,       # ~6 trading months
    "step_days": 252,       # yearly step (fewer retrains)
    "top_n": 10,
    "min_prob": 0.55,
    "trading_cost_bps": 10.0,
    "slippage_bps": 5.0,
    "brokerage_bps": 2.0,
    "tax_bps": 3.0,
    "drift_recent_days": 63,
    "drift_baseline_days": 252,
    "drift_min_samples": 300,
    "thresholds": [0.55, 0.60, 0.65],
    "output_dir": "/opt/airflow/files/reports",
    "return_results": False,
}


with DAG(
    dag_id="ai_quant_model_validation_mutual_funds",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 7 * * 0",  # Weekly on Sunday, 07:00 UTC
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    default_args=default_args,
    tags=["ai_quant", "validation", "ml", "mutual_funds", "weekly_light"],
) as dag:
    eval_mutual_funds = PythonOperator(
        task_id="walk_forward_mutual_funds",
        python_callable=run_walk_forward_evaluation,
        op_kwargs={**MUTUAL_FUNDS_EVAL_KWARGS, "source_table": "silver.mutual_funds_features_daily"},
        pool=MUTUAL_FUNDS_LOCK_POOL,
        pool_slots=1,
        do_xcom_push=False,
    )

    validate_reports = PythonOperator(
        task_id="validate_walk_forward_reports",
        python_callable=verify_walk_forward_reports,
        op_kwargs={
            "source_tables": ["silver.mutual_funds_features_daily"],
            "reports_dir": "/opt/airflow/files/reports",
            "max_age_hours": 96,
        },
        execution_timeout=timedelta(minutes=20),
        do_xcom_push=False,
    )

    eval_mutual_funds >> validate_reports
