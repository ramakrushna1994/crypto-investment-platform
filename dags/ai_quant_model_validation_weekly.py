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
    "execution_timeout": timedelta(hours=3),
}

COMMON_EVAL_KWARGS = {
    "model_type": "auto",
    "min_train_days": 504,  # ~2 trading years
    "test_days": 126,       # ~6 trading months
    "step_days": 63,        # ~quarterly step
    "top_n": 10,
    "min_prob": 0.55,
    "trading_cost_bps": 10.0,
    "slippage_bps": 5.0,
    "brokerage_bps": 2.0,
    "tax_bps": 3.0,
    "drift_recent_days": 63,
    "drift_baseline_days": 252,
    "drift_min_samples": 300,
    "thresholds": [0.50, 0.55, 0.60, 0.65, 0.70],
    "output_dir": "/opt/airflow/files/reports",
    "return_results": False,
}


with DAG(
    dag_id="ai_quant_model_validation_weekly",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="0 6 * * 0",  # Weekly on Sunday, 06:00 UTC
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=8),
    default_args=default_args,
    tags=["ai_quant", "validation", "ml"],
) as dag:
    eval_crypto = PythonOperator(
        task_id="walk_forward_crypto",
        python_callable=run_walk_forward_evaluation,
        op_kwargs={**COMMON_EVAL_KWARGS, "source_table": "silver.crypto_features_daily"},
        do_xcom_push=False,
    )

    eval_nifty50 = PythonOperator(
        task_id="walk_forward_nifty50",
        python_callable=run_walk_forward_evaluation,
        op_kwargs={**COMMON_EVAL_KWARGS, "source_table": "silver.nifty50_features_daily"},
        do_xcom_push=False,
    )

    eval_nifty_midcap = PythonOperator(
        task_id="walk_forward_nifty_midcap",
        python_callable=run_walk_forward_evaluation,
        op_kwargs={**COMMON_EVAL_KWARGS, "source_table": "silver.nifty_midcap_features_daily"},
        do_xcom_push=False,
    )

    eval_nifty_smallcap = PythonOperator(
        task_id="walk_forward_nifty_smallcap",
        python_callable=run_walk_forward_evaluation,
        op_kwargs={**COMMON_EVAL_KWARGS, "source_table": "silver.nifty_smallcap_features_daily"},
        do_xcom_push=False,
    )

    validate_reports = PythonOperator(
        task_id="validate_walk_forward_reports",
        python_callable=verify_walk_forward_reports,
        op_kwargs={
            "source_tables": [
                "silver.crypto_features_daily",
                "silver.nifty50_features_daily",
                "silver.nifty_midcap_features_daily",
                "silver.nifty_smallcap_features_daily",
            ],
            "reports_dir": "/opt/airflow/files/reports",
            "max_age_hours": 72,
        },
        execution_timeout=timedelta(minutes=20),
        do_xcom_push=False,
    )

    # Run sequentially to avoid CPU/RAM spikes on local machines.
    eval_crypto >> eval_nifty50 >> eval_nifty_midcap >> eval_nifty_smallcap >> validate_reports
