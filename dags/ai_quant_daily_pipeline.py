from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import timedelta
import pendulum

from src.ingestion.binance_ingest import ingest_binance_data
from src.etl.pyspark_etl import run_spark_etl
from src.recommendation.strategy_engine import generate_signals
from src.recommendation.train_model import train_all_horizons
from src.recommendation.email_alert import send_opportunity_email
from src.recommendation.pipeline_guards import (
    assert_source_freshness_sla,
    assert_model_promotion_gate,
)
from src.audit.audit_tasks import (
    audit_etl_task,
    quality_check_task,
    reconcile_task,
    generate_daily_audit_report
)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

from src.ingestion.yfinance_ingest import ingest_nifty50, ingest_nifty_midcap, ingest_nifty_smallcap
from src.ingestion.mfapi_ingest import ingest_mutual_funds

with DAG(
    dag_id="ai_quant_daily_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="30 2 * * 1-5",  # 8:00 AM IST, Monday-Friday
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["ai_quant", "investing", "etl", "ml"],
) as dag:

    # --- CRYPTO PIPELINE ---
    ingest_crypto = PythonOperator(
        task_id="ingest_binance_crypto",
        python_callable=ingest_binance_data,
    )
    etl_crypto = PythonOperator(
        task_id="spark_etl_crypto",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "bronze.crypto_price_raw", "dest_table": "silver.crypto_features_daily"}
    )
    freshness_crypto = PythonOperator(
        task_id="freshness_sla_crypto",
        python_callable=assert_source_freshness_sla,
        op_kwargs={
            "source_table": "silver.crypto_features_daily",
            "max_age_hours": 30,
            "min_rows": 500,
        },
        execution_timeout=timedelta(minutes=15),
    )
    train_crypto = PythonOperator(
        task_id="train_model_crypto",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "silver.crypto_features_daily"}
    )
    promote_crypto = PythonOperator(
        task_id="promotion_gate_crypto",
        python_callable=assert_model_promotion_gate,
        op_kwargs={
            "source_table": "silver.crypto_features_daily",
            "required_horizons": "1y",
            "max_report_age_days": 10,
        },
        execution_timeout=timedelta(minutes=15),
    )
    signal_crypto = PythonOperator(
        task_id="generate_signals_crypto",
        python_callable=generate_signals,
        op_kwargs={"source_table": "silver.crypto_features_daily", "dest_table": "gold.crypto_investment_signals"}
    )
    audit_crypto = PythonOperator(
        task_id="audit_etl_crypto",
        python_callable=audit_etl_task,
        op_kwargs={"asset_class": "crypto", "source_table": "silver.crypto_features_daily", "target_table": "gold.crypto_investment_signals"}
    )
    quality_crypto = PythonOperator(
        task_id="quality_check_crypto",
        python_callable=quality_check_task,
        op_kwargs={"asset_class": "crypto", "table_name": "gold.crypto_investment_signals"}
    )
    reconcile_crypto = PythonOperator(
        task_id="reconcile_crypto",
        python_callable=reconcile_task,
        op_kwargs={"asset_class": "crypto", "source_table": "silver.crypto_features_daily", "target_table": "gold.crypto_investment_signals"}
    )

    # --- NIFTY 50 PIPELINE ---
    ingest_stocks = PythonOperator(
        task_id="ingest_yfinance_nifty50",
        python_callable=ingest_nifty50,
    )
    etl_stocks = PythonOperator(
        task_id="spark_etl_nifty50",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "bronze.nifty50_price_raw", "dest_table": "silver.nifty50_features_daily"}
    )
    freshness_stocks = PythonOperator(
        task_id="freshness_sla_nifty50",
        python_callable=assert_source_freshness_sla,
        op_kwargs={
            "source_table": "silver.nifty50_features_daily",
            "max_age_hours": 96,
            "min_rows": 500,
        },
        execution_timeout=timedelta(minutes=15),
    )
    train_stocks = PythonOperator(
        task_id="train_model_nifty50",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "silver.nifty50_features_daily"}
    )
    promote_stocks = PythonOperator(
        task_id="promotion_gate_nifty50",
        python_callable=assert_model_promotion_gate,
        op_kwargs={
            "source_table": "silver.nifty50_features_daily",
            "required_horizons": "1y",
            "max_report_age_days": 10,
        },
        execution_timeout=timedelta(minutes=15),
    )
    signal_stocks = PythonOperator(
        task_id="generate_signals_nifty50",
        python_callable=generate_signals,
        op_kwargs={"source_table": "silver.nifty50_features_daily", "dest_table": "gold.nifty50_investment_signals"}
    )
    audit_stocks = PythonOperator(
        task_id="audit_etl_nifty50",
        python_callable=audit_etl_task,
        op_kwargs={"asset_class": "nifty50", "source_table": "silver.nifty50_features_daily", "target_table": "gold.nifty50_investment_signals"}
    )
    quality_stocks = PythonOperator(
        task_id="quality_check_nifty50",
        python_callable=quality_check_task,
        op_kwargs={"asset_class": "nifty50", "table_name": "gold.nifty50_investment_signals"}
    )
    reconcile_stocks = PythonOperator(
        task_id="reconcile_nifty50",
        python_callable=reconcile_task,
        op_kwargs={"asset_class": "nifty50", "source_table": "silver.nifty50_features_daily", "target_table": "gold.nifty50_investment_signals"}
    )

    # --- NIFTY MID CAP PIPELINE ---
    ingest_midcap = PythonOperator(
        task_id="ingest_yfinance_nifty_midcap",
        python_callable=ingest_nifty_midcap,
    )
    etl_midcap = PythonOperator(
        task_id="spark_etl_nifty_midcap",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "bronze.nifty_midcap_price_raw", "dest_table": "silver.nifty_midcap_features_daily"}
    )
    freshness_midcap = PythonOperator(
        task_id="freshness_sla_nifty_midcap",
        python_callable=assert_source_freshness_sla,
        op_kwargs={
            "source_table": "silver.nifty_midcap_features_daily",
            "max_age_hours": 96,
            "min_rows": 500,
        },
        execution_timeout=timedelta(minutes=15),
    )
    train_midcap = PythonOperator(
        task_id="train_model_nifty_midcap",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "silver.nifty_midcap_features_daily"}
    )
    promote_midcap = PythonOperator(
        task_id="promotion_gate_nifty_midcap",
        python_callable=assert_model_promotion_gate,
        op_kwargs={
            "source_table": "silver.nifty_midcap_features_daily",
            "required_horizons": "1y",
            "max_report_age_days": 10,
        },
        execution_timeout=timedelta(minutes=15),
    )
    signal_midcap = PythonOperator(
        task_id="generate_signals_nifty_midcap",
        python_callable=generate_signals,
        op_kwargs={"source_table": "silver.nifty_midcap_features_daily", "dest_table": "gold.nifty_midcap_investment_signals"}
    )
    audit_midcap = PythonOperator(
        task_id="audit_etl_nifty_midcap",
        python_callable=audit_etl_task,
        op_kwargs={"asset_class": "nifty_midcap", "source_table": "silver.nifty_midcap_features_daily", "target_table": "gold.nifty_midcap_investment_signals"}
    )
    quality_midcap = PythonOperator(
        task_id="quality_check_nifty_midcap",
        python_callable=quality_check_task,
        op_kwargs={"asset_class": "nifty_midcap", "table_name": "gold.nifty_midcap_investment_signals"}
    )
    reconcile_midcap = PythonOperator(
        task_id="reconcile_nifty_midcap",
        python_callable=reconcile_task,
        op_kwargs={"asset_class": "nifty_midcap", "source_table": "silver.nifty_midcap_features_daily", "target_table": "gold.nifty_midcap_investment_signals"}
    )

    # --- NIFTY SMALL CAP PIPELINE ---
    ingest_smallcap = PythonOperator(
        task_id="ingest_yfinance_nifty_smallcap",
        python_callable=ingest_nifty_smallcap,
    )
    etl_smallcap = PythonOperator(
        task_id="spark_etl_nifty_smallcap",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "bronze.nifty_smallcap_price_raw", "dest_table": "silver.nifty_smallcap_features_daily"}
    )
    freshness_smallcap = PythonOperator(
        task_id="freshness_sla_nifty_smallcap",
        python_callable=assert_source_freshness_sla,
        op_kwargs={
            "source_table": "silver.nifty_smallcap_features_daily",
            "max_age_hours": 96,
            "min_rows": 500,
        },
        execution_timeout=timedelta(minutes=15),
    )
    train_smallcap = PythonOperator(
        task_id="train_model_nifty_smallcap",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "silver.nifty_smallcap_features_daily"}
    )
    promote_smallcap = PythonOperator(
        task_id="promotion_gate_nifty_smallcap",
        python_callable=assert_model_promotion_gate,
        op_kwargs={
            "source_table": "silver.nifty_smallcap_features_daily",
            "required_horizons": "1y",
            "max_report_age_days": 10,
        },
        execution_timeout=timedelta(minutes=15),
    )
    signal_smallcap = PythonOperator(
        task_id="generate_signals_nifty_smallcap",
        python_callable=generate_signals,
        op_kwargs={"source_table": "silver.nifty_smallcap_features_daily", "dest_table": "gold.nifty_smallcap_investment_signals"}
    )
    audit_smallcap = PythonOperator(
        task_id="audit_etl_nifty_smallcap",
        python_callable=audit_etl_task,
        op_kwargs={"asset_class": "nifty_smallcap", "source_table": "silver.nifty_smallcap_features_daily", "target_table": "gold.nifty_smallcap_investment_signals"}
    )
    quality_smallcap = PythonOperator(
        task_id="quality_check_nifty_smallcap",
        python_callable=quality_check_task,
        op_kwargs={"asset_class": "nifty_smallcap", "table_name": "gold.nifty_smallcap_investment_signals"}
    )
    reconcile_smallcap = PythonOperator(
        task_id="reconcile_nifty_smallcap",
        python_callable=reconcile_task,
        op_kwargs={"asset_class": "nifty_smallcap", "source_table": "silver.nifty_smallcap_features_daily", "target_table": "gold.nifty_smallcap_investment_signals"}
    )

    # --- MUTUAL FUNDS PIPELINE ---
    ingest_mf = PythonOperator(
        task_id="ingest_yfinance_mutual_funds",
        python_callable=ingest_mutual_funds,
    )
    etl_mf = PythonOperator(
        task_id="spark_etl_mutual_funds",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "bronze.mutual_funds_price_raw", "dest_table": "silver.mutual_funds_features_daily"}
    )
    freshness_mf = PythonOperator(
        task_id="freshness_sla_mutual_funds",
        python_callable=assert_source_freshness_sla,
        op_kwargs={
            "source_table": "silver.mutual_funds_features_daily",
            "max_age_hours": 120,
            "min_rows": 500,
        },
        execution_timeout=timedelta(minutes=15),
    )
    train_mf = PythonOperator(
        task_id="train_model_mutual_funds",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "silver.mutual_funds_features_daily"}
    )
    promote_mf = PythonOperator(
        task_id="promotion_gate_mutual_funds",
        python_callable=assert_model_promotion_gate,
        op_kwargs={
            "source_table": "silver.mutual_funds_features_daily",
            "required_horizons": "1y",
            "max_report_age_days": 10,
            "max_high_drift_features": 15,
        },
        execution_timeout=timedelta(minutes=15),
    )
    signal_mf = PythonOperator(
        task_id="generate_signals_mutual_funds",
        python_callable=generate_signals,
        op_kwargs={"source_table": "silver.mutual_funds_features_daily", "dest_table": "gold.mutual_funds_investment_signals"}
    )
    audit_mf = PythonOperator(
        task_id="audit_etl_mutual_funds",
        python_callable=audit_etl_task,
        op_kwargs={"asset_class": "mutual_funds", "source_table": "silver.mutual_funds_features_daily", "target_table": "gold.mutual_funds_investment_signals"}
    )
    quality_mf = PythonOperator(
        task_id="quality_check_mutual_funds",
        python_callable=quality_check_task,
        op_kwargs={"asset_class": "mutual_funds", "table_name": "gold.mutual_funds_investment_signals"}
    )
    reconcile_mf = PythonOperator(
        task_id="reconcile_mutual_funds",
        python_callable=reconcile_task,
        op_kwargs={"asset_class": "mutual_funds", "source_table": "silver.mutual_funds_features_daily", "target_table": "gold.mutual_funds_investment_signals"}
    )

    # --- EMAIL ALERT PIPELINE ---
    audit_report = PythonOperator(
        task_id="generate_daily_audit_report",
        python_callable=generate_daily_audit_report,
    )
    send_email = PythonOperator(
        task_id="send_opportunity_email",
        python_callable=send_opportunity_email,
    )

    # Orchestration: Run lightweight pipelines (crypto & Nifty 50) in parallel.
    # Because we expanded the symbol lists, Midcap (150) and Smallcap (250) and Mutual Funds (thousands) 
    # generate substantial data volumes, so we must run them sequentially to avoid OOM.
    
    start_parallel = EmptyOperator(task_id="start_parallel")
    end_parallel = EmptyOperator(task_id="wait_for_parallel")

    start_parallel >> ingest_crypto >> etl_crypto >> freshness_crypto >> train_crypto >> promote_crypto >> signal_crypto >> audit_crypto >> quality_crypto >> reconcile_crypto >> end_parallel
    start_parallel >> ingest_stocks >> etl_stocks >> freshness_stocks >> train_stocks >> promote_stocks >> signal_stocks >> audit_stocks >> quality_stocks >> reconcile_stocks >> end_parallel

    (
        end_parallel
        >> ingest_midcap >> etl_midcap >> freshness_midcap >> train_midcap >> promote_midcap >> signal_midcap >> audit_midcap >> quality_midcap >> reconcile_midcap
        >> ingest_smallcap >> etl_smallcap >> freshness_smallcap >> train_smallcap >> promote_smallcap >> signal_smallcap >> audit_smallcap >> quality_smallcap >> reconcile_smallcap
        >> ingest_mf >> etl_mf >> freshness_mf >> train_mf >> promote_mf >> signal_mf >> audit_mf >> quality_mf >> reconcile_mf
        >> audit_report >> send_email
    )

    # Also wire parallel audit tasks to final audit report
    reconcile_crypto >> audit_report
    reconcile_stocks >> audit_report
