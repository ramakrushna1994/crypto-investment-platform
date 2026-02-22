from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pendulum

from src.ingestion.binance_ingest import ingest_binance_data
from src.etl.spark_etl import run_spark_etl
from src.recommendation.strategy_engine import generate_signals
from src.recommendation.train_model import train_all_horizons

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

from src.ingestion.yfinance_ingest import ingest_nifty50, ingest_nifty_midcap, ingest_nifty_smallcap
from src.ingestion.mfapi_ingest import ingest_mutual_funds

with DAG(
    dag_id="crypto_daily_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="30 2,14 * * 1-6",  # 8:00 AM & 8:00 PM IST (Mon–Sat only, markets closed Sunday)
    catchup=False,
    default_args=default_args,
    tags=["crypto", "stocks", "etl", "ml"],
) as dag:

    # --- CRYPTO PIPELINE ---
    ingest_crypto = PythonOperator(
        task_id="ingest_binance_crypto",
        python_callable=ingest_binance_data,
    )
    etl_crypto = PythonOperator(
        task_id="spark_etl_crypto",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "public.crypto_price_raw", "dest_table": "public.crypto_features_daily"}
    )
    train_crypto = PythonOperator(
        task_id="train_model_crypto",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "public.crypto_features_daily"}
    )
    signal_crypto = PythonOperator(
        task_id="generate_signals_crypto",
        python_callable=generate_signals,
        op_kwargs={"source_table": "public.crypto_features_daily", "dest_table": "public.crypto_investment_signals"}
    )

    # --- NIFTY 50 PIPELINE ---
    ingest_stocks = PythonOperator(
        task_id="ingest_yfinance_nifty50",
        python_callable=ingest_nifty50,
    )
    etl_stocks = PythonOperator(
        task_id="spark_etl_nifty50",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "public.nifty50_price_raw", "dest_table": "public.nifty50_features_daily"}
    )
    train_stocks = PythonOperator(
        task_id="train_model_nifty50",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "public.nifty50_features_daily"}
    )
    signal_stocks = PythonOperator(
        task_id="generate_signals_nifty50",
        python_callable=generate_signals,
        op_kwargs={"source_table": "public.nifty50_features_daily", "dest_table": "public.nifty50_investment_signals"}
    )

    # --- NIFTY MID CAP PIPELINE ---
    ingest_midcap = PythonOperator(
        task_id="ingest_yfinance_nifty_midcap",
        python_callable=ingest_nifty_midcap,
    )
    etl_midcap = PythonOperator(
        task_id="spark_etl_nifty_midcap",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "public.nifty_midcap_price_raw", "dest_table": "public.nifty_midcap_features_daily"}
    )
    train_midcap = PythonOperator(
        task_id="train_model_nifty_midcap",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "public.nifty_midcap_features_daily"}
    )
    signal_midcap = PythonOperator(
        task_id="generate_signals_nifty_midcap",
        python_callable=generate_signals,
        op_kwargs={"source_table": "public.nifty_midcap_features_daily", "dest_table": "public.nifty_midcap_investment_signals"}
    )

    # --- NIFTY SMALL CAP PIPELINE ---
    ingest_smallcap = PythonOperator(
        task_id="ingest_yfinance_nifty_smallcap",
        python_callable=ingest_nifty_smallcap,
    )
    etl_smallcap = PythonOperator(
        task_id="spark_etl_nifty_smallcap",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "public.nifty_smallcap_price_raw", "dest_table": "public.nifty_smallcap_features_daily"}
    )
    train_smallcap = PythonOperator(
        task_id="train_model_nifty_smallcap",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "public.nifty_smallcap_features_daily"}
    )
    signal_smallcap = PythonOperator(
        task_id="generate_signals_nifty_smallcap",
        python_callable=generate_signals,
        op_kwargs={"source_table": "public.nifty_smallcap_features_daily", "dest_table": "public.nifty_smallcap_investment_signals"}
    )

    # --- MUTUAL FUNDS PIPELINE ---
    ingest_mf = PythonOperator(
        task_id="ingest_yfinance_mutual_funds",
        python_callable=ingest_mutual_funds,
    )
    etl_mf = PythonOperator(
        task_id="spark_etl_mutual_funds",
        python_callable=run_spark_etl,
        op_kwargs={"source_table": "public.mutual_funds_price_raw", "dest_table": "public.mutual_funds_features_daily"}
    )
    train_mf = PythonOperator(
        task_id="train_model_mutual_funds",
        python_callable=train_all_horizons,
        op_kwargs={"source_table": "public.mutual_funds_features_daily"}
    )
    signal_mf = PythonOperator(
        task_id="generate_signals_mutual_funds",
        python_callable=generate_signals,
        op_kwargs={"source_table": "public.mutual_funds_features_daily", "dest_table": "public.mutual_funds_investment_signals"}
    )

    # Orchestration: ingest >> etl >> train >> signals (all 5 pipelines in parallel)
    ingest_crypto   >> etl_crypto   >> train_crypto   >> signal_crypto
    ingest_stocks   >> etl_stocks   >> train_stocks   >> signal_stocks
    ingest_midcap   >> etl_midcap   >> train_midcap   >> signal_midcap
    ingest_smallcap >> etl_smallcap >> train_smallcap >> signal_smallcap
    ingest_mf       >> etl_mf       >> train_mf       >> signal_mf
