from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pendulum

from src.ingestion.binance_ingest import ingest_binance_data
from src.etl.spark_etl import run_spark_etl
from src.recommendation.strategy_engine import generate_signals


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="crypto_daily_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["crypto", "etl"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_binance",
        python_callable=ingest_binance_data,
    )

    etl = PythonOperator(
        task_id="spark_etl",
        python_callable=run_spark_etl,
    )

    signal = PythonOperator(
        task_id="generate_signals",
        python_callable=generate_signals,
    )

    ingest >> etl >> signal
