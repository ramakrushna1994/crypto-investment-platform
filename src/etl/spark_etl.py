from src.storage.postgres import write_spark_df
from src.config.settings import POSTGRES
import time
import logging
logger = logging.getLogger(__name__)

def run_spark_etl():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import avg, stddev
        from pyspark.sql.window import Window
        logger.info("🚀 Starting Spark ETL job")
        spark = SparkSession.builder \
            .appName("CryptoETL") \
            .master("local[*]") \
            .config("spark.jars","/opt/airflow/jars/postgresql-42.7.3.jar")\
            .config("spark.ui.enabled", "true")\
            .config("spark.eventLog.enabled", "true")\
            .config("spark.eventLog.dir", "/opt/airflow/logs/spark") \
            .getOrCreate()
        
        logger.info("Spark session created")

        df = spark.read \
            .format("jdbc") \
            .option("url", POSTGRES.jdbc_url) \
            .option("dbtable", "public.crypto_price_raw") \
            .option("user", POSTGRES.user) \
            .option("password", POSTGRES.password) \
            .option("driver", "org.postgresql.Driver")\
            .load()
        
        record_count = df.count()
        logger.info("Read data from Postgres", extra={"rows": record_count})

        time.sleep(200)

        window_7d = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-7, 0)

        features_df = df.withColumn("moving_avg_7d", avg("close").over(window_7d))\
            .withColumn("volatility_7d", stddev("close").over(window_7d))\
            .dropna()

        logger.info("✅ Spark ETL completed successfully",extra={"output_rows": features_df.count()},)

        write_spark_df(features_df, "public.crypto_features_daily")

    except Exception:
        logger.exception("❌ Spark ETL job failed")
        raise

    finally:
        if 'spark' in locals() and spark is not None:
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    run_spark_etl()
