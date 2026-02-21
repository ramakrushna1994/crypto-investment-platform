from src.storage.postgres import write_spark_df
from src.config.settings import POSTGRES
import time
import logging
logger = logging.getLogger(__name__)

def run_spark_etl():
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import avg, stddev, lag, when, col
        from pyspark.sql.window import Window
        logger.info("🚀 Starting Spark ETL job")
        spark = SparkSession.builder \
            .appName("CryptoETL") \
            .master("local[*]") \
            .config("spark.jars","/opt/airflow/jars/postgresql-42.7.3.jar")\
            .config("spark.ui.enabled", "true")\
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

        # 1. RSI (14-period)
        window_1d = Window.partitionBy("symbol").orderBy("event_time")
        df_change = df.withColumn("prev_close", lag("close", 1).over(window_1d)) \
            .withColumn("change", col("close") - col("prev_close"))
            
        df_change = df_change.withColumn("gain", when(col("change") > 0, col("change")).otherwise(0)) \
            .withColumn("loss", when(col("change") < 0, -col("change")).otherwise(0))
            
        window_14d = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-14, 0)
        df_rsi = df_change.withColumn("avg_gain", avg("gain").over(window_14d)) \
            .withColumn("avg_loss", avg("loss").over(window_14d))
            
        df_rsi = df_rsi.withColumn("rs", col("avg_gain") / when(col("avg_loss") == 0, 1e-9).otherwise(col("avg_loss"))) \
            .withColumn("rsi_14", 100 - (100 / (1 + col("rs"))))

        # 2. MACD (Approximation with SMA for prototype)
        window_12d = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-12, 0)
        window_26d = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-26, 0)
        
        df_macd = df_rsi.withColumn("sma_12", avg("close").over(window_12d)) \
            .withColumn("sma_26", avg("close").over(window_26d)) \
            .withColumn("macd", col("sma_12") - col("sma_26"))
            
        window_9d_macd = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-9, 0)
        features_df = df_macd.withColumn("macd_signal", avg("macd").over(window_9d_macd))

        # 3. Base features
        window_7d = Window.partitionBy("symbol").orderBy("event_time").rowsBetween(-7, 0)
        features_df = features_df.withColumn("moving_avg_7d", avg("close").over(window_7d))\
            .withColumn("volatility_7d", stddev("close").over(window_7d))\
            .drop("prev_close", "change", "gain", "loss", "avg_gain", "avg_loss", "rs", "sma_12", "sma_26") \
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
