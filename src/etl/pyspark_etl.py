"""
pyspark_etl.py

Distributed Data Engineering ETL pipeline using Apache Spark.
Computes 15+ technical indicators on high-volume financial time series data.
"""
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_spark_session(app_name="CryptoETL"):
    """Initialize a local Spark Session optimized for Docker environments."""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .config("spark.jars", "/opt/airflow/jars/postgresql-42.7.3.jar")
            .config("spark.sql.shuffle.partitions", "10") # Small partitions for local testing
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", "file:///opt/airflow/logs/spark-events")
            .getOrCreate())


def get_postgres_properties():
    """Return dict of JDBC connection properties for Spark."""
    return {
        "user": POSTGRES.user,
        "password": POSTGRES.password,
        "driver": "org.postgresql.Driver"
    }


def compute_indicators(df):
    """
    Computes technical indicators using distributed Spark Window functions.
    This avoids bringing data to the driver node, unlike Pandas.
    """
    # 1. Define window specifications
    w_symbol_time = Window.partitionBy("symbol").orderBy("event_time")
    
    # Simple Moving Averages (SMA) bounds
    w_7  = w_symbol_time.rowsBetween(-6, 0)
    w_9  = w_symbol_time.rowsBetween(-8, 0)
    w_12 = w_symbol_time.rowsBetween(-11, 0)
    w_14 = w_symbol_time.rowsBetween(-13, 0)
    w_20 = w_symbol_time.rowsBetween(-19, 0)
    w_26 = w_symbol_time.rowsBetween(-25, 0)
    w_50 = w_symbol_time.rowsBetween(-49, 0)
    w_200 = w_symbol_time.rowsBetween(-199, 0)

    # 2. Compute basic rolling indicators
    df = df.withColumn("moving_avg_7d", F.mean("close").over(w_7))
    df = df.withColumn("volatility_7d", F.stddev("close").over(w_7))
    df = df.withColumn("sma_50", F.mean("close").over(w_50))
    df = df.withColumn("sma_200", F.mean("close").over(w_200))
    
    # 3. Approximate EMA & MACD using SMA (Pure Spark SQL approach avoids recursive UDFs)
    df = df.withColumn("ema_20", F.mean("close").over(w_20))
    df = df.withColumn("macd", F.mean("close").over(w_12) - F.mean("close").over(w_26))
    df = df.withColumn("macd_signal", F.mean("macd").over(w_9))
    
    sma_20 = F.mean("close").over(w_20)
    std_20 = F.stddev("close").over(w_20)
    df = df.withColumn("bb_upper", sma_20 + (F.lit(2) * std_20))
    df = df.withColumn("bb_lower", sma_20 - (F.lit(2) * std_20))
    
    # 4. Compute RSI 14
    delta = F.col("close") - F.lag("close", 1).over(w_symbol_time)
    gain = F.when(delta > 0, delta).otherwise(0)
    loss = F.when(delta < 0, -delta).otherwise(0)
    avg_gain = F.mean(gain).over(w_14)
    avg_loss = F.mean(loss).over(w_14)
    rs = avg_gain / F.when(avg_loss == 0, None).otherwise(avg_loss)
    df = df.withColumn("rsi_14", 100 - (100 / (1 + rs)))

    # 5. Compute True Range (TR) for ATR
    prev_close = F.lag("close", 1).over(w_symbol_time)
    tr1 = F.col("high") - F.col("low")
    tr2 = F.abs(F.col("high") - prev_close)
    tr3 = F.abs(F.col("low") - prev_close)
    
    true_range = F.greatest(tr1, tr2, tr3)
    df = df.withColumn("tr", true_range)
    df = df.withColumn("atr_14", F.mean("tr").over(w_14))
    df = df.drop("tr")

    # 6. Compute Stochastic Oscillator
    lowest_low = F.min("low").over(w_14)
    highest_high = F.max("high").over(w_14)
    df = df.withColumn("stoch_k", ((F.col("close") - lowest_low) / (highest_high - lowest_low)) * 100)
    
    w_3 = w_symbol_time.rowsBetween(-2, 0)
    df = df.withColumn("stoch_d", F.mean("stoch_k").over(w_3))

    return df


def run_spark_etl(source_table: str, dest_table: str):
    """
    End-to-end PySpark ETL: Read from Postgres -> Transform -> Write to Postgres
    """
    logger.info(f"Starting PySpark ETL pipeline: {source_table} -> {dest_table}")
    
    spark = create_spark_session()
    jdbc_url = POSTGRES.jdbc_url
    props = get_postgres_properties()

    try:
        # Step 1: Read Raw Data from PostgreSQL via JDBC
        logger.info(f"Reading data from {source_table}")
        raw_df = spark.read.jdbc(
            url=jdbc_url,
            table=source_table,
            properties=props
        )

        if raw_df.rdd.isEmpty():
            logger.warning(f"No data found in {source_table}")
            return

        # Ensure correct types
        raw_df = raw_df.withColumn("close", F.col("close").cast(DoubleType())) \
                       .withColumn("high", F.col("high").cast(DoubleType())) \
                       .withColumn("low", F.col("low").cast(DoubleType()))

        # Step 2: Distributed Feature Engineering
        logger.info("Computing technical indicators across cluster...")
        features_df = compute_indicators(raw_df)

        # Drop rows where long-term indicators are null (need warmup period)
        features_df = features_df.dropna(subset=["sma_50"])

        # Step 3: Write Engineered Features back to PostgreSQL
        logger.info(f"Writing features to {dest_table}. Modes: Overwrite/Append")
        
        # In a real production setup we would use UPSERT logic via temp tables,
        # but for this portfolio demonstration, we'll overwrite the destination table
        # to guarantee a clean state of PySpark-generated features.
        features_df.write.jdbc(
            url=jdbc_url,
            table=dest_table,
            mode="overwrite",
            properties=props
        )
        
        logger.info(f"Successfully processed and wrote data to {dest_table}")

    except Exception as e:
        logger.exception(f"PySpark ETL Failed: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "public.crypto_price_raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "public.crypto_features_daily"
    run_spark_etl(src, dst)
