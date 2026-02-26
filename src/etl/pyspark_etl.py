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
            .config("spark.executor.memory", "3g")
            .config("spark.network.timeout", "900s")
            .config("spark.executor.heartbeatInterval", "180s")
            .config("spark.sql.shuffle.partitions", "20")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.jars", "/opt/airflow/jars/postgresql-42.7.3.jar")
            .config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", "file:///opt/airflow/logs/spark-events")
            # Reduce memory overhead for large datasets
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
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
    Optimized to minimize memory overhead and shuffling.
    """
    # Cache the input after basic type casting - avoid repeated reads
    df = df.cache()
    
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

    # 2. Compute all indicators in a single DataFrame transformation pass
    # to minimize memory bloat from intermediate DataFrames
    
    sma_50 = F.mean("close").over(w_50)
    sma_200 = F.mean("close").over(w_200)
    sma_20 = F.mean("close").over(w_20)
    std_20 = F.stddev("close").over(w_20)
    
    df = df.withColumn("sma_50", sma_50) \
           .withColumn("sma_200", sma_200) \
           .withColumn("moving_avg_7d", F.mean("close").over(w_7)) \
           .withColumn("volatility_7d", F.stddev("close").over(w_7)) \
           .withColumn("ema_20", sma_20) \
           .withColumn("macd", F.mean("close").over(w_12) - F.mean("close").over(w_26)) \
           .withColumn("bb_upper", sma_20 + (F.lit(2) * std_20)) \
           .withColumn("bb_lower", sma_20 - (F.lit(2) * std_20))
    
    # Compute MACD signal in single pass
    df = df.withColumn("macd_signal", F.mean("macd").over(w_9))
    
    # 3. Compute RSI 14 - delta column for reuse
    delta = F.col("close") - F.lag("close", 1).over(w_symbol_time)
    gain = F.when(delta > 0, delta).otherwise(0)
    loss = F.when(delta < 0, -delta).otherwise(0)
    avg_gain = F.mean(gain).over(w_14)
    avg_loss = F.mean(loss).over(w_14)
    rs = avg_gain / F.when(avg_loss == 0, F.lit(1)).otherwise(avg_loss)
    df = df.withColumn("rsi_14", 100 - (100 / (1 + rs)))

    # 4. Compute True Range (TR) for ATR
    prev_close = F.lag("close", 1).over(w_symbol_time)
    tr1 = F.col("high") - F.col("low")
    tr2 = F.abs(F.col("high") - prev_close)
    tr3 = F.abs(F.col("low") - prev_close)
    
    true_range = F.greatest(tr1, tr2, tr3)
    df = df.withColumn("atr_14", F.mean(true_range).over(w_14))

    # 5. Compute Stochastic Oscillator
    lowest_low = F.min("low").over(w_14)
    highest_high = F.max("high").over(w_14)
    
    w_3 = w_symbol_time.rowsBetween(-2, 0)
    stoch_k = ((F.col("close") - lowest_low) / (highest_high - lowest_low)) * 100
    df = df.withColumn("stoch_k", stoch_k) \
           .withColumn("stoch_d", F.mean(stoch_k).over(w_3))

    # Remove cache after compute to free memory
    df = df.unpersist()
    return df


def run_spark_etl(source_table: str, dest_table: str):
    """
    End-to-end PySpark ETL: Read from Postgres -> Transform -> Write to Postgres
    Optimized for large datasets (millions of rows) with memory constraints.
    """
    logger.info(f"Starting PySpark ETL pipeline: {source_table} -> {dest_table}")
    
    spark = create_spark_session()
    jdbc_url = POSTGRES.jdbc_url
    props = get_postgres_properties()

    try:
        # Step 1: Read Raw Data from PostgreSQL via JDBC
        logger.info(f"Reading data from {source_table}")
        
        # Performance adjustment: Determine partitioning bounds for the JDBC read
        import psycopg2
        conn = psycopg2.connect(
            user=POSTGRES.user,
            password=POSTGRES.password,
            host=POSTGRES.host,
            database=POSTGRES.db
        )
        
        min_id, max_id, row_count = None, None, 0
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT MIN(id), MAX(id), COUNT(*) FROM {source_table}")
                min_id, max_id, row_count = cur.fetchone()
        except Exception as e:
            logger.warning(f"Could not fetch metadata for {source_table}: {e}. Falling back to single-partition read.")
            conn.rollback() 
        finally:
            if conn:
                conn.close()

        # Dynamically determine optimal partition count based on data size
        # For large datasets (>1M rows), use fewer partitions to avoid OOM
        optimal_partitions = 3
        if row_count > 0:
            logger.info(f"Table {source_table} has {row_count:,} rows")
            # Use formula: partitions = min(sqrt(rows/100000), 8)
            # For 2.8M rows: sqrt(28) ≈ 5, capped at 8
            import math
            optimal_partitions = min(max(1, int(math.sqrt(row_count / 100000))), 8)
            logger.info(f"Using {optimal_partitions} partitions for JDBC read")

        # If we have valid IDs, read in parallel. Otherwise fall back to single partition.
        if min_id is not None and max_id is not None:
            logger.info(f"Parallel read enabled: partitioning {source_table} via 'id' [{min_id} to {max_id}]")
            raw_df = spark.read.jdbc(
                url=jdbc_url,
                table=source_table,
                column="id",
                lowerBound=min_id,
                upperBound=max_id,
                numPartitions=optimal_partitions,
                properties=props
            )
        else:
            logger.info(f"Single-partition read for {source_table}")
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

        # CRITICAL OPTIMIZATION: Repartition by symbol early.
        # This ensures all technical indicator windows (which partitionBy symbol) 
        # happen within the same executor partition, avoiding massive shuffles.
        # Use optimal partition count based on unique symbols
        num_unique_symbols = raw_df.select("symbol").distinct().count()
        num_repartitions = max(2, min(num_unique_symbols, 10))
        logger.info(f"Repartitioning {num_unique_symbols} unique symbols into {num_repartitions} partitions...")
        raw_df = raw_df.repartition(num_repartitions, "symbol")

        # Step 2: Distributed Feature Engineering
        logger.info("Computing technical indicators across cluster...")
        features_df = compute_indicators(raw_df)

        # Drop rows where long-term indicators are null (warmup period: SMA 200/50 etc)
        # This reduces output size and only keeps valid/ready data
        features_df = features_df.dropna(subset=["sma_50"])
        
        row_count_features = features_df.count()
        logger.info(f"Generated {row_count_features:,} feature rows after filtering")

        # Step 3: Write Engineered Features back to PostgreSQL
        logger.info(f"Writing features to {dest_table}. Mode: overwrite")
        
        # Use coalesce to reduce output partitions and minimize file overhead
        # For PostgreSQL, too many tiny partitions cause overhead
        features_df.coalesce(2).write.jdbc(
            url=jdbc_url,
            table=dest_table,
            mode="overwrite",
            properties=props
        )
        
        logger.info(f"Successfully processed and wrote {row_count_features:,} rows to {dest_table}")

    except Exception as e:
        logger.exception(f"PySpark ETL Failed: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "public.crypto_price_raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "public.crypto_features_daily"
    run_spark_etl(src, dst)
