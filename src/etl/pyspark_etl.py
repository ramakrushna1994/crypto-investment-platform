"""
pyspark_etl.py

Distributed Data Engineering ETL pipeline using Apache Spark.
Computes 15+ technical indicators on high-volume financial time series data.
"""
import os
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

from src.config.settings import POSTGRES, validate_table_name

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _prepare_spark_eventlog_dir():
    """
    Ensure Spark event log directory exists before SparkContext starts.
    Falls back to /tmp if mounted logs path is unavailable.
    """
    preferred = os.getenv("SPARK_EVENTLOG_DIR", "/opt/airflow/logs/spark-events")
    for path in [preferred, "/tmp/spark-events"]:
        try:
            os.makedirs(path, exist_ok=True)
            return True, f"file://{path}"
        except OSError as e:
            logger.warning(f"Unable to prepare Spark event log dir at {path}: {e}")
    return False, ""


def add_ema_column(df, value_col: str, span: int, out_col: str):
    """
    Add an EMA column using the recursive EMA definition (adjust=False):
        EMA_t = alpha * X_t + (1-alpha) * EMA_{t-1}, EMA_0 = X_0

    Closed-form implemented with window ops (per symbol, ordered by event_time)
    to avoid Python UDF overhead and keep execution distributed.
    """
    alpha = 2.0 / (span + 1.0)
    beta = 1.0 - alpha

    w_ordered = Window.partitionBy("symbol").orderBy("event_time")
    w_cum = w_ordered.rowsBetween(Window.unboundedPreceding, 0)

    rn_col = f"__ema_rn_{out_col}"
    first_col = f"__ema_first_{out_col}"
    term_col = f"__ema_term_{out_col}"
    cum_col = f"__ema_cum_{out_col}"

    return (
        df.withColumn(rn_col, F.row_number().over(w_ordered) - 1)
        .withColumn(first_col, F.first(F.col(value_col), ignorenulls=True).over(w_cum))
        .withColumn(
            term_col,
            F.when(
                F.col(rn_col) > 0,
                F.col(value_col) * F.pow(F.lit(beta), -F.col(rn_col))
            ).otherwise(F.lit(0.0))
        )
        .withColumn(cum_col, F.sum(F.col(term_col)).over(w_cum))
        .withColumn(
            out_col,
            F.pow(F.lit(beta), F.col(rn_col)) *
            (F.col(first_col) + F.lit(alpha) * F.col(cum_col))
        )
        .drop(rn_col, first_col, term_col, cum_col)
    )


def create_spark_session(app_name="CryptoETL"):
    """Initialize a local Spark Session optimized for Docker environments."""
    eventlog_enabled, eventlog_uri = _prepare_spark_eventlog_dir()
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "3g")
        .config("spark.network.timeout", "900s")
        .config("spark.executor.heartbeatInterval", "180s")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.jars", "/opt/airflow/jars/postgresql-42.7.3.jar")
        # Reduce memory overhead for large datasets
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
    )

    if eventlog_enabled:
        builder = builder.config("spark.eventLog.enabled", "true").config("spark.eventLog.dir", eventlog_uri)
    else:
        logger.warning("Spark event logging disabled: no writable event log directory found.")
        builder = builder.config("spark.eventLog.enabled", "false")

    return builder.getOrCreate()


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
    
    # Rolling bounds
    w_7  = w_symbol_time.rowsBetween(-6, 0)
    w_14 = w_symbol_time.rowsBetween(-13, 0)
    w_20 = w_symbol_time.rowsBetween(-19, 0)
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
           .withColumn("bb_upper", sma_20 + (F.lit(2) * std_20)) \
            .withColumn("bb_lower", sma_20 - (F.lit(2) * std_20))

    # Use EMA definitions for trend/momentum features.
    df = add_ema_column(df, "close", 12, "__ema_12")
    df = add_ema_column(df, "close", 20, "ema_20")
    df = add_ema_column(df, "close", 26, "__ema_26")
    df = df.withColumn("macd", F.col("__ema_12") - F.col("__ema_26"))
    df = add_ema_column(df, "macd", 9, "macd_signal").drop("__ema_12", "__ema_26")
    
    # 3. Compute RSI 14 - delta column for reuse
    delta = F.col("close") - F.lag("close", 1).over(w_symbol_time)
    gain = F.when(delta > 0, delta).otherwise(0)
    loss = F.when(delta < 0, -delta).otherwise(0)
    avg_gain = F.mean(gain).over(w_14)
    avg_loss = F.mean(loss).over(w_14)
    rs = avg_gain / avg_loss
    df = df.withColumn(
        "rsi_14",
        F.when((avg_gain == 0) & (avg_loss == 0), F.lit(50.0))
        .when(avg_loss == 0, F.lit(100.0))
        .otherwise(100 - (100 / (1 + rs)))
    )

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
    stoch_range = highest_high - lowest_low
    stoch_k = F.when(
        stoch_range != 0,
        ((F.col("close") - lowest_low) / stoch_range) * 100
    ).otherwise(F.lit(50.0))
    df = df.withColumn("stoch_k", stoch_k) \
           .withColumn("stoch_d", F.mean(stoch_k).over(w_3))

    # Remove cache after compute to free memory
    df = df.unpersist()
    return df

# NAV-specific features that supplement standard indicators for mutual funds.
NAV_FEATURE_COLUMNS = (
    "rolling_return_30d",
    "rolling_return_90d",
    "sortino_30d",
    "max_drawdown_30d",
    "nav_momentum_14d",
)


def compute_nav_features(df):
    """
    Compute NAV-appropriate features for mutual fund data.

    MF data has open=high=low=close=NAV with volume=0.  While the standard
    indicators still work (ATR captures |daily change|, Stochastic uses
    cross-day NAV range), these NAV-specific features capture fund dynamics
    more effectively: rolling returns, downside risk, drawdown, and momentum.
    """
    w_sym = Window.partitionBy("symbol").orderBy("event_time")
    w_30 = w_sym.rowsBetween(-29, 0)
    w_14 = w_sym.rowsBetween(-13, 0)

    # ── Rolling Returns ──────────────────────────────────────────────────
    prev_30 = F.lag("close", 30).over(w_sym)
    prev_90 = F.lag("close", 90).over(w_sym)
    df = df.withColumn(
        "rolling_return_30d",
        F.when(prev_30 > 0, (F.col("close") / prev_30) - 1).otherwise(F.lit(None)),
    ).withColumn(
        "rolling_return_90d",
        F.when(prev_90 > 0, (F.col("close") / prev_90) - 1).otherwise(F.lit(None)),
    )

    # ── Daily Return (intermediate) ──────────────────────────────────────
    prev_close = F.lag("close", 1).over(w_sym)
    daily_ret_col = "__daily_ret"
    df = df.withColumn(
        daily_ret_col,
        F.when(prev_close > 0, (F.col("close") / prev_close) - 1).otherwise(F.lit(0.0)),
    )

    # ── Sortino-like Ratio (30d) ─────────────────────────────────────────
    # Sortino = mean(return) / stddev(negative returns only)
    downside_col = "__downside_ret"
    df = df.withColumn(
        downside_col,
        F.when(F.col(daily_ret_col) < 0, F.col(daily_ret_col)).otherwise(F.lit(0.0)),
    )
    mean_ret_30 = F.mean(daily_ret_col).over(w_30)
    downside_std_30 = F.stddev(downside_col).over(w_30)
    df = df.withColumn(
        "sortino_30d",
        F.when(
            (downside_std_30.isNotNull()) & (downside_std_30 > 1e-8),
            mean_ret_30 / downside_std_30,
        ).otherwise(F.lit(0.0)),
    )

    # ── Max Drawdown (30d rolling) ───────────────────────────────────────
    # drawdown_t = close_t / running_max - 1  (always <= 0)
    running_max_30 = F.max("close").over(w_30)
    df = df.withColumn(
        "max_drawdown_30d",
        F.when(
            running_max_30 > 0,
            (F.col("close") / running_max_30) - 1,
        ).otherwise(F.lit(0.0)),
    )

    # ── NAV Momentum (14d) ───────────────────────────────────────────────
    # How far current NAV is above/below its 14-day SMA (relative strength)
    sma_14 = F.mean("close").over(w_14)
    df = df.withColumn(
        "nav_momentum_14d",
        F.when(sma_14 > 0, (F.col("close") / sma_14) - 1).otherwise(F.lit(0.0)),
    )

    # Clean up intermediate columns
    df = df.drop(daily_ret_col, downside_col)
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

        # Step 2b: Add NAV-specific features for mutual fund tables
        is_mutual_fund = "mutual_fund" in source_table.lower()
        if is_mutual_fund:
            logger.info("Detected mutual fund source — computing NAV-specific features...")
            features_df = compute_nav_features(features_df)

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
    src = sys.argv[1] if len(sys.argv) > 1 else "bronze.crypto_price_raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "silver.crypto_features_daily"
    run_spark_etl(src, dst)
