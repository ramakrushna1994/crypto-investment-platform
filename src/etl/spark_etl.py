"""
Incremental Feature Engineering ETL
=====================================
Processes only NEW raw price rows per symbol rather than reloading all history.

Strategy:
  1. Find MAX(event_time) per symbol that's already in `features_daily`
  2. For each symbol, load raw rows from (latest_feature_date - LOOKBACK_DAYS)
     — needed to warm up rolling windows like SMA-200
  3. Compute all 13 technical indicators
  4. UPSERT only rows where event_time > latest_feature_date

On the 1st-ever run (no existing features), processes full history.
On daily Airflow runs, reads ~201 rows per symbol and writes ~1 row — very fast.

Pure Pandas — no PySpark needed at this data volume (< 50M rows).
"""
import sys
import io
import logging
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values
from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Indicator warmup: SMA-200 needs 200 days; we load this many days BEFORE the
# latest feature date so rolling windows are accurate for new rows.
LOOKBACK_DAYS = 210


def get_db_connection():
    return psycopg2.connect(POSTGRES.dsn)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 13 technical indicators for every row in df, grouped by symbol."""
    results = []
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("event_time").copy()

        g["rsi_14"]      = compute_rsi(g["close"])

        ema12            = g["close"].ewm(span=12, adjust=False).mean()
        ema26            = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"]        = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()

        g["moving_avg_7d"] = g["close"].rolling(7,  min_periods=1).mean()
        g["volatility_7d"] = g["close"].rolling(7,  min_periods=1).std()
        g["ema_20"]        = g["close"].ewm(span=20, adjust=False).mean()

        sma20            = g["close"].rolling(20, min_periods=1).mean()
        std20            = g["close"].rolling(20, min_periods=1).std()
        g["bb_upper"]    = sma20 + 2 * std20
        g["bb_lower"]    = sma20 - 2 * std20

        g["sma_50"]      = g["close"].rolling(50,  min_periods=1).mean()
        g["sma_200"]     = g["close"].rolling(200, min_periods=1).mean()

        hl               = g["high"] - g["low"]
        hc               = (g["high"] - g["close"].shift()).abs()
        lc               = (g["low"]  - g["close"].shift()).abs()
        g["atr_14"]      = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14, min_periods=1).mean()

        lo14             = g["low"].rolling(14,  min_periods=1).min()
        hi14             = g["high"].rolling(14, min_periods=1).max()
        g["stoch_k"]     = 100 * (g["close"] - lo14) / (hi14 - lo14).replace(0, np.nan)
        g["stoch_d"]     = g["stoch_k"].rolling(3, min_periods=1).mean()

        results.append(g)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_spark_etl(source_table: str = "public.crypto_price_raw",
                  dest_table:   str = "public.crypto_features_daily"):
    """Incremental ETL: only process raw rows newer than what's already in features_daily."""
    logger.info(f"Incremental ETL: {source_table} → {dest_table}")

    conn = get_db_connection()
    cur  = conn.cursor()

    # ── 1. Ensure destination table exists ────────────────────────────────────
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {dest_table} (
            symbol TEXT, event_time TIMESTAMP,
            open DOUBLE PRECISION, high DOUBLE PRECISION,
            low DOUBLE PRECISION, close DOUBLE PRECISION,
            volume BIGINT, asset_name TEXT,
            rsi_14 DOUBLE PRECISION, macd DOUBLE PRECISION,
            macd_signal DOUBLE PRECISION, moving_avg_7d DOUBLE PRECISION,
            volatility_7d DOUBLE PRECISION, ema_20 DOUBLE PRECISION,
            bb_upper DOUBLE PRECISION, bb_lower DOUBLE PRECISION,
            sma_50 DOUBLE PRECISION, sma_200 DOUBLE PRECISION,
            atr_14 DOUBLE PRECISION, stoch_k DOUBLE PRECISION,
            stoch_d DOUBLE PRECISION,
            PRIMARY KEY (symbol, event_time)
        )
    """)
    conn.commit()

    # ── 2. Find unique symbols and latest processed date per symbol ────────────
    cur.execute(f"SELECT symbol, MAX(event_time)::DATE FROM {dest_table} GROUP BY symbol")
    latest = {row[0]: row[1] for row in cur.fetchall()}
    
    # Get all symbols from source table to process in batches
    cur.execute(f"SELECT DISTINCT symbol FROM {source_table}")
    all_symbols = [row[0] for row in cur.fetchall()]
    
    is_first_run = len(latest) == 0
    logger.info(f"  Total symbols to check: {len(all_symbols)}")
    logger.info(f"  Known symbols in features: {len(latest)} ({'first run' if is_first_run else 'incremental'})")

    if not all_symbols:
        logger.info("  No source symbols found. Exiting.")
        cur.close(); conn.close()
        return

    # ── 3. Path setup and common variables ─────────────────────────────────────
    BATCH_SIZE_SYMBOLS = 500
    TOTAL_SYMBOLS = len(all_symbols)
    total_upserted = 0

    feature_cols = [
        "symbol", "event_time", "open", "high", "low", "close", "volume",
        "rsi_14", "macd", "macd_signal", "moving_avg_7d", "volatility_7d",
        "ema_20", "bb_upper", "bb_lower", "sma_50", "sma_200",
        "atr_14", "stoch_k", "stoch_d"
    ]
    # We'll determine columns dynamically for each batch
    
    # ── 4. Process symbols in batches to prevent OOM ──────────────────────────
    import datetime
    for i in range(0, TOTAL_SYMBOLS, BATCH_SIZE_SYMBOLS):
        batch_symbols = all_symbols[i : i + BATCH_SIZE_SYMBOLS]
        logger.info(f"  Batch {i//BATCH_SIZE_SYMBOLS + 1}: Processing {len(batch_symbols)} symbols...")

        # Build query for this batch
        if is_first_run:
            query = f"SELECT * FROM {source_table} WHERE symbol IN %s ORDER BY symbol, event_time"
            params = (tuple(batch_symbols),)
        else:
            # We still use a global earliest cutoff for the batch to simplify, or could be per symbol.
            # Per-symbol is more precise but harder SQL. We'll stick to a generous global batch cutoff.
            # Get latest date for symbols in THIS batch
            batch_latest = {s: latest.get(s) for s in batch_symbols if s in latest}
            if not batch_latest:
                # New symbols in this batch
                query = f"SELECT * FROM {source_table} WHERE symbol IN %s ORDER BY symbol, event_time"
                params = (tuple(batch_symbols),)
            else:
                earliest_cutoff = min(batch_latest.values())
                if hasattr(earliest_cutoff, 'date'): earliest_cutoff = earliest_cutoff.date()
                cutoff_with_buffer = earliest_cutoff - datetime.timedelta(days=LOOKBACK_DAYS)
                
                query = f"SELECT * FROM {source_table} WHERE symbol IN %s AND event_time >= %s ORDER BY symbol, event_time"
                params = (tuple(batch_symbols), cutoff_with_buffer)

        # Direct psycopg2 execution to avoid SQLAlchemy formatting issues
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        
        if df.empty:
            continue

        # Data cleaning
        df["event_time"] = pd.to_datetime(df["event_time"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

        # Compute indicators
        features = compute_indicators(df)
        
        # Filter to only NEW rows
        if not is_first_run:
            def is_new(row):
                lat = latest.get(row["symbol"])
                if lat is None: return True
                return pd.Timestamp(row["event_time"]) > pd.Timestamp(lat)
            features = features[features.apply(is_new, axis=1)]

        if features.empty:
            continue

        # Prepare for upsert
        local_cols = [c for c in feature_cols if c in features.columns]
        if "asset_name" in features.columns and "asset_name" not in local_cols:
            local_cols.insert(7, "asset_name")
        
        # Drop rows with NaN RSI (usually early data)
        features = features[local_cols].dropna(subset=["rsi_14"])
        if features.empty:
            continue

        # Upsert in smaller DB batches
        update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in local_cols if c not in ("symbol", "event_time"))
        
        DB_BATCH = 2000
        for start in range(0, len(features), DB_BATCH):
            chunk = features.iloc[start : start + DB_BATCH]
            rows = [tuple(row) for row in chunk.itertuples(index=False, name=None)]
            execute_values(cur, f"""
                INSERT INTO {dest_table} ({", ".join(local_cols)})
                VALUES %s
                ON CONFLICT (symbol, event_time) DO UPDATE SET {update_sql}
            """, rows)
            total_upserted += len(rows)
            
        conn.commit()

    cur.close()
    conn.close()
    logger.info(f"✅ ETL complete — upserted {total_upserted:,} rows into {dest_table}")



if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "public.crypto_price_raw"
    dest   = sys.argv[2] if len(sys.argv) > 2 else "public.crypto_features_daily"
    run_spark_etl(source, dest)
