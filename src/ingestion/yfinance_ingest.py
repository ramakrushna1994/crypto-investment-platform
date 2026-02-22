"""
yfinance_ingest.py

Fetches historical daily OHLCV data for Indian Equity indices 
(Nifty 50, Midcap, Smallcap) and specific Mutual Funds via the yfinance library.
"""
import pandas as pd
import yfinance as yf
import psycopg2
import logging
import time
from psycopg2.extras import execute_values
from datetime import datetime
from src.config.settings import POSTGRES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core indices to track
NIFTY_50_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TRENT.NS"
]

NIFTY_MIDCAP_SYMBOLS = [
    "FEDERALBNK.NS", "VOLTAS.NS", "PAGEIND.NS", "OFSS.NS", "AUBANK.NS",
    "CGPOWER.NS", "CUMMINSIND.NS", "DIXON.NS", "ESCORTS.NS", "IDFCFIRSTB.NS",
    "LUPIN.NS", "MRF.NS", "OBEROIRLTY.NS", "PATANJALI.NS", "POLYCAB.NS",
    "TATACOMM.NS", "TVSMOTOR.NS", "UBL.NS", "GODREJPROP.NS", "INDHOTEL.NS"
]

NIFTY_SMALLCAP_SYMBOLS = [
    "SUZLON.NS", "CDSL.NS", "HFCL.NS", "ANGELONE.NS", "BSE.NS",
    "CAMPUS.NS", "CHALET.NS", "CYIENT.NS", "RADICO.NS", "SONACOMS.NS",
    "APARINDS.NS", "CEATLTD.NS", "GLENMARK.NS", "IRB.NS", "JYOTHYLAB.NS",
    "MULTIBASE.NS", "PVRINOX.NS", "RITES.NS", "TATAINVEST.NS", "UTIAMC.NS"
]

def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES.host,
        port=POSTGRES.port,
        user=POSTGRES.user,
        password=POSTGRES.password,
        database=POSTGRES.db
    )

def ensure_table_exists(table_name):
    query = f"""
    CREATE TABLE IF NOT EXISTS public.{table_name} (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(50) NOT NULL,
        asset_name VARCHAR(255),
        event_time TIMESTAMP NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume NUMERIC,
        UNIQUE (symbol, event_time)
    );
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to create table {table_name}: {e}")
        raise

def ingest_index(symbols, table_name, start_date=None):
    """
    Fetches historical data for a list of symbols and upserts them into PostgreSQL.
    Supports incremental fetching by querying the latest date in the target table.
    """
    ensure_table_exists(table_name)
    all_records = []
    
    # Pre-fetch the latest date for incremental loading
    latest_dates = {}
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT symbol, MAX(event_time) FROM public.{table_name} GROUP BY symbol;")
                for row in cur.fetchall():
                    latest_dates[row[0]] = row[1]
    except Exception as e:
        logger.warning(f"Could not pre-fetch max dates. Defaulting to full backfill: {e}")
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            # Fetch human-readable name, fallback to symbol if missing
            asset_name = ticker.info.get("longName", symbol)
            
            fetch_start = start_date
            last_date_in_db = latest_dates.get(symbol)
            
            if last_date_in_db:
                fetch_start = last_date_in_db.strftime('%Y-%m-%d')
                logger.info(f"Existing data found for {symbol}. Incremental fetch from {fetch_start}...")
            else:
                logger.info(f"No existing data for {symbol}. Full historical backfill from {fetch_start}...")
            
            if fetch_start:
                hist = ticker.history(start=fetch_start)
            else:
                hist = ticker.history(period="1mo")
            
            for index, row in hist.iterrows():
                all_records.append((
                    symbol,
                    asset_name,
                    index.to_pydatetime(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume'])
                ))
                
            # Rate limit mitigation for Yahoo Finance
            time.sleep(2)
        except Exception as e:
            logger.error(f"Failed fetching {symbol}: {e}")

    if not all_records:
        logger.warning(f"No records fetched for {table_name}.")
        return

    insert_query = f"""
        INSERT INTO public.{table_name} (symbol, asset_name, event_time, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (symbol, event_time) DO UPDATE SET asset_name = EXCLUDED.asset_name;
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, insert_query, all_records)
                conn.commit()
        logger.info(f"✅ Successfully inserted {len(all_records)} records into {table_name}.")
    except Exception as e:
        logger.error(f"Database insertion failed for {table_name}: {e}")
        raise

def ingest_nifty50():
    ingest_index(NIFTY_50_SYMBOLS, "nifty50_price_raw", start_date="2016-01-01")

def ingest_nifty_midcap():
    ingest_index(NIFTY_MIDCAP_SYMBOLS, "nifty_midcap_price_raw", start_date="2016-01-01")

def ingest_nifty_smallcap():
    ingest_index(NIFTY_SMALLCAP_SYMBOLS, "nifty_smallcap_price_raw", start_date="2016-01-01")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "midcap":
        ingest_nifty_midcap()
    elif len(sys.argv) > 1 and sys.argv[1] == "smallcap":
        ingest_nifty_smallcap()
    else:
        ingest_nifty50()
