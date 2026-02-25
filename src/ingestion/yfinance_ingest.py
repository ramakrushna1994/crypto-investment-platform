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

import requests
import io

def get_nse_index_symbols(index_name):
    """
    Dynamically fetches the latest constituents for an NSE index directly from the National Stock Exchange.
    """
    url_map = {
        "nifty50": "ind_nifty50list.csv",
        "nifty_midcap": "ind_niftymidcap150list.csv",
        "nifty_smallcap": "ind_niftysmallcap250list.csv"
    }
    
    if index_name not in url_map:
        raise ValueError(f"Unknown index {index_name}")
        
    url = f"https://archives.nseindia.com/content/indices/{url_map[index_name]}"
    
    try:
        # NSE blocks requests without a User-Agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        # The column is usually 'Symbol', add .NS suffix for Yahoo Finance
        symbols = [f"{sym}.NS" for sym in df['Symbol'].tolist()]
        logger.info(f"Successfully fetched {len(symbols)} dynamic symbols for {index_name} from NSE.")
        return symbols
    except Exception as e:
        logger.error(f"Failed to fetch dynamic symbols for {index_name} from NSE: {e}")
        # Return fallback lists just in case NSE site is down
        fallbacks = {
            "nifty50": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
            "nifty_midcap": ["FEDERALBNK.NS", "VOLTAS.NS", "PAGEIND.NS", "OFSS.NS", "AUBANK.NS"],
            "nifty_smallcap": ["SUZLON.NS", "CDSL.NS", "HFCL.NS", "ANGELONE.NS", "BSE.NS"]
        }
        return fallbacks[index_name]

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
    symbols = get_nse_index_symbols("nifty50")
    ingest_index(symbols, "nifty50_price_raw", start_date="2016-01-01")

def ingest_nifty_midcap():
    symbols = get_nse_index_symbols("nifty_midcap")
    ingest_index(symbols, "nifty_midcap_price_raw", start_date="2016-01-01")

def ingest_nifty_smallcap():
    symbols = get_nse_index_symbols("nifty_smallcap")
    ingest_index(symbols, "nifty_smallcap_price_raw", start_date="2016-01-01")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "midcap":
        ingest_nifty_midcap()
    elif len(sys.argv) > 1 and sys.argv[1] == "smallcap":
        ingest_nifty_smallcap()
    else:
        ingest_nifty50()
