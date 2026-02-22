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

# List of top Nifty 50 symbols
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

MUTUAL_FUNDS_SYMBOLS = [
    # Top Flexi-Cap & Multi-Cap Funds
    "0P0001AMBE.BO", # Parag Parikh Flexi Cap Fund
    "0P0000XVWG.BO", # Kotak Flexicap Fund
    "0P0000XVIK.BO", # UTI Flexi Cap Fund
    "0P00005W1U.BO", # Quant Active Fund
    
    # Top Small Cap Funds
    "0P0000XW8F.BO", # SBI Small Cap Fund
    "0P0000XVUA.BO", # Nippon India Small Cap Fund
    "0P0000Y1G4.BO", # Quant Small Cap Fund
    "0P0000XVYW.BO", # HDFC Small Cap Fund
    
    # Top Mid Cap Funds
    "0P0000XVYQ.BO", # HDFC Mid-Cap Opportunities
    "0P000122R2.BO", # Motilal Oswal Midcap Fund
    "0P0000XVMI.BO", # DSP Midcap Fund
    "0P0000XVQD.BO", # Franklin India Prima Fund
    
    # Top Large Cap / Bluechip Funds
    "0P0000XW01.BO", # ICICI Prudential Bluechip
    "0P0000XW8D.BO", # SBI Bluechip Fund
    "0P0000XVRR.BO", # Mirae Asset Large Cap Fund
    "0P0000XW5F.BO", # Axis Bluechip Fund
    "0P0000YX2Z.BO", # Canara Robeco Bluechip Equity
    
    # Value / Thematic Funds
    "0P0000XW14.BO", # ICICI Prudential Value Discovery
    "0P0000XVV8.BO", # Invesco India Contra Fund
    "0P0001AASZ.BO", # Tata Digital India Fund
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
    ensure_table_exists(table_name)
    all_records = []
    
    # 1. Fetch the latest date we have in our Postgres table for incremental loading
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
                # Data mart incremental logic: Only pull from the max date we already have
                fetch_start = last_date_in_db.strftime('%Y-%m-%d')
                logger.info(f"Table contains data for {symbol}. Incremental fetch from {fetch_start}...")
            else:
                logger.info(f"Empty table for {symbol}. Full historical backfill from {fetch_start}...")
            
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
                
            # To reduce load on the API and avoid IP bans, pause between ticker fetches
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

def ingest_mutual_funds():
    # Write historical NAVs to the raw Postgres table. Backfill from 2016 as requested by the user.
    ingest_index(MUTUAL_FUNDS_SYMBOLS, "mutual_funds_price_raw", start_date="2016-01-01")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "midcap":
        ingest_nifty_midcap()
    elif len(sys.argv) > 1 and sys.argv[1] == "smallcap":
        ingest_nifty_smallcap()
    elif len(sys.argv) > 1 and sys.argv[1] == "mutualfunds":
        ingest_mutual_funds()
    else:
        ingest_nifty50()
