"""
mfapi_ingest.py

Dynamically discovers and ingests All Direct Growth mutual funds from the 20 
largest AMCs in India using the public MFAPI.in REST API.
"""
import requests
import psycopg2
import logging
import time
from psycopg2.extras import execute_values
from datetime import datetime
from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Top AMCs in India by AUM (as of 2025). Used to filter the MFAPI catalogue.
TOP_AMC_KEYWORDS = [
    "SBI",
    "ICICI Prudential",
    "HDFC",
    "Nippon India",
    "Kotak",
    "Axis",
    "Mirae Asset",
    "UTI",
    "DSP",
    "Tata",
    "Franklin India",
    "Canara Robeco",
    "PGIM India",
    "Sundaram",
    "Motilal Oswal",
    "Bandhan",           # formerly IDFC
    "Edelweiss",
    "Aditya Birla Sun Life",
    "Quant",
    "WhiteOak Capital",
    "Parag Parikh",
    "Invesco India",
    "Navi",
]

# Only ingest "Direct" + "Growth" plans — avoids duplicating Regular plans & IDCW
REQUIRED_KEYWORDS   = ["Direct", "Growth"]
EXCLUDED_KEYWORDS   = ["IDCW", "Dividend", "Weekly", "Monthly", "Bonus", "Segregated"]

def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES.host, port=POSTGRES.port,
        user=POSTGRES.user, password=POSTGRES.password,
        database=POSTGRES.db
    )

def discover_funds() -> list[dict]:
    """
    Fetch the full MFAPI catalogue and return all Direct Growth schemes
    from our top-AMC list. Returns list of {schemeCode, schemeName}.
    """
    logger.info("Fetching full MFAPI scheme catalogue...")
    resp = requests.get("https://api.mfapi.in/mf", timeout=30)
    resp.raise_for_status()
    all_schemes = resp.json()
    logger.info(f"  Total schemes in MFAPI: {len(all_schemes):,}")

    selected = []
    for scheme in all_schemes:
        name = scheme.get("schemeName", "")

        # Must be from a top AMC
        if not any(kw.lower() in name.lower() for kw in TOP_AMC_KEYWORDS):
            continue
        # Must be Direct + Growth
        if not all(kw in name for kw in REQUIRED_KEYWORDS):
            continue
        # Exclude IDCW/dividend/etc.
        if any(kw.lower() in name.lower() for kw in EXCLUDED_KEYWORDS):
            continue

        selected.append(scheme)

    logger.info(f"  Matched {len(selected)} Direct Growth funds from top AMCs")
    return selected

def fetch_nav_history(scheme_code: int, fund_name: str) -> list[dict]:
    """Fetch full NAV history for a single fund."""
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        records = []
        symbol = f"MF{scheme_code}"
        for entry in resp.json().get("data", []):
            try:
                nav_date = datetime.strptime(entry["date"], "%d-%m-%Y").date()
                nav_val  = float(entry["nav"])
                records.append({
                    "symbol":     symbol,
                    "asset_name": fund_name,
                    "event_time": nav_date,
                    "close": nav_val, "open": nav_val,
                    "high":  nav_val, "low":  nav_val,
                    "volume": 0,
                })
            except (ValueError, KeyError):
                continue
        return records
    except Exception as e:
        logger.warning(f"Failed to fetch {fund_name} ({scheme_code}): {e}")
        return []

def ingest_mutual_funds(max_workers: int = 20):
    """Airflow-callable entry point: discover + ingest all top-AMC funds.

    Args:
        max_workers: Number of concurrent HTTP threads for fetching NAV data.
                     Default 20 is safe for MFAPI.in without getting throttled.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    funds = discover_funds()

    conn = get_db_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS public.mutual_funds_price_raw (
            symbol     TEXT,
            asset_name TEXT,
            event_time DATE,
            open       NUMERIC,
            high       NUMERIC,
            low        NUMERIC,
            close      NUMERIC,
            volume     BIGINT,
            PRIMARY KEY (symbol, event_time)
        )
    """)
    conn.commit()

    # Latest date per symbol for incremental loading
    cur.execute("""
        SELECT symbol, MAX(event_time) FROM public.mutual_funds_price_raw GROUP BY symbol
    """)
    latest_dates = {row[0]: row[1] for row in cur.fetchall()}

    # Fetch NAV data in parallel
    logger.info(f"Fetching NAV data with {max_workers} parallel workers...")

    def fetch_fund(scheme):
        code  = scheme["schemeCode"]
        name  = scheme["schemeName"]
        sym   = f"MF{code}"
        last  = latest_dates.get(sym)
        records = fetch_nav_history(code, name)
        if records and last:
            last_date = last.date() if hasattr(last, 'date') else last
            records = [r for r in records if r["event_time"] > last_date]
        return records

    all_results = []  # list of (records_list,)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_fund, s): s for s in funds}
        for i, future in enumerate(as_completed(futures), 1):
            records = future.result()
            if records:
                all_results.append(records)
            if i % 100 == 0:
                logger.info(f"  Fetched {i}/{len(funds)} funds...")

    logger.info(f"  All fetches done. {len(all_results)} funds had new data.")

    # Sequential, batched database writes
    total_new = 0
    for records in all_results:
        rows = [(
            r["symbol"], r["asset_name"], r["event_time"],
            r["open"], r["high"], r["low"], r["close"], r["volume"]
        ) for r in records]
        execute_values(cur, """
            INSERT INTO public.mutual_funds_price_raw
                (symbol, asset_name, event_time, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, event_time) DO UPDATE SET
                close = EXCLUDED.close, asset_name = EXCLUDED.asset_name
        """, rows)
        total_new += len(rows)

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Completed ingestion. Inserted {total_new:,} new NAV records from {len(all_results)} funds.")



if __name__ == "__main__":
    ingest_mutual_funds()
