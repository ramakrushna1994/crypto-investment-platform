"""
binance_ingest.py

Fetches historical daily OHLCV klines for configured cryptocurrency pairs
from the public Binance API and stores them in PostgreSQL.
"""
import requests
import psycopg2
from datetime import datetime
from src.config.settings import POSTGRES, BINANCE
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# Allows Airflow to mark the task as "Skipped" instead of "Failed" on API outage
from airflow.exceptions import AirflowSkipException


def ingest_binance_data():
    try:
        logger.info("Starting Binance ingestion task")
        conn = psycopg2.connect(POSTGRES.dsn)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bronze.crypto_price_raw (
                id SERIAL PRIMARY KEY,
                symbol TEXT,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume NUMERIC,
                event_time TIMESTAMP,
                UNIQUE (symbol, event_time)
            )
        """)

        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.HTTPError))
        )
        def fetch_binance_klines(symbol):
            response = requests.get(
                BINANCE.base_url,
                params={
                    "symbol": symbol,
                    "interval": BINANCE.interval,
                    "limit": BINANCE.limit
                },
                timeout=BINANCE.timeout_seconds
            )
            # Raise for 429 Too Many Requests, 500, 502, 503, 504
            if response.status_code in (429, 500, 502, 503, 504):
                response.raise_for_status()
            
            # If it's a 4xx error (other than 429), don't retry, just log and skip symbol
            if 400 <= response.status_code < 500:
                logger.error(f"Client error {response.status_code} for {symbol}: {response.text}")
                return []
                
            response.raise_for_status()
            return response.json()

        for symbol in BINANCE.symbols:
            logger.info(f"Fetching data for {symbol}")
            try:
                data = fetch_binance_klines(symbol)
                if not data:
                    continue
            except Exception as e:
                # Circuit Breaker: If we exhaust retries (e.g. Binance is down), skip the task gracefully
                logger.error(f"Circuit Breaker tripped for Binance API on {symbol} after retries. Error: {e}")
                raise AirflowSkipException(f"Binance API unavailable: {e}")

            logger.info("Fetched data from Binance API", extra={"records": len(data), "symbol": symbol})

            for r in data:
                cur.execute("""
                    INSERT INTO bronze.crypto_price_raw (symbol, open, high, low, close, volume, event_time) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                """, (
                    symbol,
                float(r[1]), float(r[2]), float(r[3]),
                float(r[4]), float(r[5]),
                datetime.fromtimestamp(r[0] / 1000)
            ))

        conn.commit()
        conn.close()
        logger.info("Binance ingestion completed successfully")

    except Exception as e:
        logger.exception("Binance ingestion failed")
        raise


if __name__ == "__main__":
    ingest_binance_data()