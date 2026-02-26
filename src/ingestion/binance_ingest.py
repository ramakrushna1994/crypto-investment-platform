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

logger = logging.getLogger(__name__)


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

        for symbol in BINANCE.symbols:
            logger.info(f"Fetching data for {symbol}")
            response = requests.get(
                BINANCE.base_url,
                params={
                    "symbol": symbol,
                    "interval": BINANCE.interval,
                    "limit": BINANCE.limit
                }
            )
            response.raise_for_status()
            data = response.json()
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