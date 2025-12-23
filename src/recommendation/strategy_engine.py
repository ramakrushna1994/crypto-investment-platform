import psycopg2
from src.config.settings import POSTGRES
import logging

logger = logging.getLogger(__name__)

def generate_signals():
    logger.info("🚀 Starting strategy engine")
    try:
        conn = psycopg2.connect(POSTGRES.dsn)
        cur = conn.cursor()

        cur.execute("""SELECT
                symbol,
                CURRENT_DATE,
                CASE
                    WHEN volatility_7d < 5 THEN 'BUY'
                    ELSE 'HOLD'
                END,
                random()
            FROM crypto_features_daily""")
        rows = cur.fetchall()

        logger.info("Fetched feature data", extra={"rows": len(rows)})

        cur.execute("""
            CREATE TABLE IF NOT EXISTS crypto_investment_signals (
                symbol TEXT,
                trade_date DATE,
                signal TEXT,
                confidence NUMERIC,
                PRIMARY KEY (symbol, trade_date)
            )
        """)

        cur.execute("""
            INSERT INTO crypto_investment_signals
            SELECT
                symbol,
                CURRENT_DATE,
                CASE
                    WHEN volatility_7d < 5 THEN 'BUY'
                    ELSE 'HOLD'
                END,
                random()
            FROM crypto_features_daily
            ON CONFLICT DO NOTHING
        """)

        conn.commit()
        conn.close()

        logger.info("✅ Strategy signals generated successfully")

    except Exception:
        logger.exception("❌ Strategy engine failed")
        raise

    finally:
        try:
            if 'conn' in locals() and conn is not None:
                conn.close()
                logger.info("Database connection closed")
        except Exception:
            logger.exception("Failed closing DB connection")


if __name__ == "__main__":
    generate_signals()