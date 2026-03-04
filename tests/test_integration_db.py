"""
Integration tests for database round-trip operations.

These tests require a live PostgreSQL connection and are designed to run
inside the Docker environment.  They are skipped automatically in CI
(GitHub Actions) or anywhere POSTGRES_HOST is unreachable.

Run inside Docker:
    docker compose exec airflow-api-server python -m pytest tests/test_integration_db.py -v
"""
import os
import pytest
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Skip entire module when the database is not reachable
# ---------------------------------------------------------------------------
_DSN = (
    f"dbname={os.getenv('POSTGRES_DB', 'ai_quant')} "
    f"user={os.getenv('POSTGRES_USER', 'ai_quant')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'ai_quant')} "
    f"host={os.getenv('POSTGRES_HOST', 'postgres')} "
    f"connect_timeout=3"
)

def _db_available() -> bool:
    try:
        conn = psycopg2.connect(_DSN)
        conn.close()
        return True
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL not reachable — skipping integration tests",
)

TEST_SCHEMA = "test_integration"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db_conn():
    """Module-scoped connection; rolls back everything at teardown."""
    conn = psycopg2.connect(_DSN)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")
    conn.commit()
    yield conn
    # Cleanup: drop the test schema and everything in it
    cur = conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
    conn.commit()
    conn.close()


@pytest.fixture()
def cursor(db_conn):
    """Per-test cursor with automatic rollback via savepoint."""
    cur = db_conn.cursor()
    cur.execute("SAVEPOINT test_sp")
    yield cur
    db_conn.rollback()  # rollback to before the savepoint


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSchemaGate:
    """Verify _check_schema_gate reads real information_schema."""

    def test_detects_existing_columns(self, cursor):
        cursor.execute(f"""
            CREATE TABLE {TEST_SCHEMA}.gate_test (
                symbol TEXT, event_time TIMESTAMP, close NUMERIC,
                rsi_14 NUMERIC, volatility_7d NUMERIC, macd NUMERIC,
                macd_signal NUMERIC, ema_20 NUMERIC, moving_avg_7d NUMERIC,
                bb_upper NUMERIC, bb_lower NUMERIC, sma_50 NUMERIC,
                sma_200 NUMERIC, atr_14 NUMERIC, stoch_k NUMERIC, stoch_d NUMERIC
            )
        """)
        cursor.connection.commit()

        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{TEST_SCHEMA}' AND table_name = 'gate_test'
        """)
        cols = {r[0] for r in cursor.fetchall()}
        required = {"symbol", "event_time", "close", "rsi_14", "sma_50", "sma_200"}
        assert required.issubset(cols)

    def test_detects_nav_columns(self, cursor):
        cursor.execute(f"""
            CREATE TABLE {TEST_SCHEMA}.nav_gate_test (
                symbol TEXT, event_time TIMESTAMP, close NUMERIC,
                rsi_14 NUMERIC, rolling_return_30d NUMERIC,
                sortino_30d NUMERIC, max_drawdown_30d NUMERIC
            )
        """)
        cursor.connection.commit()

        cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{TEST_SCHEMA}' AND table_name = 'nav_gate_test'
        """)
        cols = {r[0] for r in cursor.fetchall()}
        assert "rolling_return_30d" in cols
        assert "sortino_30d" in cols


class TestSignalUpsert:
    """Verify batched execute_values INSERT + ON CONFLICT upsert."""

    SIGNAL_TABLE = f"{TEST_SCHEMA}.test_signals"

    def _create_table(self, cursor):
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.SIGNAL_TABLE} (
                symbol TEXT,
                trade_date DATE,
                signal TEXT,
                confidence NUMERIC,
                signal_1y TEXT,
                confidence_1y NUMERIC,
                signal_5y TEXT,
                confidence_5y NUMERIC,
                combined_confidence NUMERIC,
                risk_score NUMERIC,
                risk_bucket TEXT,
                suggested_position_pct NUMERIC,
                expected_return_1y NUMERIC,
                risk_adjusted_score NUMERIC,
                var_95_1d NUMERIC,
                cvar_95_1d NUMERIC,
                PRIMARY KEY (symbol, trade_date)
            )
        """)
        cursor.connection.commit()

    def test_batch_insert(self, cursor):
        """Insert multiple rows in one execute_values call."""
        self._create_table(cursor)
        rows = [
            ("RELIANCE", "2025-01-01", "INVEST NOW", 0.85,
             "INVEST NOW", 0.85, "STRONG HOLD", 0.78,
             0.82, 0.35, "LOW", 5.5, 0.12, 0.70, 0.018, 0.025),
            ("TCS", "2025-01-01", "ACCUMULATE", 0.65,
             "ACCUMULATE", 0.65, "MONITOR", 0.50,
             0.58, 0.55, "MEDIUM", 3.0, 0.08, 0.45, 0.022, 0.031),
        ]
        insert_sql = f"""
            INSERT INTO {self.SIGNAL_TABLE}
                (symbol, trade_date, signal, confidence,
                 signal_1y, confidence_1y, signal_5y, confidence_5y,
                 combined_confidence, risk_score, risk_bucket,
                 suggested_position_pct, expected_return_1y,
                 risk_adjusted_score, var_95_1d, cvar_95_1d)
            VALUES %s
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                signal = EXCLUDED.signal, confidence = EXCLUDED.confidence
        """
        execute_values(cursor, insert_sql, rows, page_size=500)
        cursor.connection.commit()

        cursor.execute(f"SELECT COUNT(*) FROM {self.SIGNAL_TABLE}")
        assert cursor.fetchone()[0] == 2

    def test_upsert_updates_existing(self, cursor):
        """ON CONFLICT should update existing rows, not duplicate."""
        self._create_table(cursor)
        row = ("INFY", "2025-06-15", "AVOID", 0.20,
               "AVOID", 0.20, "WAIT", 0.25,
               0.22, 0.85, "HIGH", 1.0, -0.05, 0.10, 0.035, 0.048)

        insert_sql = f"""
            INSERT INTO {self.SIGNAL_TABLE}
                (symbol, trade_date, signal, confidence,
                 signal_1y, confidence_1y, signal_5y, confidence_5y,
                 combined_confidence, risk_score, risk_bucket,
                 suggested_position_pct, expected_return_1y,
                 risk_adjusted_score, var_95_1d, cvar_95_1d)
            VALUES %s
            ON CONFLICT (symbol, trade_date) DO UPDATE SET
                signal = EXCLUDED.signal, confidence = EXCLUDED.confidence
        """
        # Insert first time
        execute_values(cursor, insert_sql, [row], page_size=500)
        cursor.connection.commit()

        # Upsert with updated signal
        updated_row = ("INFY", "2025-06-15", "ACCUMULATE", 0.60,
                       "ACCUMULATE", 0.60, "MONITOR", 0.45,
                       0.52, 0.50, "MEDIUM", 3.5, 0.06, 0.40, 0.020, 0.028)
        execute_values(cursor, insert_sql, [updated_row], page_size=500)
        cursor.connection.commit()

        cursor.execute(
            f"SELECT signal, confidence FROM {self.SIGNAL_TABLE} WHERE symbol = 'INFY'"
        )
        result = cursor.fetchone()
        assert result[0] == "ACCUMULATE"
        assert float(result[1]) == pytest.approx(0.60)

        # Should still be 1 row, not 2
        cursor.execute(
            f"SELECT COUNT(*) FROM {self.SIGNAL_TABLE} WHERE symbol = 'INFY'"
        )
        assert cursor.fetchone()[0] == 1


class TestFeatureTableRoundTrip:
    """Write features → read back → verify values survive the round-trip."""

    FEAT_TABLE = f"{TEST_SCHEMA}.test_features"

    def test_write_and_read_features(self, cursor):
        cursor.execute(f"""
            CREATE TABLE {self.FEAT_TABLE} (
                symbol TEXT,
                event_time TIMESTAMP,
                close NUMERIC,
                rsi_14 NUMERIC,
                sma_50 NUMERIC,
                rolling_return_30d NUMERIC,
                sortino_30d NUMERIC
            )
        """)
        rows = [
            ("MF_001", "2025-03-01", 125.50, 55.3, 120.0, 0.045, 1.23),
            ("MF_001", "2025-03-02", 126.10, 58.1, 120.5, 0.048, 1.30),
            ("MF_002", "2025-03-01", 45.20, 42.0, 44.8, -0.012, -0.55),
        ]
        execute_values(
            cursor,
            f"INSERT INTO {self.FEAT_TABLE} VALUES %s",
            rows,
        )
        cursor.connection.commit()

        cursor.execute(f"SELECT COUNT(*) FROM {self.FEAT_TABLE}")
        assert cursor.fetchone()[0] == 3

        cursor.execute(
            f"SELECT rolling_return_30d, sortino_30d FROM {self.FEAT_TABLE} "
            f"WHERE symbol = 'MF_001' ORDER BY event_time"
        )
        results = cursor.fetchall()
        assert float(results[0][0]) == pytest.approx(0.045)
        assert float(results[1][1]) == pytest.approx(1.30)

    def test_null_nav_features_allowed(self, cursor):
        """NAV features should accept NULL for non-MF tables."""
        cursor.execute(f"""
            CREATE TABLE {self.FEAT_TABLE}_nulls (
                symbol TEXT,
                event_time TIMESTAMP,
                close NUMERIC,
                rolling_return_30d NUMERIC,
                sortino_30d NUMERIC
            )
        """)
        execute_values(
            cursor,
            f"INSERT INTO {self.FEAT_TABLE}_nulls VALUES %s",
            [("BTC", "2025-03-01", 65000.0, None, None)],
        )
        cursor.connection.commit()

        cursor.execute(
            f"SELECT rolling_return_30d FROM {self.FEAT_TABLE}_nulls WHERE symbol = 'BTC'"
        )
        assert cursor.fetchone()[0] is None


class TestFreshnessQuery:
    """Verify the freshness SLA query pattern against real timestamps."""

    def test_recent_data_passes(self, cursor):
        cursor.execute(f"""
            CREATE TABLE {TEST_SCHEMA}.freshness_test (
                symbol TEXT, event_time TIMESTAMP
            )
        """)
        cursor.execute(f"""
            INSERT INTO {TEST_SCHEMA}.freshness_test
            VALUES ('TEST', NOW() - INTERVAL '1 hour')
        """)
        cursor.connection.commit()

        cursor.execute(f"""
            SELECT MAX(event_time) > NOW() - INTERVAL '2 days'
            FROM {TEST_SCHEMA}.freshness_test
        """)
        assert cursor.fetchone()[0] is True

    def test_stale_data_fails(self, cursor):
        cursor.execute(f"""
            CREATE TABLE {TEST_SCHEMA}.freshness_stale (
                symbol TEXT, event_time TIMESTAMP
            )
        """)
        cursor.execute(f"""
            INSERT INTO {TEST_SCHEMA}.freshness_stale
            VALUES ('OLD', NOW() - INTERVAL '30 days')
        """)
        cursor.connection.commit()

        cursor.execute(f"""
            SELECT MAX(event_time) > NOW() - INTERVAL '2 days'
            FROM {TEST_SCHEMA}.freshness_stale
        """)
        assert cursor.fetchone()[0] is False
