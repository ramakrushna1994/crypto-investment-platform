"""
postgres.py
bulk writes to postgres using COPY.
"""
from src.config.settings import POSTGRES
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    """Returns a reusable sqlalchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}"
            f"@{POSTGRES.host}/{POSTGRES.db}",
            pool_size=2, max_overflow=0
        )
    return _engine


def write_features_df(df: pd.DataFrame, table: str):
    """
    Writes a pandas DataFrame to PostgreSQL using psycopg2 COPY.
    """
    schema, tbl = table.split(".", 1) if "." in table else ("public", table)
    full_table = f'{schema}."{tbl}"'

    # Map pandas dtypes → Postgres types
    type_map = {
        "int": "BIGINT",
        "float": "DOUBLE PRECISION",
        "datetime": "TIMESTAMP",
        "object": "TEXT",
        "bool": "BOOLEAN",
    }

    def pg_type(dtype):
        dtype_str = str(dtype)
        for k, v in type_map.items():
            if k in dtype_str:
                return v
        return "TEXT"

    col_defs = ", ".join(f'"{c}" {pg_type(t)}' for c, t in df.dtypes.items())

    conn = psycopg2.connect(POSTGRES.dsn)
    cur = conn.cursor()

    try:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {full_table} ({col_defs})")
        cur.execute(f"TRUNCATE TABLE {full_table}")

        # Stream DataFrame as CSV into Postgres via COPY
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep="\\N")
        buffer.seek(0)

        cols = ", ".join(f'"{c}"' for c in df.columns)
        cur.copy_expert(
            f"COPY {full_table} ({cols}) FROM STDIN WITH CSV NULL '\\N'",
            buffer
        )

        conn.commit()
        logger.info(f"wrote {len(df)} rows to {full_table}")

    except Exception as e:
        conn.rollback()
        logger.exception(f"write failed: {e}")
        raise
    finally:
        cur.close()
        conn.close()
