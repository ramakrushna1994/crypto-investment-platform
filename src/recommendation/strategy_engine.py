"""
strategy_engine.py

Uses trained Random Forest models to inference the latest features for each 
asset and generate clear "BUY/SELL/HOLD" signals with probability confidence scores.
"""
import sys
import os
import psycopg2
import pandas as pd
import numpy as np
import joblib
import logging
from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)

# Investor-grade signal thresholds for each horizon
# The ML model outputs P(asset meets growth target). These thresholds map that to human advice.
SIGNAL_THRESHOLDS = {
    '1y': [
        (0.72, 'INVEST NOW'),     # Very high probability of 10%+ gain in 1 year
        (0.58, 'ACCUMULATE'),     # Good probability, consider building position
        (0.42, 'MONITOR'),        # Uncertain - watch for a better entry
        (0.25, 'WAIT'),           # Below average near-term outlook
        (0.00, 'AVOID'),          # Low probability of growth in 1 year
    ],
    '5y': [
        (0.70, 'STRONG HOLD'),    # Very high probability of 50%+ gain over 5 years — core portfolio
        (0.55, 'ACCUMULATE'),     # Good 5Y compounding expected — buy on dips
        (0.40, 'MONITOR'),        # Moderate 5Y outlook — too early to commit
        (0.20, 'WAIT'),           # Low 5Y probability — look for better alternatives
        (0.00, 'AVOID'),          # Fundamentally weak 5Y outlook
    ],
}

FEATURES = [
    'rsi_14', 'volatility_7d', 'macd', 'macd_signal', 'ema_20',
    'moving_avg_7d', 'bb_upper', 'bb_lower', 'sma_50', 'sma_200',
    'atr_14', 'stoch_k', 'stoch_d'
]

def _prob_to_signal(prob, horizon):
    for threshold, label in SIGNAL_THRESHOLDS[horizon]:
        if prob >= threshold:
            return label
    return SIGNAL_THRESHOLDS[horizon][-1][1]

def _safe_predict_proba(model, X):
    """Safely get the probability of the positive class (1)."""
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        # Only one class present in the model
        if model.classes_[0] == 1:
            return probs[:, 0]  # All class 1
        else:
            return np.zeros(probs.shape[0])  # All class 0
    return probs[:, 1]

def generate_signals(source_table="public.mutual_funds_features_daily", dest_table="public.mutual_funds_investment_signals"):
    """Fetch latest features, run inference models, and update signals table."""
    logger.info(f"Generating signals: {source_table} -> {dest_table}")

    # Get base table name for model path (handle schema.table_name)
    clean_name = source_table.split(".")[-1] if "." in source_table else source_table

    # Load both investor horizon models
    models = {}
    for horizon in ['1y', '5y']:
        path = f"/opt/airflow/models/{clean_name}_rf_{horizon}_model.joblib"
        if not os.path.exists(path):
            logger.error(f"Model missing: {path}. Run train_model.py first.")
            raise FileNotFoundError(f"Model not found: {path}")
        models[horizon] = joblib.load(path)
        logger.info(f"Loaded [{horizon}] model from {path}")

    try:
        conn = psycopg2.connect(POSTGRES.dsn)

        # Fetch the most recent feature record per symbol
        query = f"""
            SELECT DISTINCT ON (symbol) symbol, event_time::DATE as trade_date,
                {', '.join(FEATURES)}
            FROM {source_table}
            WHERE rsi_14 IS NOT NULL AND sma_50 IS NOT NULL AND sma_200 IS NOT NULL
            ORDER BY symbol, event_time DESC
        """
        df = pd.read_sql(query, conn)

        if df.empty:
            logger.warning(f"No feature data in {source_table}")
            return

        X = df[FEATURES].fillna(0)

        # Run inference for both horizons
        proba_1y = _safe_predict_proba(models['1y'], X)
        proba_5y = _safe_predict_proba(models['5y'], X)

        cur = conn.cursor()

        # Expanded signals table with both investment horizons
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {dest_table} (
                symbol       TEXT,
                trade_date   DATE,
                signal       TEXT,       -- 1-year outlook (primary signal shown in UI)
                confidence   NUMERIC,    -- 1-year probability
                signal_1y    TEXT,
                confidence_1y NUMERIC,
                signal_5y    TEXT,
                confidence_5y NUMERIC,
                PRIMARY KEY (symbol, trade_date)
            )
        """)

        # Add new columns to existing tables if they were created before this update
        for col_def in [
            "signal_1y TEXT", "confidence_1y NUMERIC",
            "signal_5y TEXT", "confidence_5y NUMERIC"
        ]:
            col_name = col_def.split()[0]
            try:
                cur.execute(f"ALTER TABLE {dest_table} ADD COLUMN IF NOT EXISTS {col_def}")
            except Exception:
                pass

        for idx, row in df.iterrows():
            p1y = float(proba_1y[idx])
            p5y = float(proba_5y[idx])
            sig1y = _prob_to_signal(p1y, '1y')
            sig5y = _prob_to_signal(p5y, '5y')

            cur.execute(f"""
                INSERT INTO {dest_table}
                    (symbol, trade_date, signal, confidence, signal_1y, confidence_1y, signal_5y, confidence_5y)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, trade_date) DO UPDATE SET
                    signal = EXCLUDED.signal,
                    confidence = EXCLUDED.confidence,
                    signal_1y = EXCLUDED.signal_1y,
                    confidence_1y = EXCLUDED.confidence_1y,
                    signal_5y = EXCLUDED.signal_5y,
                    confidence_5y = EXCLUDED.confidence_5y
            """, (row['symbol'], row['trade_date'], sig1y, p1y, sig1y, p1y, sig5y, p5y))

        conn.commit()
        cur.close()
        logger.info(f"Inserted {len(df)} signals into {dest_table}")

    except Exception:
        logger.exception("Investor signal engine failed")
        raise
    finally:
        try:
            if 'conn' in locals() and conn is not None:
                conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    src_table = sys.argv[1] if len(sys.argv) > 1 else "public.mutual_funds_features_daily"
    dst_table = sys.argv[2] if len(sys.argv) > 2 else "public.mutual_funds_investment_signals"
    generate_signals(src_table, dst_table)