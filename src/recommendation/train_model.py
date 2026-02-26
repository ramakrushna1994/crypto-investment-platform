"""
train_model.py

Trains a Random Forest Classifier to predict mid-to-long term 
price growth probabilities based on daily technical features. 
Supports 1-year and 5-year investment horizons.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Investor-grade horizon configurations (no short-term trading horizons)
# Maps horizon label -> (forward trading days, minimum gain threshold, description)
INVESTOR_HORIZONS = {
    '1y':  (252,  1.10, ">=10% gain in 1 year"),
    '5y':  (1260, 1.50, ">=50% gain in 5 years"),
}

FEATURES = [
    'rsi_14', 'volatility_7d', 'macd', 'macd_signal', 'ema_20',
    'moving_avg_7d', 'bb_upper', 'bb_lower', 'sma_50', 'sma_200',
    'atr_14', 'stoch_k', 'stoch_d'
]

def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}@{POSTGRES.host}/{POSTGRES.db}"
    )

def train_model(source_table="public.mutual_funds_features_daily", horizon="1y"):
    """Train a Random Forest for a given investment horizon and save the model artifact."""
    if horizon not in INVESTOR_HORIZONS:
        raise ValueError(f"Invalid horizon '{horizon}'. Choose from: {list(INVESTOR_HORIZONS.keys())}")

    days_forward, gain_threshold, description = INVESTOR_HORIZONS[horizon]
    logger.info(f"Training {horizon.upper()} model on {source_table}")

    try:
        engine = get_engine()
        
        # Construct feature selection dynamically
        feature_cols = ', '.join(FEATURES)
        where_clauses = ["future_close IS NOT NULL"]
        for f in FEATURES:
            where_clauses.append(f"{f} IS NOT NULL")
        query_where = " AND ".join(where_clauses)
        
        # Use CTE to calculate labels in SQL to prevent OOM
        query = f"""
            WITH labeled AS (
                SELECT 
                    symbol, 
                    event_time, 
                    close,
                    LEAD(close, {days_forward}) OVER (PARTITION BY symbol ORDER BY event_time ASC) AS future_close,
                    {feature_cols}
                FROM {source_table}
            )
            SELECT * FROM labeled
            WHERE {query_where}
            ORDER BY symbol, event_time ASC
        """

        logger.info(f"Fetching all eligible labeled rows from DB for {horizon} horizon...")
        df = pd.read_sql(query, engine)
        engine.dispose()

        ml_df = df.copy()

        if not ml_df.empty:
            ml_df['symbol'] = ml_df['symbol'].astype('category')
            ml_df['target'] = (ml_df['future_close'] > ml_df['close'] * gain_threshold).astype(int)
        
        logger.info(f"Loaded {len(ml_df):,} labeled rows across {ml_df['symbol'].nunique()} assets.")

        os.makedirs("/opt/airflow/models", exist_ok=True)
        # Get base table name for model path (handle schema.table_name)
        clean_name = source_table.split(".")[-1] if "." in source_table else source_table
        model_path = f"/opt/airflow/models/{clean_name}_rf_{horizon}_model.joblib"

        if ml_df.empty:
            logger.warning(f"Not enough history for {horizon} horizon (need {days_forward}+ days). Saving fallback 'HOLD' model.")
            # Create a simple model that always predicts 0
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy="constant", constant=0)
            # Fit on a single dummy row so it has the right classes
            dummy_X = pd.DataFrame([[0]*len(FEATURES)], columns=FEATURES)
            dummy_y = pd.Series([0])
            model.fit(dummy_X, dummy_y)
            joblib.dump(model, model_path)
            return

        X = ml_df[FEATURES]
        y = ml_df['target']

        pos_rate = y.mean() * 100
        logger.info(f"Dataset: {X.shape[0]:,} samples. Positive (will grow) rate: {pos_rate:.1f}%")

        # Chronological split — use last 20% of time as test to avoid look-ahead bias
        split_idx = int(len(ml_df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

        # More trees + depth for longer horizons since patterns are more complex
        n_estimators = 200 if horizon == '5y' else 100
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12, random_state=42,
            class_weight='balanced',  # Handle imbalanced classes (fewer 5Y winners)
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"[{horizon}] Test Accuracy: {acc * 100:.2f}%")
        logger.info("\n" + classification_report(y_test, preds))

        os.makedirs("/opt/airflow/models", exist_ok=True)
        # Get base table name for model path (handle schema.table_name)
        clean_name = source_table.split(".")[-1] if "." in source_table else source_table
        model_path = f"/opt/airflow/models/{clean_name}_rf_{horizon}_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully: {model_path}")

    except Exception as e:
        logger.exception(f"Training failed for {source_table} [{horizon}]: {e}")
        raise


def train_all_horizons(source_table="public.mutual_funds_features_daily"):
    """Train 1Y and 5Y investor models for the given feature table. Called by Airflow DAG."""
    for horizon in INVESTOR_HORIZONS:
        train_model(source_table=source_table, horizon=horizon)


if __name__ == "__main__":
    src_table = sys.argv[1] if len(sys.argv) > 1 else "public.mutual_funds_features_daily"
    for h in INVESTOR_HORIZONS:
        train_model(src_table, horizon=h)
