import sys
import os
import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
    logger.info(f"=== Training [{horizon.upper()}] model on {source_table} | Target: {description} ===")

    try:
        engine = get_engine()
        # Only load columns we actually use to save memory
        cols_to_fetch = ["symbol", "event_time", "close"] + FEATURES
        query = f"SELECT {', '.join(cols_to_fetch)} FROM {source_table} ORDER BY symbol, event_time ASC"
        
        logger.info(f"Fetching {len(cols_to_fetch)} columns for training...")
        df = pd.read_sql(query, engine)
        engine.dispose()

        if df.empty:
            logger.error(f"No data in {source_table}. Run the ETL pipeline first.")
            return

        logger.info(f"Loaded {len(df):,} rows across {df['symbol'].nunique()} assets.")

        # Convert symbol to category to save memory
        df['symbol'] = df['symbol'].astype('category')

        # --- Forward-looking investor label (Must do BEFORE downsampling) ---
        logger.info(f"Computing {horizon} forward labels...")
        df['future_close'] = df.groupby('symbol')['close'].shift(-days_forward)
        df['target'] = (df['future_close'] > df['close'] * gain_threshold).astype(int)

        ml_df = df.dropna(subset=FEATURES + ['target', 'future_close']).copy()
        
        # Free up memory from raw df
        del df

        MAX_ROWS = 300_000
        if len(ml_df) > MAX_ROWS:
            logger.info(f"Downsampling labeled dataset from {len(ml_df):,} to {MAX_ROWS:,} for training stability.")
            ml_df = ml_df.sample(n=MAX_ROWS, random_state=42).sort_values(['symbol', 'event_time'])

        os.makedirs("/opt/airflow/models", exist_ok=True)
        clean_name = source_table.replace("public.", "")
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
        clean_name = source_table.replace("public.", "")
        model_path = f"/opt/airflow/models/{clean_name}_rf_{horizon}_model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"✅ [{horizon}] Saved -> {model_path}")

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
