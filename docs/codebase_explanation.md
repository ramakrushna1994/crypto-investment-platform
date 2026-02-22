# Crypto Investment Platform — Low-level Codebase Explanation

This document explains the repository at a low level: files, key functions, data flow, and runtime considerations. It focuses on the entrypoints and the components observed in `api/main.py`, `dags/crypto_daily_pipeline.py`, and `streamlit_app/app.py`, with inferred behavior for referenced `src` modules.

## High-level architecture

- Orchestration: Airflow DAGs in `dags/` run periodic ETL + ML tasks.
- Ingestion: `src.ingestion.*` collects data (Binance, yfinance, mutual funds API).
- ETL: Spark ETL (`src.etl.pyspark_etl`) consumes raw price tables and writes feature tables.
- Training & Signals: `src.recommendation.*` trains models and generates investment signals persisted to Postgres tables.
- API: `api/main.py` exposes a FastAPI endpoint to read signals from Postgres.
- UI: Streamlit app at `streamlit_app/app.py` queries Postgres, shows dashboards, and calls a local LLM (`src.llm.ollama_analyst`) for textual analysis.

## Database & tables (inferred)

- Postgres is used as the persistence layer (DSN provided by `src.config.settings.POSTGRES`).
- Raw ingestion tables: `public.crypto_price_raw`, `public.nifty50_price_raw`, etc.
- Feature tables (ETL output): `public.crypto_features_daily`, `public.nifty50_features_daily`, etc.
- Signals tables (ML output): `public.crypto_investment_signals`, `public.nifty50_investment_signals`, etc.

Each signals table rows include at least: `symbol`, `trade_date`, `signal_1y`, `confidence_1y`, `signal_5y`, `confidence_5y` (or legacy `signal`, `confidence`). Feature tables include `event_time`, `close`, technical indicators (`rsi_14`, `macd`, `ema_20`, `volatility_7d`, `moving_avg_7d`, `sma_50`, `sma_200`, `bb_upper`, `bb_lower`, `atr_14`, `stoch_k`, `stoch_d`).

## File: `api/main.py` (detailed)

- FastAPI app instance configured with title/description/version.
- `get_db_connection()` returns a `psycopg2.connect(POSTGRES.dsn)` connection — low-level DB cursor usage (no pool).
- `GET /api/v1/signals/{asset_class}` handler:
  - Validates `asset_class` is one of the known classes.
  - Builds a table name string like `{asset_class}_investment_signals`.
  - Runs a parameterized SQL query (uses `%s` placeholders) to fetch DISTINCT ON (symbol) rows ordered by `trade_date DESC`.
  - Filters by `confidence_1y >= min_confidence OR confidence_5y >= min_confidence`.
  - Returns JSON with `asset_class`, `count`, and `data` as list-of-dicts. On DB error raises HTTP 500.

Important low-level notes:
- The code uses string interpolation for `table_name` which is derived from validated input, but care must be taken if new classes are introduced.
- `psycopg2` cursors return `cur.description` used to build dicts; no explicit timezone/decimal handling is applied here.

## File: `dags/crypto_daily_pipeline.py` (detailed)

- Defines Airflow DAG `crypto_daily_pipeline` with schedule `30 2,14 * * 1-6` (two runs per day in UTC to align with IST).
- Default args: `retries=1` and `depends_on_past=False`.
- Imports functions from `src.ingestion`, `src.etl.pyspark_etl`, `src.recommendation.train_model`, `src.recommendation.strategy_engine`.
- For each asset class (crypto, nifty50, nifty_midcap, nifty_smallcap, mutual_funds) it declares a sequence of PythonOperator tasks:
  - `ingest_*` calls ingestion functions.
  - `spark_etl_*` calls `run_spark_etl` with `source_table` and `dest_table` parameters.
  - `train_model_*` calls `train_all_horizons` with `source_table`.
  - `generate_signals_*` calls `generate_signals` with `source_table` and `dest_table`.
- Orchestration wiring uses `>>` to chain ingest >> etl >> train >> signals for each pipeline in parallel.

Low-level implications:
- The DAG expects Python-callable functions that are importable in the Airflow worker environment — they must be side-effect free at import time and perform I/O inside the called function.
- `run_spark_etl` likely submits or runs a PySpark job that reads from Postgres or an upstream table, computes features, and writes back to Postgres (or a DB accessible by the app).

## File: `streamlit_app/app.py` (detailed)

- Uses `streamlit` for an interactive dashboard. `st.set_page_config` and `st.title` at top-level.
- Database access uses SQLAlchemy `create_engine` built from `POSTGRES` credentials and cached with `@st.cache_resource`.
- Cached data loaders using `@st.cache_data(ttl=600)`:
  - `load_latest_signals(engine, signals_table)` returns distinct latest signal per symbol.
  - `load_latest_features(engine, features_table)` returns latest features per symbol.
  - `load_symbol_history(engine, features_table, symbol)` returns full history for selected symbol.
- `render_asset_dashboard(...)` merges latest signals and latest features, computes human-readable ML label (emoji map), sorts by signal priority and confidence, and renders a table via `st.dataframe` with `on_select='rerun'` so selecting a row reloads and shows deep-dive.
- On selection, the app loads symbol history, formats metrics (`metric` widgets), computes historical returns (1y/3y/5y/10y), and shows charts using `st.line_chart`.
- AI analysis integrates with a local Ollama model via `src.llm.ollama_analyst.get_ollama_analysis` and `chat_with_ollama`. If unavailable, a deterministic rule-based fallback composes textual notes from indicators.

Low-level notes:
- Streamlit caches are used to avoid repeated DB hits — TTL is 10 minutes.
- The app expects certain columns in feature and signal tables; missing columns fall back to legacy names.
- `st.chat_input` and `st.chat_message` are used to provide an AI chat; chat history persisted in `st.session_state`.

## `requirements.txt` highlights

- Key packages: `pyspark`, `streamlit`, `uvicorn[standard]`, `fastapi`, `psycopg2-binary`, `pandas`, `requests`, `scikit-learn`, `apache-airflow-providers-fab`, `yfinance`.

## Runtime & deployment notes (low-level)

- Airflow needs the `src` package on the PYTHONPATH of the worker and scheduler processes.
- The FastAPI app likely runs with `uvicorn api.main:app` and needs access to Postgres; consider using a DB connection pool instead of raw `psycopg2.connect` per request.
- Spark jobs must be runnable in the environment where Airflow calls `run_spark_etl`. Confirm PySpark config (master, jars, environment variables).
- The Streamlit app uses `create_engine` with `pool_size` and `max_overflow`, but the connection string depends on `POSTGRES` credentials; running multiple Streamlit instances may require adjusting pooling.

## Next actions and suggestions for deeper low-level inspection

- Inspect `src/ingestion/*` to document exact ingestion rate limits, batching and API pagination logic.
- Inspect `src/etl/pyspark_etl.py` to list transformation steps, feature engineering math, and schema of feature tables.
- Inspect `src/recommendation/*` to see ML model types, feature sets per horizon, model serialization (joblib/pickle), and the SQL schema used to persist signals.
- Inspect `src/llm/*` to verify prompts, context window trimming, and safety/hallucination mitigations.
