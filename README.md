# AI Quant Investment Engine

An open, end-to-end investment analytics product for developers.

This project demonstrates how to build a production-style analytics pipeline that goes from market data ingestion to ML-driven investment signals, with auditability, API access, dashboard UX, and automated email reporting.

## Product Positioning
`AI Quant Investment Engine` is designed as a reference product for the tech community:
- A reproducible architecture for data + ML + serving.
- A practical template for medallion data pipelines (`bronze` -> `silver` -> `gold`).
- A real-world example of adding audit, quality checks, and reconciliation into ML operations.
- A deployment-ready stack using Docker, Airflow, PostgreSQL, FastAPI, Streamlit, and optional local LLM.

This is educational tooling, not financial advice.

## Why This Matters To The Tech Community
- It is opinionated but extensible: enough structure to ship, enough flexibility to learn and extend.
- It covers full lifecycle engineering, not only model training.
- It includes both user-facing delivery channels (UI/API/email) and operator-facing controls (audit and reconciliation).
- It demonstrates how to make ML outputs consumable by non-technical users.

## Core Capabilities
- Multi-source ingestion:
  - `Binance` (crypto OHLCV)
  - `NSE + yfinance` (Nifty 50 / Midcap / Smallcap)
  - `MFAPI` (mutual fund NAV history)
- Distributed feature engineering with PySpark technical indicators.
- Dual-horizon model training (`1y`, `5y`) using RandomForest.
- Signal generation with probability-to-label business mapping.
- Auditing framework:
  - ETL start/end logging
  - source-target reconciliation
  - quality metrics and summary views
- Distribution:
  - Streamlit dashboard for exploration
  - FastAPI endpoint for machine consumption
  - formatted Excel report over SMTP email
- Optional local LLM narrative layer via Ollama.

## Tech Stack
- Infrastructure: Docker, Docker Compose, Apache Airflow
- Database: PostgreSQL 15
- Data/ML: Python, PySpark, pandas, scikit-learn, psycopg2
- API/UI: FastAPI, Streamlit
- Local LLM: Ollama (`llama3.2`)

## High-Level Architecture

```mermaid
graph LR
    A[External Market APIs] --> B[Bronze Raw Tables]
    B --> C[Silver Feature Tables]
    C --> D[Gold Signal Tables]
    D --> E[FastAPI]
    D --> F[Streamlit]
    D --> G[Email Report XLSX]
    C --> H[Model Training Artifacts]
    H --> I[Signal Engine]
    I --> D
    J[Audit Schema] --> K[Quality + Reconciliation + Daily Audit Report]
```

## Airflow DAG Flow (`ai_quant_daily_pipeline`)

```mermaid
graph TD
    Start[start_parallel]

    C1[crypto: ingest -> etl -> train -> signals -> audit -> quality -> reconcile]
    N1[nifty50: ingest -> etl -> train -> signals -> audit -> quality -> reconcile]

    Wait[wait_for_parallel]

    M1[nifty_midcap chain]
    S1[nifty_smallcap chain]
    F1[mutual_funds chain]

    AR[generate_daily_audit_report]
    EM[send_opportunity_email]

    Start --> C1 --> Wait
    Start --> N1 --> Wait
    Wait --> M1 --> S1 --> F1 --> AR --> EM
    C1 --> AR
    N1 --> AR
```

## Repository Map
- `dags/ai_quant_daily_pipeline.py` - orchestration.
- `src/ingestion/` - source connectors and raw writes.
- `src/etl/pyspark_etl.py` - feature generation into `silver`.
- `src/recommendation/train_model.py` - model training.
- `src/recommendation/strategy_engine.py` - inference + signal persistence.
- `src/recommendation/email_alert.py` - report generation + SMTP dispatch.
- `src/audit/` + `database/audit_reconciliation_framework.sql` - observability layer.
- `api/main.py` - REST API.
- `streamlit_app/app.py` - dashboard application.
- `docs/ai_quant_codebase_low_level_guide.pdf` - low-level technical walkthrough.

## Data Model (Simplified)
- `bronze.*_price_raw`: ingested OHLCV/NAV history.
- `silver.*_features_daily`: technical indicators and engineered features.
- `gold.*_investment_signals`:
  - `signal`, `confidence` (primary 1-year view)
  - `signal_1y`, `confidence_1y`
  - `signal_5y`, `confidence_5y`
- `audit.*`: ETL logs, reconciliation, and quality metrics.

## Quickstart
Prerequisites:
- Docker + Docker Compose
- ~8GB RAM minimum (16GB recommended with local LLM)

1. Clone and start:
```bash
git clone https://github.com/ramakrushna1994/ai-quant-investment-engine.git
cd ai-quant-investment-engine
docker compose up -d --build
```

2. Configure secrets via `.env` (do not commit this file):
- Postgres credentials
- SMTP settings (`SMTP_USER`, `SMTP_PASSWORD`, `RECEIVER_EMAIL`, etc.)

3. Trigger first run:
- Open Airflow: `http://localhost:8080` (default local auth: `admin/admin`)
- Enable and trigger DAG: `ai_quant_daily_pipeline`

4. Access product surfaces:
- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- Example API call:
```bash
curl "http://localhost:8000/api/v1/signals/mutual_funds?min_confidence=0.7&limit=10"
```

## Product Strategy: How We Reciprocate To The Tech Community
We want this to be a community product, not just a code dump. Our reciprocity model:

1. Open implementation patterns:
- Keep architecture and orchestration explicit (DAG + schema + model flow).
- Document decisions, tradeoffs, and failure modes.

2. Reproducible engineering:
- One-command local environment with Docker Compose.
- Deterministic pipeline behavior through explicit DAG stages and schemas.

3. Educational depth:
- Maintain low-level technical docs and code references.
- Explain not only "what" but "why" at system boundaries.

4. Responsible analytics:
- Make confidence and signal semantics explicit.
- Preserve audit and reconciliation visibility as first-class product behavior.

5. Contributor-friendly roadmap:
- Break work into trackable units (ingestion, ETL, modeling, serving, audits).
- Encourage issue-driven contributions with clear acceptance criteria.

## Community Contribution Tracks
If you want to contribute, pick one lane:
- Data connectors: add new market/exchange/fund sources.
- Feature engineering: add indicators with proper backfill behavior.
- Modeling: improve calibration, backtesting, and model evaluation transparency.
- Product UX: better ranking/filtering, explainability, and signal comparability.
- Reliability: tests, retries, drift detection, and quality guardrails.
- Governance: model cards, changelog discipline, benchmark datasets.

## Known Limitations
- Local LLM latency depends on host hardware.
- Spark jobs are memory-sensitive for large symbol universes.
- Signal quality depends on feature/data quality and is not a guarantee of returns.
- Some SQL table references are dynamic; keep inputs controlled and trusted.

## Responsible Use
- This project is for educational and engineering purposes.
- It does not provide licensed investment advice.
- Always validate outcomes with your own research and risk controls.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
