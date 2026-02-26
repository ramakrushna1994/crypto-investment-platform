from fastapi import FastAPI, HTTPException, Query
import psycopg2
from src.config.settings import POSTGRES

app = FastAPI(
    title="AI Quant Investment API",
    description="REST API for accessing AI-generated investment signals.",
    version="1.0.0"
)

def get_db_connection():
    return psycopg2.connect(POSTGRES.dsn)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Investment Signal API is running."}

@app.get("/api/v1/signals/{asset_class}")
def get_investment_signals(
    asset_class: str,
    limit: int = 50,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Fetch the latest investment signals for a given asset class.
    Valid asset classes: 'crypto', 'mutual_funds', 'nifty50', 'nifty_midcap', 'nifty_smallcap'
    """
    valid_classes = ['crypto', 'mutual_funds', 'nifty50', 'nifty_midcap', 'nifty_smallcap']
    if asset_class not in valid_classes:
        raise HTTPException(status_code=400, detail=f"Invalid asset class. Must be one of {valid_classes}")

    table_name = f"gold.{asset_class}_investment_signals"
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # We fetch the most recent signal per symbol that meets the confidence threshold
        query = f"""
            SELECT DISTINCT ON (symbol) 
                symbol, trade_date, signal_1y, confidence_1y, signal_5y, confidence_5y
            FROM {table_name}
            WHERE confidence_1y >= %s OR confidence_5y >= %s
            ORDER BY symbol, trade_date DESC
            LIMIT %s
        """
        cur.execute(query, (min_confidence, min_confidence, limit))
        
        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return {
            "asset_class": asset_class,
            "count": len(results),
            "data": results
        }
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

