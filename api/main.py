from fastapi import FastAPI
import psycopg2
from src.config.settings import POSTGRES

app = FastAPI()

@app.get("/signals")
def get_signals():
    conn = psycopg2.connect(POSTGRES.dsn)
    cur = conn.cursor()
    cur.execute("SELECT * FROM crypto_investment_signals")
    rows = cur.fetchall()
    conn.close()
    return rows
