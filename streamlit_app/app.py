import streamlit as st
import psycopg2
import pandas as pd
from src.config.settings import POSTGRES

st.title("📈 Crypto Investment Dashboard")

conn = psycopg2.connect(POSTGRES.dsn)
df = pd.read_sql("SELECT * FROM public.crypto_investment_signals", conn)
st.dataframe(df)
