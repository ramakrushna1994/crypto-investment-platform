# streamlit_app/app.py
import streamlit as st
import psycopg2
import pandas as pd
from src.config.settings import POSTGRES

st.set_page_config(page_title="Crypto AI Platform", layout="wide")
st.title("📈 Crypto AI Investment Dashboard")

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(POSTGRES.dsn)

conn = get_db_connection()

try:
    # Fetch Signals
    signals_df = pd.read_sql("SELECT * FROM public.crypto_investment_signals ORDER BY trade_date DESC", conn)
    
    # Fetch Features for plotting
    features_df = pd.read_sql("SELECT * FROM public.crypto_features_daily ORDER BY event_time ASC", conn)

    if signals_df.empty or features_df.empty:
        st.info("No investment signals generated yet. Please wait for the daily pipeline to complete.")
    else:
        symbols = signals_df['symbol'].unique()
        selected_symbol = st.selectbox("Select Cryptocurrency", symbols)
        
        # Filter data
        symbol_features = features_df[features_df['symbol'] == selected_symbol].copy()
        symbol_features['event_time'] = pd.to_datetime(symbol_features['event_time'])
        symbol_features.set_index('event_time', inplace=True)
        
        latest_signal = signals_df[signals_df['symbol'] == selected_symbol].iloc[0]
        latest_facts = symbol_features.iloc[-1]
        
        # --- Top KPIs ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${latest_facts['close']:,.2f}")
        col2.metric("AI Signal", latest_signal['signal'])
        col3.metric("RSI (14d)", f"{latest_facts['rsi_14']:.1f}")
        col4.metric("Confidence", f"{latest_signal['confidence'] * 100:.1f}%")
        
        st.markdown("---")
        
        # --- AI Recommendation Notes ---
        st.subheader("🤖 AI Analytics & Reasoning")
        
        rsi = latest_facts['rsi_14']
        macd = latest_facts['macd']
        macd_signal = latest_facts['macd_signal']
        price = latest_facts['close']
        ma7 = latest_facts['moving_avg_7d']
        
        notes = f"**Current Context for {selected_symbol}:**\n"
        notes += f"- The asset is currently trading at **${price:,.2f}**, which is **{'above' if price > ma7 else 'below'}** its 7-day moving average of ${ma7:,.2f}.\n"
        
        if pd.notna(rsi):
            if rsi > 70:
                notes += f"- **RSI** is at **{rsi:.1f}**, indicating the asset is strongly **overbought**. A price correction may be imminent.\n"
            elif rsi < 30:
                notes += f"- **RSI** is at **{rsi:.1f}**, indicating the asset is strongly **oversold**. This might be a buying opportunity.\n"
            else:
                notes += f"- **RSI** is neutral at **{rsi:.1f}**.\n"
                
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal:
                notes += f"- **MACD** ({macd:.2f}) has crossed **above** its signal line ({macd_signal:.2f}), suggesting **bullish** momentum.\n"
            else:
                notes += f"- **MACD** ({macd:.2f}) has crossed **below** its signal line ({macd_signal:.2f}), suggesting **bearish** momentum.\n"
                
        st.info(notes)
        
        st.markdown("---")
        
        # --- Plots ---
        st.subheader("Price Action & Moving Average")
        price_chart_data = symbol_features[['close', 'moving_avg_7d']].rename(columns={'close': 'Price', 'moving_avg_7d': '7D MA'})
        st.line_chart(price_chart_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Relative Strength Index (RSI)")
            if 'rsi_14' in symbol_features.columns:
                st.line_chart(symbol_features[['rsi_14']], color="#ffaa00")
            
        with col2:
            st.subheader("MACD vs Signal")
            if 'macd' in symbol_features.columns and 'macd_signal' in symbol_features.columns:
                macd_data = symbol_features[['macd', 'macd_signal']]
                st.line_chart(macd_data)
                
        # --- Raw Data ---
        with st.expander("View Raw Signals Data"):
            st.dataframe(signals_df[signals_df['symbol'] == selected_symbol])

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("System initializing. Please wait for the first data pipeline run to complete.")
