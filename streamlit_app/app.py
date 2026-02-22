# streamlit_app/app.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from src.config.settings import POSTGRES
from src.llm.ollama_analyst import get_ollama_analysis

st.set_page_config(page_title="AI Investment Platform", layout="wide")
st.title("📈 AI Investment Engine (Crypto & Nifty 50)")

@st.cache_resource
def get_db_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}@{POSTGRES.host}/{POSTGRES.db}",
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=1,
    )

engine = get_db_engine()

# ── Cached data loaders ───────────────────────────────────────────────────────
# These run once and are reused across tab switches (TTL = 10 min)

@st.cache_data(ttl=600, show_spinner=False)
def load_latest_signals(_engine, signals_table: str) -> pd.DataFrame:
    """Latest signal per symbol — one row per symbol, very fast."""
    return pd.read_sql(f"""
        SELECT DISTINCT ON (symbol) *
        FROM {signals_table}
        ORDER BY symbol, trade_date DESC
    """, _engine)

@st.cache_data(ttl=600, show_spinner=False)
def load_latest_features(_engine, features_table: str) -> pd.DataFrame:
    """Latest feature row per symbol — only what the overview table needs."""
    return pd.read_sql(f"""
        SELECT DISTINCT ON (symbol) *
        FROM {features_table}
        ORDER BY symbol, event_time DESC
    """, _engine)

@st.cache_data(ttl=600, show_spinner="Loading historical data...")
def load_symbol_history(_engine, features_table: str, symbol: str) -> pd.DataFrame:
    """Full history for ONE symbol — loaded on demand when user clicks a row."""
    return pd.read_sql(f"""
        SELECT * FROM {features_table}
        WHERE symbol = %(sym)s
        ORDER BY event_time ASC
    """, _engine, params={"sym": symbol})

def render_asset_dashboard(signals_table, features_table, asset_label, currency_prefix="$", currency_suffix=""):
    latest_signals = load_latest_signals(engine, signals_table)
    latest_features = load_latest_features(engine, features_table)


    if latest_signals.empty or latest_features.empty:
        st.info(f"No {asset_label} investment signals generated yet. Please wait for the daily pipeline to complete.")
        return

    merge_cols = ['symbol', 'close', 'rsi_14', 'volatility_7d']
    has_asset_name = 'asset_name' in latest_features.columns
    if has_asset_name:
        merge_cols.insert(1, 'asset_name')
        
    # --- Horizon selector ---
    horizon = st.radio(
        "📅 Investment Horizon",
        options=["1-Year Outlook", "5-Year Outlook"],
        horizontal=True,
        help="Switch between 1-Year (≥10% growth) and 5-Year (≥50% growth) ML models",
        key=f"horizon_{asset_label}"
    )
    use_5y = horizon == "5-Year Outlook"
    sig_col = 'signal_5y' if use_5y else 'signal_1y'
    conf_col = 'confidence_5y' if use_5y else 'confidence_1y'

    # Fallback: if new horizon columns don't exist yet (old table schema), use 'signal'/'confidence'
    if sig_col not in latest_signals.columns:
        sig_col, conf_col = 'signal', 'confidence'

    EMOJI_MAP = {
        'INVEST NOW':  '🚀 INVEST NOW',
        'ACCUMULATE':  '✅ ACCUMULATE',
        'STRONG HOLD': '💎 STRONG HOLD',
        'MONITOR':     '👀 MONITOR',
        'WAIT':        '⏸ WAIT',
        'AVOID':       '🔴 AVOID',
        # Legacy fallbacks
        'STRONG BUY':  '🚀 STRONG BUY',
        'BUY':         '✅ BUY',
        'HOLD':        '⏸ HOLD',
        'SELL':        '⚠️ SELL',
        'STRONG SELL': '🔴 STRONG SELL',
    }
    SORT_ORDER = {
        'INVEST NOW': 0, 'STRONG HOLD': 0, 'ACCUMULATE': 1,
        'MONITOR': 2, 'WAIT': 3, 'AVOID': 4,
        'STRONG BUY': 0, 'BUY': 1, 'HOLD': 2, 'SELL': 3, 'STRONG SELL': 4,
    }

    def format_signal(row):
        sig = row.get(sig_col, row.get('signal', 'MONITOR'))
        prob = float(row.get(conf_col, row.get('confidence', 0))) * 100
        label = EMOJI_MAP.get(sig, sig)
        return f"{label}  ({prob:.0f}%)"

    table_view = pd.merge(latest_signals, latest_features[merge_cols], on='symbol', how='inner')
    table_view['ML Outlook'] = table_view.apply(format_signal, axis=1)
    table_view['Price'] = table_view['close'].apply(lambda x: f"{currency_prefix}{x:,.2f}{currency_suffix}")
    table_view['RSI (14d)'] = table_view['rsi_14'].round(1)
    table_view['Volatility'] = table_view['volatility_7d'].round(2)

    active_signal = table_view.get(sig_col, table_view.get('signal', pd.Series(dtype=str)))
    table_view['_sort_key'] = active_signal.map(SORT_ORDER).fillna(5)
    table_view.sort_values(by=['_sort_key', conf_col if conf_col in table_view.columns else 'confidence'],
                           ascending=[True, False], inplace=True)
    table_view.reset_index(drop=True, inplace=True)

    if has_asset_name:
        table_view.rename(columns={'asset_name': 'Asset Name', 'symbol': 'Symbol'}, inplace=True)
        display_df = table_view[['Symbol', 'Asset Name', 'ML Outlook', 'Price', 'RSI (14d)', 'Volatility']]
    else:
        table_view.rename(columns={'symbol': 'Symbol'}, inplace=True)
        display_df = table_view[['Symbol', 'ML Outlook', 'Price', 'RSI (14d)', 'Volatility']]
    
    st.markdown(f"### 📊 Market Overview ({len(display_df)} assets)")
    
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"table_{asset_label}"
    )
    
    selected_rows = event.selection.rows
    if not selected_rows:
        st.info("👆 Select a row in the table above to view deep-dive AI analysis.")
        return
        
    selected_index = selected_rows[0]
    selected_symbol = display_df.iloc[selected_index]['Symbol']
    
    st.markdown("---")
    
    # Show the long descriptive asset name for the Deep Dive title if available
    display_name = display_df.iloc[selected_index]['Asset Name'] if has_asset_name else selected_symbol
    st.markdown(f"### 🔍 Deep Dive: {display_name} ({selected_symbol})")
    
    symbol_features = load_symbol_history(engine, features_table, selected_symbol)
    symbol_features['event_time'] = pd.to_datetime(symbol_features['event_time'])
    symbol_features.set_index('event_time', inplace=True)
    
    latest_signal = latest_signals[latest_signals['symbol'] == selected_symbol].iloc[0]
    latest_facts = symbol_features.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{currency_prefix}{latest_facts['close']:,.2f}{currency_suffix}")
    col2.metric("AI Signal", latest_signal['signal'])
    col3.metric("RSI (14d)", f"{latest_facts['rsi_14']:.1f}")
    col4.metric("AI Conviction", f"{latest_signal['confidence'] * 100:.1f}%")
    
    st.markdown("---")
    
    st.subheader("Historical Growth Performance")
    
    # Helper to calculate historical returns based on the 10-year features dataframe
    def get_historical_return(years_back):
        target_date = latest_facts.name - pd.DateOffset(years=years_back)
        # Find the closest available trading day before or exactly on the target date
        past_data = symbol_features[symbol_features.index <= target_date]
        if past_data.empty:
            return "N/A"
        past_price = past_data.iloc[-1]['close']
        current_price = latest_facts['close']
        return ((current_price - past_price) / past_price) * 100

    r1y = get_historical_return(1)
    r3y = get_historical_return(3)
    r5y = get_historical_return(5)
    r10y = get_historical_return(10)
    
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    
    def format_return(val):
        return f"+{val:.2f}%" if isinstance(val, float) and val > 0 else (f"{val:.2f}%" if isinstance(val, float) else val)

    g_col1.metric("1-Year Growth", format_return(r1y))
    g_col2.metric("3-Year Growth", format_return(r3y))
    g_col3.metric("5-Year Growth", format_return(r5y))
    g_col4.metric("10-Year Growth", format_return(r10y))
    
    st.markdown("---")
    
    st.subheader("AI Analytics & Reasoning (Powered by Local Ollama)")
    
    rsi = latest_facts.get('rsi_14', pd.NA)
    macd = latest_facts.get('macd', pd.NA)
    macd_signal = latest_facts.get('macd_signal', pd.NA)
    price = latest_facts['close']
    ma7 = latest_facts.get('moving_avg_7d', pd.NA)
    ema20 = latest_facts.get('ema_20', pd.NA)
    bb_upper = latest_facts.get('bb_upper', pd.NA)
    bb_upper = latest_facts.get('bb_upper', pd.NA)
    bb_lower = latest_facts.get('bb_lower', pd.NA)
    volatility = latest_facts.get('volatility_7d', 0.0)
    sma_50 = latest_facts.get('sma_50', pd.NA)
    sma_200 = latest_facts.get('sma_200', pd.NA)
    atr_14 = latest_facts.get('atr_14', pd.NA)
    stoch_k = latest_facts.get('stoch_k', pd.NA)
    stoch_d = latest_facts.get('stoch_d', pd.NA)

    # try ollama first, fallback to rule-based if it fails
    ai_notes = None
    if all(pd.notna(v) for v in [rsi, macd, macd_signal, ema20, ma7, bb_upper, bb_lower]):
        with st.spinner("Analyzing market data..."):
            ai_notes = get_ollama_analysis(
                symbol=selected_symbol,
                asset_type=asset_label,
                price=float(price),
                currency_prefix=currency_prefix,
                currency_suffix=currency_suffix,
                signal=str(latest_signal['signal']),
                confidence=float(latest_signal['confidence']),
                rsi=float(rsi),
                macd=float(macd),
                macd_signal_val=float(macd_signal),
                ema_20=float(ema20),
                ma_7=float(ma7),
                bb_upper=float(bb_upper),
                bb_lower=float(bb_lower),
                volatility=float(volatility),
                sma_50=float(sma_50) if pd.notna(sma_50) else 0.0,
                sma_200=float(sma_200) if pd.notna(sma_200) else 0.0,
                atr_14=float(atr_14) if pd.notna(atr_14) else 0.0,
                stoch_k=float(stoch_k) if pd.notna(stoch_k) else 0.0,
                stoch_d=float(stoch_d) if pd.notna(stoch_d) else 0.0,
            )

    if ai_notes:
        st.info(ai_notes)
        st.caption("⚡ Analysis generated by llama3.2")
    else:
        # Rule-based fallback (used when Ollama is unreachable)
        notes = f"**Current Context for {selected_symbol}:**\n"
        
        # 1. Trend (EMA)
        if price > ema20:
            notes += f"- **Trend**: Bullish (Price {currency_prefix}{price:,.2f}{currency_suffix} is above the 20-day EMA of {currency_prefix}{ema20:,.2f}{currency_suffix}).\n"
        else:
            notes += f"- **Trend**: Bearish (Price {currency_prefix}{price:,.2f}{currency_suffix} is below the 20-day EMA {currency_prefix}{ema20:,.2f}{currency_suffix}).\n"
            
        # 2. Momentum (RSI)
        if rsi > 70:
            notes += f"- **Momentum**: Overbought (RSI is high at {rsi:.1f}). Expect potential pullback.\n"
        elif rsi < 30:
            notes += f"- **Momentum**: Oversold (RSI is low at {rsi:.1f}). Potential bounce impending.\n"
        else:
            notes += f"- **Momentum**: Neutral (RSI at {rsi:.1f}).\n"
            
        # 3. Volatility (Bollinger Bands)
        if price > bb_upper:
            notes += f"- **Volatility Breakout**: Price has broken above the upper Bollinger Band ({currency_prefix}{bb_upper:,.2f}{currency_suffix}). High momentum but prone to correction.\n"
        elif price < bb_lower:
            notes += f"- **Volatility Breakout**: Price has dropped below the lower Bollinger Band ({currency_prefix}{bb_lower:,.2f}{currency_suffix}). Heavily oversold territory.\n"
        
        notes += "\n*(Start the local ollama service for deeper AI analysis)*"
        st.info(notes)
    
    st.markdown("---")
    
    st.subheader("Price Action & Moving Averages")
    
    plot_cols = {'close': 'Price'}
    if 'moving_avg_7d' in symbol_features.columns:
        plot_cols['moving_avg_7d'] = '7D SMA'
    if 'ema_20' in symbol_features.columns:
        plot_cols['ema_20'] = '20D EMA'
    if 'bb_upper' in symbol_features.columns:
        plot_cols['bb_upper'] = 'Upper BB'
    if 'bb_lower' in symbol_features.columns:
        plot_cols['bb_lower'] = 'Lower BB'
    if 'sma_50' in symbol_features.columns:
        plot_cols['sma_50'] = '50D SMA'
    if 'sma_200' in symbol_features.columns:
        plot_cols['sma_200'] = '200D SMA'
        
    price_chart_data = symbol_features[list(plot_cols.keys())].rename(columns=plot_cols)
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
            
    with st.expander("View Raw Signals Data"):
        st.dataframe(latest_signals[latest_signals['symbol'] == selected_symbol])

    st.markdown("---")
    st.subheader(f"💬 Chat with AI about {selected_symbol}")
    st.info("Ask specific questions about the data above. The AI is restricted to zero-hallucination factual analysis.")

    chat_key = f"chat_history_{selected_symbol}_{asset_label}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input(f"Ask about {selected_symbol}'s technicals...", key=f"chat_{asset_label}"):
        st.session_state[chat_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data context..."):
                context_data = {
                    "price": price,
                    "rsi": rsi,
                    "macd": macd,
                    "ema_20": ema20,
                    "ma_7": ma7,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "volatility": volatility,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "atr_14": atr_14,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "ai_conviction": latest_signal['confidence']
                }
                
                from src.llm.ollama_analyst import chat_with_ollama
                response = chat_with_ollama(selected_symbol, context_data, user_input)
                st.markdown(response)
                st.session_state[chat_key].append({"role": "assistant", "content": response})

try:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Nifty 50", "Nifty Mid Cap", "Nifty Small Cap", "Indian Mutual Funds", "Crypto"])
    
    with tab1:
        render_asset_dashboard(
            signals_table="public.nifty50_investment_signals",
            features_table="public.nifty50_features_daily",
            asset_label="Nifty 50",
            currency_prefix="₹",
            currency_suffix=" INR"
        )
        
    with tab2:
        render_asset_dashboard(
            signals_table="public.nifty_midcap_investment_signals",
            features_table="public.nifty_midcap_features_daily",
            asset_label="Nifty Mid Cap",
            currency_prefix="₹",
            currency_suffix=" INR"
        )
        
    with tab3:
        render_asset_dashboard(
            signals_table="public.nifty_smallcap_investment_signals",
            features_table="public.nifty_smallcap_features_daily",
            asset_label="Nifty Small Cap",
            currency_prefix="₹",
            currency_suffix=" INR"
        )
        
    with tab4:
        render_asset_dashboard(
            signals_table="public.mutual_funds_investment_signals",
            features_table="public.mutual_funds_features_daily",
            asset_label="Indian Mutual Funds",
            currency_prefix="₹",
            currency_suffix=" INR"
        )
        
    with tab5:
        render_asset_dashboard(
            signals_table="public.crypto_investment_signals",
            features_table="public.crypto_features_daily",
            asset_label="Cryptocurrency",
            currency_prefix="$"
        )

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    st.info("System initializing. Please wait for the first data pipeline run to complete.")
