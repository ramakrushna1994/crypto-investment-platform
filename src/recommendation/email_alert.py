import pandas as pd
from sqlalchemy import create_engine
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import os
import logging
from src.config.settings import POSTGRES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES.user}:{POSTGRES.password}@{POSTGRES.host}/{POSTGRES.db}"
    )

def fetch_and_filter_data(engine, signals_table, features_table):
    query = f"""
        SELECT s.*, f.close, f.rsi_14, f.volatility_7d 
        FROM (SELECT DISTINCT ON (symbol) * FROM {signals_table} ORDER BY symbol, trade_date DESC) s
        JOIN (SELECT DISTINCT ON (symbol) * FROM {features_table} ORDER BY symbol, event_time DESC) f
        ON s.symbol = f.symbol
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return pd.DataFrame()

    target_signals = ['INVEST NOW', 'ACCUMULATE', 'STRONG BUY', 'BUY']
    mask = pd.Series(False, index=df.index)
    
    if 'signal_5y' in df.columns:
        mask = mask | df['signal_5y'].isin(target_signals)
    if 'signal_1y' in df.columns:
        mask = mask | df['signal_1y'].isin(target_signals)
    if 'signal' in df.columns:
        mask = mask | df['signal'].isin(target_signals)
        
    filtered = df[mask].copy()
    
    # Organize columns
    cols_to_keep = ['symbol']
    
    if 'signal_1y' in df.columns and 'confidence_1y' in df.columns:
        cols_to_keep.extend(['signal_1y', 'confidence_1y'])
    if 'signal_5y' in df.columns and 'confidence_5y' in df.columns:
        cols_to_keep.extend(['signal_5y', 'confidence_5y'])
    if 'signal' in df.columns and 'confidence' in df.columns:
        cols_to_keep.extend(['signal', 'confidence'])
        
    cols_to_keep.extend(['close', 'rsi_14', 'volatility_7d'])
    
    return filtered[cols_to_keep] if not filtered.empty else pd.DataFrame()

def send_opportunity_email():
    engine = get_engine()
    
    # Definition of tabs
    tabs = {
        "Nifty 50": ("public.nifty50_investment_signals", "public.nifty50_features_daily"),
        "Nifty Mid Cap": ("public.nifty_midcap_investment_signals", "public.nifty_midcap_features_daily"),
        "Nifty Small Cap": ("public.nifty_smallcap_investment_signals", "public.nifty_smallcap_features_daily"),
        "Mutual Funds": ("public.mutual_funds_investment_signals", "public.mutual_funds_features_daily"),
        "Crypto": ("public.crypto_investment_signals", "public.crypto_features_daily")
    }
    
    filepath = "/opt/airflow/files/investment_opportunities.xlsx"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    has_data = False
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for tab_name, (sig_tbl, feat_tbl) in tabs.items():
                try:
                    df = fetch_and_filter_data(engine, sig_tbl, feat_tbl)
                    if not df.empty:
                        df.to_excel(writer, sheet_name=tab_name, index=False)
                        has_data = True
                    else:
                        pd.DataFrame({'Message': ['No assets matched INVEST NOW/ACCUMULATE']}).to_excel(writer, sheet_name=tab_name, index=False)
                except Exception as e:
                    logger.error(f"Error processing {tab_name}: {e}")
                    pd.DataFrame({'Error': [str(e)]}).to_excel(writer, sheet_name=tab_name, index=False)
    except Exception as e:
        logger.error(f"Failed to create Excel file: {e}")
        engine.dispose()
        return
                
    engine.dispose()
    logger.info(f"Excel file generated successfully at {filepath}")
    
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SENDER_EMAIL", "noreply@investmentplatform.local")
    receiver = os.environ.get("RECEIVER_EMAIL", "user@example.com")
    
    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not provided in env. Skipping email dispatch, but file generated.")
        return
        
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = "Daily AI Investment Opportunities | Crypto & Nifty Market Signals"
    
    body = """Attached is the latest daily report for assets marked 'INVEST NOW' or 'ACCUMULATE'.

Disclaimer: This is not investment advice and is created for educational purposes only. Please consult a qualified financial advisor before making any investment decisions. Use this information at your own risk.

- AI Investment Engine"""

    msg.attach(MIMEText(body, 'plain'))
    
    with open(filepath, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename=investment_opportunities.xlsx")
    msg.attach(part)
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        logger.info(f"Email sent successfully to {receiver}.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

if __name__ == '__main__':
    send_opportunity_email()
