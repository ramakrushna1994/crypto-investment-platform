import pandas as pd
from sqlalchemy import create_engine, text
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

def table_exists(engine, table_name: str) -> bool:
    """Return True if the given table exists in the database."""
    try:
        with engine.connect() as conn:
            res = conn.execute(f"SELECT to_regclass('{table_name}')").scalar()
            return res is not None
    except Exception:
        return False


def table_has_column(engine, table_name: str, column_name: str) -> bool:
    """Return True if the table contains the given column."""
    try:
        schema, table = table_name.split(".", 1) if "." in table_name else ("public", table_name)
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table
                          AND column_name = :column
                    )
                """),
                {"schema": schema, "table": table, "column": column_name},
            ).scalar()
            return bool(res)
    except Exception:
        return False


def fetch_and_filter_data(engine, signals_table, features_table):
    # if either source or features table is missing, treat as empty
    if not table_exists(engine, signals_table):
        logger.warning(f"Signals table {signals_table} does not exist; skipping")
        return pd.DataFrame()
    if not table_exists(engine, features_table):
        logger.warning(f"Features table {features_table} does not exist; skipping")
        return pd.DataFrame()

    # signals_table and features_table should already include schema qualification
    has_asset_name = table_has_column(engine, features_table, "asset_name")
    asset_select = (
        "COALESCE(NULLIF(TRIM(f.asset_name), ''), s.symbol) AS asset_name,"
        if has_asset_name
        else "s.symbol AS asset_name,"
    )
    query = f"""
        SELECT s.*, {asset_select} f.close, f.rsi_14, f.volatility_7d 
        FROM (SELECT DISTINCT ON (symbol) * FROM {signals_table} ORDER BY symbol, trade_date DESC) s
        JOIN (SELECT DISTINCT ON (symbol) * FROM {features_table} ORDER BY symbol, event_time DESC) f
        ON s.symbol = f.symbol
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        logger.error(f"SQL error fetching data from {signals_table}/{features_table}: {e}")
        return pd.DataFrame()
    
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
    cols_to_keep = ['asset_name']
    
    if 'signal_1y' in df.columns and 'confidence_1y' in df.columns:
        cols_to_keep.extend(['signal_1y', 'confidence_1y'])
    if 'signal_5y' in df.columns and 'confidence_5y' in df.columns:
        cols_to_keep.extend(['signal_5y', 'confidence_5y'])
    if 'signal' in df.columns and 'confidence' in df.columns:
        cols_to_keep.extend(['signal', 'confidence'])
        
    cols_to_keep.extend(['close', 'rsi_14', 'volatility_7d'])
    
    return filtered[cols_to_keep] if not filtered.empty else pd.DataFrame()


def write_excel_report(engine, tabs, filepath):
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for tab_name, (sig_tbl, feat_tbl) in tabs.items():
            try:
                df = fetch_and_filter_data(engine, sig_tbl, feat_tbl)
                if not df.empty:
                    df.to_excel(writer, sheet_name=tab_name, index=False)
                else:
                    pd.DataFrame({'Message': ['No assets matched INVEST NOW/ACCUMULATE']}).to_excel(writer, sheet_name=tab_name, index=False)
            except Exception as e:
                logger.error(f"Error processing {tab_name}: {e}")
                pd.DataFrame({'Error': [str(e)]}).to_excel(writer, sheet_name=tab_name, index=False)

    # after writing, apply conditional formatting based on signal columns
    from openpyxl.styles import PatternFill
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl import load_workbook

    wb = load_workbook(filepath)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        # find signal columns used to trigger row highlight
        signal_col_letters = []
        confidence_col_letters = []
        for colcell in ws[1]:
            if colcell.value in {'signal', 'signal_1y', 'signal_5y'}:
                signal_col_letters.append(colcell.column_letter)
            if colcell.value in {'confidence', 'confidence_1y', 'confidence_5y'}:
                confidence_col_letters.append(colcell.column_letter)
        if signal_col_letters and ws.max_row >= 2:
            # apply fill to entire row when any signal column is INVEST NOW
            fill = PatternFill(start_color='00FF00', end_color='00FF00', fill_type='solid')
            formula_parts = [f'UPPER(${col}2)="INVEST NOW"' for col in signal_col_letters]
            formula = f"OR({','.join(formula_parts)})"
            # apply rule across all columns A..last
            last_col = ws.max_column
            last_letter = ws.cell(row=1, column=last_col).column_letter
            ws.conditional_formatting.add(
                f"A2:{last_letter}{ws.max_row}",
                FormulaRule(formula=[formula], stopIfTrue=True, fill=fill),
            )
        if confidence_col_letters and ws.max_row >= 2:
            # display confidence values as percentages (e.g., 0.58 -> 58.00%)
            for col in confidence_col_letters:
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws[f"{col}{row_idx}"]
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = "0.00%"
    wb.save(filepath)

def send_opportunity_email():
    engine = get_engine()
    
    # Definition of tabs - use medallion schema names, not public
    tabs = {
        "Nifty 50": ("gold.nifty50_investment_signals", "silver.nifty50_features_daily"),
        "Nifty Mid Cap": ("gold.nifty_midcap_investment_signals", "silver.nifty_midcap_features_daily"),
        "Nifty Small Cap": ("gold.nifty_smallcap_investment_signals", "silver.nifty_smallcap_features_daily"),
        "Mutual Funds": ("gold.mutual_funds_investment_signals", "silver.mutual_funds_features_daily"),
        "Crypto": ("gold.crypto_investment_signals", "silver.crypto_features_daily")
    }
    
    filepath = "/opt/airflow/files/investment_opportunities.xlsx"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    try:
        write_excel_report(engine, tabs, filepath)
    except PermissionError:
        fallback_filepath = f"/opt/airflow/files/investment_opportunities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        logger.warning(f"Permission denied for {filepath}; retrying with {fallback_filepath}")
        try:
            write_excel_report(engine, tabs, fallback_filepath)
            filepath = fallback_filepath
        except Exception as e:
            logger.error(f"Failed to create Excel file after fallback: {e}")
            engine.dispose()
            return
    except Exception as e:
        logger.error(f"Failed to create Excel file: {e}")
        engine.dispose()
        return
    
    # Read SMTP and email configuration
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SENDER_EMAIL", "noreply@investmentplatform.local")
    receiver = os.environ.get("RECEIVER_EMAIL")

    # allow Airflow Variables to override or supply missing values
    try:
        from airflow.models import Variable
        if not smtp_user:
            smtp_user = Variable.get("SMTP_USER", default_var="")
        if not smtp_pass:
            smtp_pass = Variable.get("SMTP_PASSWORD", default_var="")
        if not receiver:
            receiver = Variable.get("RECEIVER_EMAIL", default_var="user@example.com")
        if not sender:
            sender = Variable.get("SENDER_EMAIL", default_var="noreply@investmentplatform.local")
    except ImportError:
        # running outside Airflow environment, ignore
        pass

    receiver = receiver or "user@example.com"
    recipients = [
        addr.strip()
        for addr in receiver.replace(";", ",").split(",")
        if addr.strip()
    ]
    if not recipients:
        recipients = ["user@example.com"]
    receiver_display = ", ".join(recipients)

    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not provided in env or Airflow Variables. Skipping email dispatch, but file generated.")
        return
        
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver_display
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
        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        logger.info(f"Email sent successfully to {receiver_display}.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

if __name__ == '__main__':
    send_opportunity_email()
