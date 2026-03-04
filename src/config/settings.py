"""
Core configuration settings for the investment platform.
Handles environment variables and connection strings for databases and external APIs.
"""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = 5432
    db: str = os.getenv("POSTGRES_DB", "ai_quant")
    user: str = os.getenv("POSTGRES_USER", "ai_quant")
    password: str = os.getenv("POSTGRES_PASSWORD", "ai_quant")

    @property
    def jdbc_url(self):
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.db}"
    

    @property
    def dsn(self):
        return f"dbname={self.db} user={self.user} password={self.password} host={self.host}"

POSTGRES = PostgresConfig()

@dataclass(frozen=True)
class BinanceConfig:
    """Configuration for historical crypto data fetching via Binance API."""
    base_url: str = "https://api.binance.com/api/v3/klines"
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT")
    interval: str = "1d"
    limit: int = 1000

BINANCE = BinanceConfig()

ALLOWED_TABLES = {
    # Crypto
    "bronze.crypto_price_raw",
    "silver.crypto_features_daily",
    "gold.crypto_investment_signals",
    
    # Nifty 50
    "bronze.nifty50_price_raw",
    "silver.nifty50_features_daily",
    "gold.nifty50_investment_signals",
    
    # Nifty Midcap
    "bronze.nifty_midcap_price_raw",
    "silver.nifty_midcap_features_daily",
    "gold.nifty_midcap_investment_signals",
    
    # Nifty Smallcap
    "bronze.nifty_smallcap_price_raw",
    "silver.nifty_smallcap_features_daily",
    "gold.nifty_smallcap_investment_signals",
    
    # Mutual Funds
    "bronze.mutual_funds_price_raw",
    "silver.mutual_funds_features_daily",
    "gold.mutual_funds_investment_signals",
}

def validate_table_name(table_name: str) -> str:
    """Validate that the table name is in the allowed whitelist to prevent SQL injection."""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table name '{table_name}' is not in the allowed whitelist.")
    return table_name

