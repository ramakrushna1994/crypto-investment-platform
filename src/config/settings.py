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

POSTGRES = PostgresConfig()

@dataclass(frozen=True)
class BinanceConfig:
    """Configuration for historical crypto data fetching via Binance API."""
    base_url: str = "https://api.binance.com/api/v3/klines"
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT")
    interval: str = "1d"
    limit: int = 1000

BINANCE = BinanceConfig()
