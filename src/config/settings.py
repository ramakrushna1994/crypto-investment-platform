import os
from dataclasses import dataclass

@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = 5432
    db: str = os.getenv("POSTGRES_DB", "crypto")
    user: str = os.getenv("POSTGRES_USER", "crypto")
    password: str = os.getenv("POSTGRES_PASSWORD", "crypto")

    @property
    def jdbc_url(self):
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.db}"
    

    @property
    def dsn(self):
        return f"dbname={self.db} user={self.user} password={self.password} host={self.host}"

POSTGRES = PostgresConfig()

@dataclass(frozen=True)
class BinanceConfig:
    base_url: str = "https://api.binance.com/api/v3/klines"
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT")
    interval: str = "1d"
    limit: int = 1000

BINANCE = BinanceConfig()
