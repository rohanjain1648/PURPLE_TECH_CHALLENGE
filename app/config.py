"""
Central configuration loaded from environment variables.
All tuneable thresholds live here so operators can tweak without redeploying.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite+aiosqlite:///./store_intelligence.db"

    # Redis (optional – gracefully skipped if not set)
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False

    # API
    api_title: str = "Store Intelligence API"
    api_version: str = "1.0.0"
    debug: bool = False

    # Ingest limits
    max_events_per_batch: int = 500

    # Anomaly detection thresholds
    queue_spike_threshold: int = 5          # queue_depth above this is WARN
    queue_critical_threshold: int = 10      # queue_depth above this is CRITICAL
    queue_spike_duration_s: int = 120       # must persist this many seconds
    conversion_drop_pct: float = 0.20      # 20 % below 7-day avg triggers WARN
    dead_zone_minutes: int = 30            # no activity in zone triggers INFO
    stale_feed_minutes: int = 10           # no events from store triggers WARN

    # POS correlation window
    pos_correlation_window_s: int = 300    # 5-minute window before POS transaction

    # Session Re-ID
    reentry_max_gap_s: int = 300           # 5-minute gap max for re-entry detection

    # ZONE_DWELL emit interval
    dwell_emit_interval_s: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
