from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IPB Open Source Backend"
    default_area: str = "North Karelia"
    default_timeframe: str = "72h"
    auto_refresh: bool = False
    refresh_interval_seconds: int = 1800

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
