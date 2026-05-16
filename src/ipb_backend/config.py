from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IPB Open Source Backend"
    default_area: str = "North Karelia"
    default_timeframe: str = "72h"
    auto_refresh: bool = False
    refresh_interval_seconds: int = 1800
    nls_api_key: str = ""
    digiroad_api_key: str = ""
    statistics_finland_api_key: str = ""
    opencellid_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
