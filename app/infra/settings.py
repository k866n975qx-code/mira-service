from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # allow this since it's already in your .env
    mira_env: str = "dev"

    # required
    mira_db_url: str

    # --- Lunch Money ---
    lunchmoney_api_key: str | None = None
    lunchmoney_base_url: str = "https://api.lunchmoney.app/v1"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # don't crash on other future vars
    }


settings = Settings()