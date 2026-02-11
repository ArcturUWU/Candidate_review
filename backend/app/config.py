from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration."""

    database_url: str = Field(default="sqlite:///./reviewer.db", env="DATABASE_URL")
    lm_studio_url: str = Field(default="http://localhost:1234/v1/chat/completions", env="LM_STUDIO_URL")
    lm_model: str = Field(default="qwen/qwen3-8b", env="LM_MODEL")
    sandbox_code_url: str = Field(default="http://localhost:8001/run_code", env="SANDBOX_CODE_URL")
    sandbox_sql_url: str = Field(default="http://localhost:8002/run_sql", env="SANDBOX_SQL_URL")
    web_search_url: str = Field(default="http://localhost:8003/search", env="WEB_SEARCH_URL")
    allow_origins: str = Field(default="*")  # comma-separated origins

    class Config:
        env_file = ".env"


settings = Settings()
