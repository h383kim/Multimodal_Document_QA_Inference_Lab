"""Runtime configuration for the inference lab."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend defaults
    default_backend: str = "mock"
    default_model: str = "mock-vlm"
    default_quantization: str = "fp16"
    transformers_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_image_edge: int = 1280

    # Filesystem layout (relative paths resolve against the repo root)
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    sample_invoice_subdir: str = "sample_invoices"

    # Observability
    log_level: str = "INFO"

    # CORS — empty list disables CORS middleware
    cors_origins: list[str] = Field(default_factory=list)

    @field_validator("data_dir", "results_dir", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def sample_invoice_dir(self) -> Path:
        return self.data_dir / self.sample_invoice_subdir

    @property
    def routing_log_path(self) -> Path:
        return self.results_dir / "routing_log.jsonl"

    @property
    def results_db_path(self) -> Path:
        return self.results_dir / "runs.sqlite"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
