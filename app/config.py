"""Runtime configuration for the inference lab."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMI_", env_file=".env", extra="ignore")

    default_backend: str = "mock"
    default_model: str = "mock-vlm"
    default_quantization: str = "fp16"

    transformers_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_image_edge: int = 1280


def get_settings() -> Settings:
    return Settings()
