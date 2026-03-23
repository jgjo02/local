"""
Configuration module for AI Consulting Chatbot.

Uses pydantic-settings for type-safe environment variable loading
with validation and defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key",
    )
    org_id: Optional[str] = Field(
        default=None,
        alias="OPENAI_ORG_ID",
        description="OpenAI organization ID",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if v and not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class ModelSettings(BaseSettings):
    """LLM and embedding model configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chat_model: str = Field(
        default="gpt-4o-mini",
        alias="CHAT_MODEL",
        description="OpenAI chat model name",
    )
    temperature: float = Field(
        default=0.1,
        alias="CHAT_MODEL_TEMPERATURE",
        ge=0.0,
        le=2.0,
        description="Model temperature (0=deterministic, 2=creative)",
    )
    max_tokens: int = Field(
        default=2048,
        alias="CHAT_MODEL_MAX_TOKENS",
        gt=0,
        le=16384,
        description="Maximum tokens in response",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="EMBEDDING_MODEL",
        description="OpenAI embedding model name",
    )


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    store_type: Literal["chroma", "faiss"] = Field(
        default="chroma",
        alias="VECTOR_STORE_TYPE",
        description="Vector store backend",
    )
    persist_dir: Path = Field(
        default=PROJECT_ROOT / "chroma_db",
        alias="VECTOR_STORE_PERSIST_DIR",
        description="Directory for persisting vector store",
    )
    collection_name: str = Field(
        default="housing_faq",
        alias="VECTOR_STORE_COLLECTION_NAME",
        description="Collection/index name in the vector store",
    )

    @field_validator("persist_dir", mode="before")
    @classmethod
    def resolve_persist_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


class RetrieverSettings(BaseSettings):
    """Retriever configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = Field(
        default="mmr",
        alias="RETRIEVER_SEARCH_TYPE",
        description="Retrieval strategy",
    )
    k: int = Field(
        default=5,
        alias="RETRIEVER_K",
        gt=0,
        le=50,
        description="Number of documents to retrieve",
    )
    score_threshold: float = Field(
        default=0.7,
        alias="RETRIEVER_SCORE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for threshold retrieval",
    )
    fetch_k: int = Field(
        default=20,
        alias="RETRIEVER_FETCH_K",
        gt=0,
        description="Number of docs fetched before MMR reranking",
    )


class ChunkingSettings(BaseSettings):
    """Text splitting / chunking configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chunk_size: int = Field(
        default=500,
        alias="CHUNK_SIZE",
        gt=0,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        alias="CHUNK_OVERLAP",
        ge=0,
        description="Overlap between consecutive chunks",
    )

    @model_validator(mode="after")
    def validate_overlap_less_than_chunk(self) -> "ChunkingSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class GradioSettings(BaseSettings):
    """Gradio UI configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    server_port: int = Field(
        default=7860,
        alias="GRADIO_SERVER_PORT",
        gt=1024,
        lt=65536,
        description="Port for Gradio server",
    )
    server_name: str = Field(
        default="0.0.0.0",
        alias="GRADIO_SERVER_NAME",
        description="Bind address for Gradio server",
    )
    share: bool = Field(
        default=False,
        alias="GRADIO_SHARE",
        description="Create a public share link via Gradio tunnel",
    )
    theme: str = Field(
        default="soft",
        alias="GRADIO_THEME",
        description="Gradio theme name",
    )
    auth_username: Optional[str] = Field(
        default=None,
        alias="GRADIO_AUTH_USERNAME",
        description="Basic auth username (optional)",
    )
    auth_password: Optional[str] = Field(
        default=None,
        alias="GRADIO_AUTH_PASSWORD",
        description="Basic auth password (optional)",
    )

    @property
    def auth(self) -> Optional[tuple[str, str]]:
        """Return (username, password) tuple if both are set, else None."""
        if self.auth_username and self.auth_password:
            return (self.auth_username, self.auth_password)
        return None


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="주택청약 AI 상담 챗봇",
        alias="APP_NAME",
        description="Application display name",
    )
    app_description: str = Field(
        default="주택청약 관련 궁금한 사항을 AI가 안내해 드립니다.",
        alias="APP_DESCRIPTION",
        description="Application description shown in UI",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Root logger level",
    )
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        alias="DATA_DIR",
        description="Directory containing FAQ / document data",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def resolve_data_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


class Config:
    """
    Aggregated configuration object that composes all sub-settings.

    Usage:
        from src.config import Config
        cfg = Config()
        print(cfg.model.chat_model)
        print(cfg.vector_store.persist_dir)
    """

    def __init__(self) -> None:
        self.openai = OpenAISettings()
        self.model = ModelSettings()
        self.vector_store = VectorStoreSettings()
        self.retriever = RetrieverSettings()
        self.chunking = ChunkingSettings()
        self.gradio = GradioSettings()
        self.app = AppSettings()

        self._setup_logging()
        self._ensure_directories()

    def _setup_logging(self) -> None:
        """Configure root logger based on LOG_LEVEL setting."""
        log_level = getattr(logging, self.app.log_level, logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Suppress overly verbose third-party loggers
        for noisy in ("httpx", "httpcore", "openai", "chromadb"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        logger.debug("Logging configured at level %s", self.app.log_level)

    def _ensure_directories(self) -> None:
        """Create required directories if they do not exist."""
        dirs = [
            self.vector_store.persist_dir,
            self.app.data_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", d)

    def validate_openai_key(self) -> bool:
        """Return True if an OpenAI API key is configured."""
        return bool(self.openai.api_key)

    def __repr__(self) -> str:
        return (
            f"Config("
            f"chat_model={self.model.chat_model!r}, "
            f"embedding={self.model.embedding_model!r}, "
            f"vector_store={self.vector_store.store_type!r}, "
            f"retriever={self.retriever.search_type!r}(k={self.retriever.k}))"
        )


# Convenience singleton – import and reuse across modules
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Return a module-level singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
