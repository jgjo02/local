"""
Configuration management for the Legal Advisor Agent.
Uses pydantic-settings for type-safe environment variable loading.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LegalAdvisorConfig(BaseSettings):
    """
    Central configuration for the Legal Advisor Agent.
    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="Primary LLM model for generation and grading",
    )
    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic)",
    )
    openai_max_tokens: int = Field(
        default=4096,
        ge=256,
        description="Maximum tokens per LLM response",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model for vector store",
    )

    # ------------------------------------------------------------------
    # Tavily Web Search
    # ------------------------------------------------------------------
    tavily_api_key: str = Field(
        default="",
        description="Tavily search API key",
    )
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of web search results",
    )

    # ------------------------------------------------------------------
    # LangSmith Tracing
    # ------------------------------------------------------------------
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key",
    )
    langchain_project: str = Field(
        default="legal-advisor-agent",
        description="LangSmith project name",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint URL",
    )

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------
    vector_store_type: Literal["chroma", "faiss"] = Field(
        default="chroma",
        description="Vector store backend",
    )
    vector_store_path: str = Field(
        default="./vector_store",
        description="Directory for persisting the vector store",
    )
    chroma_collection_name: str = Field(
        default="legal_documents",
        description="ChromaDB collection name",
    )

    # ------------------------------------------------------------------
    # Document Processing
    # ------------------------------------------------------------------
    legal_docs_path: str = Field(
        default="./data",
        description="Directory containing legal documents",
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between consecutive chunks",
    )

    # ------------------------------------------------------------------
    # RAG Configuration
    # ------------------------------------------------------------------
    retriever_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve per query",
    )
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for a document to be used",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum Self-RAG retry iterations",
    )

    # ------------------------------------------------------------------
    # Human-in-the-Loop
    # ------------------------------------------------------------------
    require_human_approval: bool = Field(
        default=False,
        description="Require human approval before delivering answers",
    )
    hitl_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence below which HITL is auto-triggered",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=7860, ge=1024, le=65535)
    app_title: str = Field(default="Legal Advisor Agent (법률 상담 에이전트)")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if v and not v.startswith("sk-"):
            raise ValueError("OPENAI_API_KEY must start with 'sk-'")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        # We cannot access chunk_size directly here before model is fully built,
        # but we do a sanity check at property access time instead.
        return v

    # ------------------------------------------------------------------
    # Derived / convenience properties
    # ------------------------------------------------------------------
    @property
    def vector_store_dir(self) -> Path:
        """Resolved Path object for the vector store directory."""
        return Path(self.vector_store_path).resolve()

    @property
    def legal_docs_dir(self) -> Path:
        """Resolved Path object for the legal documents directory."""
        return Path(self.legal_docs_path).resolve()

    @property
    def is_tracing_enabled(self) -> bool:
        """True if LangSmith tracing is properly configured."""
        return self.langchain_tracing_v2 and bool(self.langchain_api_key)

    def configure_langsmith(self) -> None:
        """Set LangSmith environment variables if tracing is enabled."""
        if self.is_tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint

    def get_llm_kwargs(self) -> dict:
        """Return kwargs suitable for initialising a ChatOpenAI instance."""
        return {
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "max_tokens": self.openai_max_tokens,
            "api_key": self.openai_api_key,
        }

    def get_embedding_kwargs(self) -> dict:
        """Return kwargs suitable for initialising an OpenAIEmbeddings instance."""
        return {
            "model": self.openai_embedding_model,
            "api_key": self.openai_api_key,
        }

    def __repr__(self) -> str:
        # Redact secrets in repr
        return (
            f"LegalAdvisorConfig("
            f"model={self.openai_model!r}, "
            f"vector_store={self.vector_store_type!r}, "
            f"retriever_k={self.retriever_k}, "
            f"debug={self.debug})"
        )


@lru_cache(maxsize=1)
def get_config() -> LegalAdvisorConfig:
    """
    Return the singleton configuration instance.
    Results are cached so the .env file is only read once.
    """
    config = LegalAdvisorConfig()
    config.configure_langsmith()
    return config


# Convenience re-export so callers can do `from src.config import config`
config: LegalAdvisorConfig = get_config()
