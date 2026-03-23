"""
Configuration management for Company Analysis Agent.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class OpenAISettings(BaseSettings):
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    org_id: Optional[str] = Field(default=None, alias="OPENAI_ORG_ID")
    llm_model: str = Field(default="gpt-4o", alias="OPENAI_LLM_MODEL")
    vision_model: str = Field(default="gpt-4o", alias="OPENAI_VISION_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL"
    )

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class Neo4jSettings(BaseSettings):
    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: str = Field(default="password", alias="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class AnthropicSettings(BaseSettings):
    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class PathSettings(BaseSettings):
    data_dir: Path = Field(default=_PROJECT_ROOT / "data", alias="DATA_DIR")
    output_dir: Path = Field(default=_PROJECT_ROOT / "output", alias="OUTPUT_DIR")
    parsed_docs_dir: Path = Field(
        default=_PROJECT_ROOT / "output/parsed", alias="PARSED_DOCS_DIR"
    )
    graphs_dir: Path = Field(
        default=_PROJECT_ROOT / "output/graphs", alias="GRAPHS_DIR"
    )
    reports_dir: Path = Field(
        default=_PROJECT_ROOT / "output/reports", alias="REPORTS_DIR"
    )
    faiss_index_path: Path = Field(
        default=_PROJECT_ROOT / "output/faiss_index", alias="FAISS_INDEX_PATH"
    )
    cache_dir: Path = Field(
        default=_PROJECT_ROOT / "output/cache", alias="CACHE_DIR"
    )

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}

    def create_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)


class VectorStoreSettings(BaseSettings):
    top_k: int = Field(default=10, alias="VECTOR_STORE_TOP_K")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class CLIPSettings(BaseSettings):
    model_name: str = Field(default="ViT-L/14", alias="CLIP_MODEL_NAME")
    device: str = Field(default="cpu", alias="CLIP_DEVICE")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class DocumentParsingSettings(BaseSettings):
    unstructured_api_key: str = Field(
        default="", alias="UNSTRUCTURED_API_KEY"
    )
    unstructured_api_url: str = Field(
        default="https://api.unstructured.io/general/v0/general",
        alias="UNSTRUCTURED_API_URL",
    )
    use_unstructured_api: bool = Field(default=False, alias="USE_UNSTRUCTURED_API")
    tesseract_path: str = Field(default="", alias="TESSERACT_PATH")
    pdf_image_dpi: int = Field(default=200, alias="PDF_IMAGE_DPI")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class GraphRAGSettings(BaseSettings):
    community_algorithm: str = Field(
        default="leiden", alias="GRAPH_COMMUNITY_ALGORITHM"
    )
    max_community_size: int = Field(default=10, alias="GRAPH_MAX_COMMUNITY_SIZE")
    traversal_depth: int = Field(default=3, alias="GRAPH_TRAVERSAL_DEPTH")
    local_search_top_k: int = Field(default=5, alias="GRAPH_LOCAL_SEARCH_TOP_K")
    global_search_top_k: int = Field(default=3, alias="GRAPH_GLOBAL_SEARCH_TOP_K")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class LLMSettings(BaseSettings):
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    timeout: int = Field(default=60, alias="LLM_TIMEOUT")
    entity_extraction_max_tokens: int = Field(
        default=2000, alias="ENTITY_EXTRACTION_MAX_TOKENS"
    )
    entity_types: list[str] = Field(
        default=[
            "Company",
            "Executive",
            "Product",
            "Subsidiary",
            "FinancialMetric",
            "Market",
            "Risk",
        ]
    )

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}

    @field_validator("entity_types", mode="before")
    @classmethod
    def parse_entity_types(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",")]
        return v


class AppSettings(BaseSettings):
    env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=7860, alias="APP_PORT")
    enable_cache: bool = Field(default=True, alias="ENABLE_CACHE")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")
    gradio_share: bool = Field(default=False, alias="GRADIO_SHARE")
    gradio_auth: Optional[str] = Field(default=None, alias="GRADIO_AUTH")

    model_config = {"populate_by_name": True, "env_file": str(_PROJECT_ROOT / ".env")}


class Config:
    """
    Central configuration object.
    Access via: from src.config import config
    """

    def __init__(self):
        self.openai = OpenAISettings()
        self.neo4j = Neo4jSettings()
        self.anthropic = AnthropicSettings()
        self.paths = PathSettings()
        self.vector_store = VectorStoreSettings()
        self.clip = CLIPSettings()
        self.document_parsing = DocumentParsingSettings()
        self.graph_rag = GraphRAGSettings()
        self.llm = LLMSettings()
        self.app = AppSettings()

        # Set OpenAI API key in environment for libraries that read it directly
        if self.openai.api_key:
            os.environ["OPENAI_API_KEY"] = self.openai.api_key

        if self.anthropic.api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic.api_key

        # Configure Tesseract path if provided (Windows)
        if self.document_parsing.tesseract_path:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = (
                    self.document_parsing.tesseract_path
                )
            except ImportError:
                pass

    def ensure_directories(self) -> None:
        """Create all output directories."""
        self.paths.create_directories()

    @property
    def is_development(self) -> bool:
        return self.app.env == "development"

    @property
    def is_production(self) -> bool:
        return self.app.env == "production"

    def __repr__(self) -> str:
        return (
            f"Config(env={self.app.env}, "
            f"openai_model={self.openai.llm_model}, "
            f"neo4j_uri={self.neo4j.uri})"
        )


# Singleton config instance
config = Config()
