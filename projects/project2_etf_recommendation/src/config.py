"""Project 2: ETF 추천 시스템 설정"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    # OpenAI
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.3

    # 검색 설정
    top_k: int = 10
    rerank_top_n: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # 데이터 경로
    etf_data_path: str = "data/sample_etf_data.json"
    vector_store_path: str = "vector_store/etf_faiss"

    # Gradio
    server_port: int = 7862
    share: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"
