"""
Week 01 - Session 02: Python Clean Code Examples
AI 서비스 개발을 위한 파이썬 클린코드 실습
"""

# ============================================================
# 1. PEP 8 및 타입 힌팅 예시
# ============================================================

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, partial, reduce
from pathlib import Path
from typing import (
    Callable,
    Final,
    Generator,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# ============================================================
# 로깅 설정
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 2. 타입 정의
# ============================================================

# Literal 타입: 특정 값만 허용
ModelName = Literal["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
Role = Literal["user", "assistant", "system"]
VectorStoreType = Literal["faiss", "chroma", "pinecone"]

# 상수
MAX_RETRIES: Final = 3
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_MODEL: Final[ModelName] = "gpt-4o-mini"

# TypeVar
T = TypeVar("T")


# ============================================================
# 3. 데이터 모델 (dataclass)
# ============================================================


@dataclass
class LLMConfig:
    """LLM 설정 데이터 모델

    Attributes:
        model: 사용할 LLM 모델명
        temperature: 창의성 조절 (0=결정적, 2=매우 창의적)
        max_tokens: 최대 생성 토큰 수
        top_p: nucleus sampling 확률
        stream: 스트리밍 응답 여부
    """

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False

    def __post_init__(self) -> None:
        """초기화 후 유효성 검사"""
        if not 0 <= self.temperature <= 2:
            raise ValueError(
                f"temperature는 0-2 사이여야 합니다. 현재값: {self.temperature}"
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(
                f"top_p는 0-1 사이여야 합니다. 현재값: {self.top_p}"
            )
        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens는 양수여야 합니다. 현재값: {self.max_tokens}"
            )

    def to_dict(self) -> dict:
        """OpenAI API 파라미터 형식으로 변환"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }


@dataclass
class ChatMessage:
    """채팅 메시지 모델

    Attributes:
        role: 메시지 역할 (user/assistant/system)
        content: 메시지 내용
        timestamp: 생성 시각 (Unix timestamp)
    """

    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_openai_format(self) -> dict[str, str]:
        """OpenAI API 형식으로 변환"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatRequest:
    """채팅 요청 모델"""

    query: str
    session_id: str
    config: LLMConfig = field(default_factory=LLMConfig)
    history: list[ChatMessage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """채팅 응답 모델"""

    answer: str
    session_id: str
    sources: list[str] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0.0
    model: str = DEFAULT_MODEL

    @property
    def cost_estimate(self) -> float:
        """대략적인 비용 추정 (USD, gpt-4o-mini 기준)"""
        cost_per_1k_tokens = 0.00015  # $0.15 per 1M tokens
        return (self.tokens_used / 1000) * cost_per_1k_tokens


# ============================================================
# 4. Pydantic 모델 (유효성 검사 강화)
# ============================================================


class RAGConfig(BaseModel):
    """RAG 파이프라인 설정 (Pydantic 기반)"""

    chunk_size: int = Field(default=1000, ge=100, le=8000, description="청크 크기 (문자 수)")
    chunk_overlap: int = Field(default=200, ge=0, description="청크 겹침 크기")
    top_k: int = Field(default=4, ge=1, le=20, description="검색할 상위 문서 수")
    embedding_model: str = Field(default="text-embedding-3-small")
    vector_store_type: VectorStoreType = "faiss"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap({v})은 chunk_size({chunk_size})보다 작아야 합니다"
            )
        return v

    def to_splitter_params(self) -> dict:
        """RecursiveCharacterTextSplitter 파라미터 형식으로 변환"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


# ============================================================
# 5. 환경 변수 설정
# ============================================================


class Settings(BaseSettings):
    """애플리케이션 설정 - 환경변수에서 자동 로드

    .env 파일 또는 시스템 환경변수에서 설정을 읽습니다.
    """

    # OpenAI 설정
    openai_api_key: str = Field(description="OpenAI API 키")

    # LangSmith 설정 (선택)
    langchain_api_key: str | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "default"

    # 앱 설정
    debug: bool = False
    log_level: str = "INFO"
    max_retries: int = MAX_RETRIES

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """설정 싱글톤 반환 (한 번만 로드)"""
    return Settings()


# ============================================================
# 6. 커스텀 예외 클래스
# ============================================================


class AIServiceError(Exception):
    """AI 서비스 기본 예외 클래스"""

    pass


class LLMGenerationError(AIServiceError):
    """LLM 생성 실패 예외"""

    def __init__(self, message: str, model: str = "", retry_count: int = 0):
        super().__init__(message)
        self.model = model
        self.retry_count = retry_count

    def __str__(self) -> str:
        return f"LLMGenerationError(model={self.model}, retries={self.retry_count}): {super().__str__()}"


class DocumentProcessingError(AIServiceError):
    """문서 처리 실패 예외"""

    def __init__(self, message: str, file_path: str = ""):
        super().__init__(message)
        self.file_path = file_path


class VectorStoreError(AIServiceError):
    """벡터저장소 관련 오류"""

    pass


class ConfigurationError(AIServiceError):
    """설정 오류"""

    pass


# ============================================================
# 7. 컨텍스트 매니저
# ============================================================


@contextmanager
def timer(operation_name: str) -> Generator[None, None, None]:
    """코드 블록의 실행 시간을 측정하는 컨텍스트 매니저

    Usage:
        with timer("LLM 응답 생성"):
            response = llm.invoke(...)
    """
    start = time.perf_counter()
    logger.info(f"[{operation_name}] 시작")
    try:
        yield
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error(f"[{operation_name}] 실패 ({elapsed:.2f}ms): {e}")
        raise
    else:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"[{operation_name}] 완료: {elapsed:.2f}ms")


# ============================================================
# 8. Protocol (인터페이스 정의)
# ============================================================


@runtime_checkable
class Embedder(Protocol):
    """임베딩 모델 인터페이스"""

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리 임베딩"""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """다수 문서 임베딩"""
        ...


@runtime_checkable
class Retriever(Protocol):
    """검색기 인터페이스"""

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """관련 문서 검색"""
        ...


# ============================================================
# 9. 함수형 프로그래밍 예시
# ============================================================


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> Generator[str, None, None]:
    """텍스트를 청크로 분할하는 제너레이터

    Args:
        text: 분할할 텍스트
        chunk_size: 청크 크기 (문자 수)
        overlap: 청크 간 겹침 크기

    Yields:
        분할된 텍스트 청크

    Example:
        >>> list(chunk_text("abcdefghij", chunk_size=4, overlap=1))
        ['abcd', 'defg', 'ghij']
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size는 양수여야 합니다: {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap은 0 이상 chunk_size 미만이어야 합니다: {overlap}")

    start = 0
    while start < len(text):
        end = start + chunk_size
        yield text[start:end]
        start += chunk_size - overlap


def clean_text(text: str) -> str:
    """텍스트 정제 함수

    - 앞뒤 공백 제거
    - 연속 공백 → 단일 공백
    - 특수 공백 문자 처리
    """
    import re

    # 앞뒤 공백 제거
    text = text.strip()
    # 연속 공백 → 단일 공백
    text = re.sub(r"\s+", " ", text)
    # 특수 공백 문자 정규화
    text = text.replace("\u200b", "").replace("\xa0", " ")
    return text


# 고차 함수 (Higher-order functions) 예시
def create_text_cleaner(
    *preprocessors: Callable[[str], str],
) -> Callable[[str], str]:
    """여러 전처리 함수를 파이프라인으로 결합

    Args:
        *preprocessors: 순서대로 적용할 전처리 함수들

    Returns:
        결합된 전처리 함수

    Example:
        >>> cleaner = create_text_cleaner(str.lower, str.strip)
        >>> cleaner("  HELLO WORLD  ")
        'hello world'
    """
    def pipeline(text: str) -> str:
        return reduce(lambda t, fn: fn(t), preprocessors, text)
    return pipeline


# 리스트 컴프리헨션과 제너레이터 예시
def filter_valid_documents(
    documents: list[dict],
    min_length: int = 100,
) -> list[dict]:
    """유효한 문서만 필터링

    Args:
        documents: 문서 딕셔너리 목록 (content 키 필요)
        min_length: 최소 문서 길이

    Returns:
        유효한 문서 목록
    """
    return [
        doc
        for doc in documents
        if doc.get("content") and len(doc["content"]) >= min_length
    ]


def batch_process(
    items: list[T],
    batch_size: int,
    processor: Callable[[list[T]], list],
) -> list:
    """대용량 데이터를 배치로 처리

    Args:
        items: 처리할 아이템 목록
        batch_size: 배치 크기
        processor: 배치를 처리하는 함수

    Returns:
        모든 배치 처리 결과를 합친 목록
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = processor(batch)
        results.extend(batch_results)
        logger.info(f"배치 처리 진행: {min(i + batch_size, len(items))}/{len(items)}")
    return results


# ============================================================
# 10. 문서 로딩 및 처리 예시 (AI 서비스 실전)
# ============================================================


@dataclass
class Document:
    """처리된 문서 모델

    Attributes:
        content: 문서 본문
        metadata: 문서 메타데이터 (파일명, 페이지, 소스 등)
        doc_id: 문서 고유 ID
    """

    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: os.urandom(8).hex())

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Document content는 비어 있을 수 없습니다")

    def truncate(self, max_length: int) -> "Document":
        """문서를 최대 길이로 자름"""
        return Document(
            content=self.content[:max_length],
            metadata={**self.metadata, "truncated": True},
        )


def load_text_file(file_path: str | Path) -> Document:
    """텍스트 파일을 Document 객체로 로드

    Args:
        file_path: 파일 경로

    Returns:
        Document 객체

    Raises:
        DocumentProcessingError: 파일 처리 실패 시
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"파일을 찾을 수 없습니다: {path}",
            file_path=str(path),
        )

    if path.suffix.lower() not in {".txt", ".md", ".rst"}:
        raise DocumentProcessingError(
            f"지원하지 않는 파일 형식: {path.suffix}",
            file_path=str(path),
        )

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="cp949")  # 한국어 파일 처리
        except UnicodeDecodeError as e:
            raise DocumentProcessingError(
                f"파일 인코딩 오류: {e}",
                file_path=str(path),
            ) from e

    return Document(
        content=clean_text(content),
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_size": path.stat().st_size,
        },
    )


# ============================================================
# 11. 재시도 로직 (Retry Pattern)
# ============================================================


def retry(
    max_retries: int = MAX_RETRIES,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """재시도 데코레이터

    Args:
        max_retries: 최대 재시도 횟수
        delay: 재시도 간격 (초)
        exceptions: 재시도할 예외 타입들

    Usage:
        @retry(max_retries=3, delay=1.0, exceptions=(RateLimitError,))
        def call_llm(prompt: str) -> str:
            ...
    """
    import functools

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # 지수 백오프
                        logger.warning(
                            f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}): "
                            f"{e}. {wait_time:.1f}초 후 재시도..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} 최종 실패 ({max_retries}회 시도): {e}"
                        )
            raise last_exception

        return wrapper

    return decorator


# ============================================================
# 12. 실습: 나쁜 코드를 좋은 코드로 리팩토링
# ============================================================


# === 나쁜 코드 (리팩토링 전) ===
def bad_process(d, n=5, t=0.7):
    """Bad: 이름이 불명확하고 문서화가 없음"""
    r = []
    for i in d:
        if len(i) > 10:
            x = i[:n]
            r.append({"t": t, "d": x})
    return r


# === 좋은 코드 (리팩토링 후) ===
def process_documents_for_llm(
    documents: list[str],
    max_preview_length: int = 5,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    """LLM 처리를 위한 문서 전처리

    긴 문서만 선택하고, 미리보기와 설정을 포함한 형식으로 변환합니다.

    Args:
        documents: 처리할 문서 텍스트 목록
        max_preview_length: 미리보기에 포함할 최소 문서 길이
        temperature: LLM 온도 파라미터

    Returns:
        처리된 문서 딕셔너리 목록
        각 딕셔너리는 'temperature'와 'document_preview' 키를 가짐

    Example:
        >>> docs = ["short", "this is a long enough document"]
        >>> result = process_documents_for_llm(docs, max_preview_length=10)
        >>> len(result)
        1
    """
    return [
        {
            "temperature": temperature,
            "document_preview": doc[:max_preview_length],
        }
        for doc in documents
        if len(doc) > max_preview_length
    ]


# ============================================================
# 메인 실행 (테스트)
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Python Clean Code 예시 실행")
    print("=" * 60)

    # 1. 데이터클래스 테스트
    print("\n[1] 데이터클래스 테스트")
    config = LLMConfig(temperature=0.3, stream=True)
    print(f"LLM Config: {config}")
    print(f"API 파라미터: {config.to_dict()}")

    # 2. 유효성 검사 테스트
    print("\n[2] 유효성 검사 테스트")
    try:
        bad_config = LLMConfig(temperature=3.0)  # 잘못된 값
    except ValueError as e:
        print(f"예상된 오류: {e}")

    # 3. RAG 설정 테스트
    print("\n[3] Pydantic 모델 테스트")
    rag_config = RAGConfig(chunk_size=500, chunk_overlap=100, top_k=5)
    print(f"RAG Config: {rag_config}")
    print(f"Splitter 파라미터: {rag_config.to_splitter_params()}")

    # 4. 텍스트 청크 분할 테스트
    print("\n[4] 텍스트 청크 분할 테스트")
    sample_text = "가나다라마바사아자차카타파하" * 10
    chunks = list(chunk_text(sample_text, chunk_size=20, overlap=5))
    print(f"원본 길이: {len(sample_text)}")
    print(f"청크 수: {len(chunks)}")
    print(f"첫 번째 청크: '{chunks[0]}'")

    # 5. 텍스트 정제 파이프라인 테스트
    print("\n[5] 텍스트 정제 파이프라인 테스트")
    cleaner = create_text_cleaner(str.strip, clean_text)
    dirty_text = "  안녕하세요   LangChain  입니다.  "
    clean = cleaner(dirty_text)
    print(f"원본: '{dirty_text}'")
    print(f"정제 후: '{clean}'")

    # 6. 타이머 컨텍스트 매니저 테스트
    print("\n[6] 타이머 테스트")
    with timer("테스트 작업"):
        time.sleep(0.1)  # 작업 시뮬레이션
        result = sum(range(1000000))
    print(f"계산 결과: {result}")

    # 7. 리팩토링 전/후 비교
    print("\n[7] 리팩토링 비교")
    test_docs = ["짧음", "이것은 충분히 긴 문서입니다"]
    before = bad_process(test_docs, n=5, t=0.5)
    after = process_documents_for_llm(test_docs, max_preview_length=5, temperature=0.5)
    print(f"리팩토링 전: {before}")
    print(f"리팩토링 후: {after}")

    # 8. 커스텀 예외 테스트
    print("\n[8] 커스텀 예외 테스트")
    try:
        raise DocumentProcessingError(
            "파일을 찾을 수 없습니다",
            file_path="/path/to/nonexistent.pdf",
        )
    except DocumentProcessingError as e:
        print(f"DocumentProcessingError: {e}")
        print(f"파일 경로: {e.file_path}")

    # 9. 배치 처리 테스트
    print("\n[9] 배치 처리 테스트")

    def mock_embed(texts: list[str]) -> list[list[float]]:
        """임베딩 모의 함수"""
        return [[float(i) for i in range(3)] for _ in texts]

    texts = [f"문서 {i}" for i in range(10)]
    embeddings = batch_process(texts, batch_size=3, processor=mock_embed)
    print(f"처리된 임베딩 수: {len(embeddings)}")

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)
