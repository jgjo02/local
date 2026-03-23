# Week 11-5: LLM 최적화 및 배포

## 학습 목표
- 프로덕션 LLM 서비스 최적화 전략
- 비용 절감 기법 (캐싱, 모델 선택, 토큰 최적화)
- HuggingFace Spaces/AWS 배포 방법

---

## 1. 비용 최적화

### 1.1 모델 계층화

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class TieredLLMSelector:
    """쿼리 복잡도에 따른 모델 선택"""

    TIERS = {
        "fast": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        "standard": ChatOpenAI(model="gpt-4o-mini", temperature=0.3),
        "powerful": ChatOpenAI(model="gpt-4o", temperature=0.3),
    }

    @classmethod
    def select(cls, complexity: str = "standard"):
        return cls.TIERS.get(complexity, cls.TIERS["standard"])

    @classmethod
    def auto_select(cls, query: str):
        """쿼리 길이/복잡도로 자동 선택"""
        if len(query) < 50 and "?" in query:
            return cls.TIERS["fast"]
        elif len(query) > 200 or "분석" in query:
            return cls.TIERS["powerful"]
        else:
            return cls.TIERS["standard"]
```

### 1.2 프롬프트 토큰 최적화

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def truncate_context(context: str, max_tokens: int = 2000, model: str = "gpt-4o-mini") -> str:
    """컨텍스트를 토큰 제한에 맞게 자르기"""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(context)
    if len(tokens) <= max_tokens:
        return context
    truncated = enc.decode(tokens[:max_tokens])
    return truncated + "...[잘림]"

# 비용 계산
def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """API 호출 비용 추정"""
    rates = {
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},  # per 1K tokens
        "gpt-4o": {"input": 0.002500, "output": 0.010000},
    }
    rate = rates.get(model, rates["gpt-4o-mini"])
    cost = (prompt_tokens / 1000 * rate["input"]) + (completion_tokens / 1000 * rate["output"])
    return cost
```

---

## 2. 캐싱 전략

### 2.1 인메모리 캐시

```python
import hashlib
import time
from typing import Optional

class LLMCache:
    def __init__(self, ttl: int = 3600):
        self._cache = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        key = self._key(prompt, model)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["time"] < self.ttl:
                self.hits += 1
                return entry["response"]
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response: str) -> None:
        key = self._key(prompt, model)
        self._cache[key] = {"response": response, "time": time.time()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


# LangChain 캐시 설정
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

set_llm_cache(InMemoryCache())
```

### 2.2 Redis 캐시 (프로덕션)

```python
from langchain_community.cache import RedisCache
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)
set_llm_cache(RedisCache(redis_client))
```

---

## 3. 응답 스트리밍 최적화

```python
from typing import Generator
import gradio as gr

def optimized_stream(prompt: str, llm) -> Generator[str, None, None]:
    """최적화된 스트리밍 응답"""
    buffer = ""
    buffer_size = 5  # 5토큰씩 버퍼링

    for chunk in llm.stream(prompt):
        buffer += chunk.content
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = ""

    if buffer:
        yield buffer


def gradio_stream_handler(message: str, history: list):
    """Gradio 스트리밍 핸들러"""
    partial = ""
    for token in optimized_stream(message, llm):
        partial += token
        yield partial
```

---

## 4. HuggingFace Spaces 배포

### 4.1 app.py 준비

```python
# requirements.txt
langchain>=0.3.0
langchain-openai>=0.2.0
gradio>=4.0.0
faiss-cpu>=1.7.4

# app.py 최상단
import os
import gradio as gr

# HuggingFace Secrets에서 API 키 읽기
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    gr.Warning("OPENAI_API_KEY가 설정되지 않았습니다.")
```

### 4.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
```

### 4.3 GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy to HuggingFace Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Push to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://huggingface.co/spaces/your-username/your-space
          git push space main
```

---

## 5. 모니터링 및 로깅

```python
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_llm_call(func):
    """LLM 호출 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"LLM 호출 성공 | 함수: {func.__name__} | 시간: {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"LLM 호출 실패 | 함수: {func.__name__} | 오류: {e} | 시간: {elapsed:.2f}s")
            raise
    return wrapper

# LangSmith 추적 (선택사항)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "ai-service-curriculum"
```

---

## 핵심 정리

**프로덕션 최적화 체크리스트**:
- [ ] 쿼리 복잡도별 모델 계층화
- [ ] LLM 응답 캐싱 구현
- [ ] 토큰 사용량 모니터링
- [ ] 에러 핸들링 및 재시도 로직
- [ ] 비용 알림 설정
- [ ] LangSmith 추적 활성화
- [ ] 응답 스트리밍 최적화
