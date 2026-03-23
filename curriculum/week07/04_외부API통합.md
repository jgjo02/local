# 4차시: 외부 API 통합

## 학습 목표
- REST API를 LangChain 도구로 통합할 수 있다
- 비동기 API 호출과 에러 처리를 구현할 수 있다
- 캐싱으로 API 비용을 최적화할 수 있다

---

## 1. REST API 통합 패턴

```python
import httpx
from langchain_core.tools import tool

@tool
async def search_law_api(query: str) -> str:
    """국가법령정보 API로 법령 검색"""
    API_KEY = os.getenv("LAW_API_KEY")
    BASE_URL = "https://www.law.go.kr/DRF/lawSearch.do"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            BASE_URL,
            params={
                "OC": API_KEY,
                "target": "law",
                "type": "JSON",
                "query": query
            },
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

    laws = data.get("LawSearch", {}).get("law", [])
    if not laws:
        return f"'{query}' 관련 법령을 찾을 수 없습니다."

    results = []
    for law in laws[:5]:
        results.append(f"- {law.get('법령명한글', '')} ({law.get('공포일자', '')})")

    return "\n".join(results)
```

---

## 2. 재시도 로직 (지수 백오프)

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def api_call_with_retry(url: str, params: dict) -> dict:
    """재시도 로직이 있는 API 호출"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        return response.json()
```

---

## 3. 응답 캐싱

```python
from functools import lru_cache
from cachetools import TTLCache

# TTL 캐시 (1시간)
cache = TTLCache(maxsize=100, ttl=3600)

def cached_law_search(query: str) -> str:
    """캐시를 활용한 법령 검색"""
    if query in cache:
        return cache[query]

    result = law_api_search(query)
    cache[query] = result
    return result
```

---

## 4. API 응답을 Document로 변환

```python
from langchain_core.documents import Document

def api_response_to_documents(api_data: dict) -> list[Document]:
    """API 응답을 LangChain Document로 변환"""
    documents = []
    for item in api_data.get("results", []):
        doc = Document(
            page_content=item.get("content", ""),
            metadata={
                "source": "law_api",
                "law_name": item.get("name", ""),
                "article": item.get("article", ""),
                "effective_date": item.get("date", "")
            }
        )
        documents.append(doc)
    return documents
```

---

## 5. Pydantic으로 API 응답 검증

```python
from pydantic import BaseModel, validator

class LawSearchResult(BaseModel):
    law_name: str
    article_number: str
    content: str
    effective_date: str

    @validator("effective_date")
    def validate_date(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y%m%d")
        except ValueError:
            raise ValueError(f"잘못된 날짜 형식: {v}")
        return v
```

---

## 💡 핵심 포인트
- `httpx`는 비동기 HTTP 클라이언트로 LangChain과 잘 통합됨
- 재시도 로직은 `tenacity` 라이브러리로 쉽게 구현 가능
- TTL 캐시로 동일 쿼리의 API 호출 비용 절감
- Pydantic으로 API 응답 검증 → 런타임 에러 방지

## ❓ 차시별 질문
1. 캐시 TTL을 얼마로 설정해야 할까요? (실시간성 vs 비용)
2. API Rate Limit에 걸렸을 때 사용자에게 어떻게 피드백을 줄까요?
