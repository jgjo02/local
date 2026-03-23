# 2차시: LangChain 내장 도구 활용 (웹 검색 도구 실습)

## 📌 학습 목표

이번 차시를 마치면 다음을 할 수 있습니다:
1. LangChain이 제공하는 주요 내장 도구의 특성과 사용 사례를 이해한다
2. TavilySearchResults, DuckDuckGoSearchRun 등 검색 도구를 실제로 설정하고 활용한다
3. 여러 도구를 조합하는 체이닝 전략을 설계할 수 있다
4. 검색 결과를 파싱하고 LangChain 파이프라인에 통합할 수 있다

---

## 🧰 LangChain 내장 도구 전체 목록

LangChain은 `langchain-community` 패키지를 통해 50개 이상의 내장 도구를 제공합니다. 주요 카테고리별로 정리하면:

### 카테고리별 도구 목록

```
┌─────────────────────────────────────────────────────────────┐
│                  LangChain 내장 도구 생태계                   │
├──────────────────┬──────────────────────────────────────────┤
│   검색 도구      │  TavilySearchResults, DuckDuckGoSearchRun │
│                  │  BingSearchRun, GoogleSerperAPIWrapper    │
│                  │  SerpAPIWrapper, YouSearchTool            │
├──────────────────┼──────────────────────────────────────────┤
│   지식 도구      │  WikipediaQueryRun, ArxivQueryRun         │
│                  │  WolframAlphaQueryRun, PubmedQueryRun     │
├──────────────────┼──────────────────────────────────────────┤
│   코드 실행      │  PythonREPLTool, BashTool                 │
│                  │  E2BDataAnalysisTool                      │
├──────────────────┼──────────────────────────────────────────┤
│   파일/데이터    │  ReadFileTool, WriteFileTool              │
│                  │  JsonGetValueTool, CSVLoader              │
├──────────────────┼──────────────────────────────────────────┤
│   벡터DB/RAG     │  VectorStoreQATool, RetrieverTool        │
├──────────────────┼──────────────────────────────────────────┤
│   외부 서비스    │  OpenWeatherMap, GoogleCalendar           │
│                  │  HumanInputRun, SlackTool                 │
├──────────────────┼──────────────────────────────────────────┤
│   멀티모달       │  DallEImageGenerator, StableDiffusion    │
└──────────────────┴──────────────────────────────────────────┘
```

---

## 🔍 TavilySearchResults 설정 및 활용

### Tavily란?

Tavily는 AI 에이전트를 위해 특별히 설계된 검색 API입니다. 일반 웹 검색과 달리:
- AI 에이전트에 최적화된 JSON 형식 반환
- 실시간 웹 크롤링으로 최신 정보 제공
- 관련도 점수 포함
- 콘텐츠 필터링 및 요약 기능

### 설치 및 설정

```bash
pip install tavily-python langchain-community
export TAVILY_API_KEY="tvly-xxxxx"
```

### 기본 사용법

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# 기본 설정
search = TavilySearchResults(
    max_results=5,           # 최대 결과 수
    search_depth="advanced", # "basic" 또는 "advanced"
    include_answer=True,     # AI 요약 답변 포함
    include_raw_content=False,  # 전체 페이지 내용 포함 여부
    include_images=False,    # 이미지 URL 포함 여부
)

# 검색 실행
results = search.invoke("2024년 한국 최저임금")
```

### 검색 결과 구조

```json
[
    {
        "title": "2024년 최저임금 9,860원 확정",
        "url": "https://example.com/news/12345",
        "content": "고용노동부는 2024년 적용 최저임금을 시간당 9,860원으로...",
        "score": 0.95,
        "published_date": "2023-08-05"
    },
    ...
]
```

### 법률 검색에 최적화하기

```python
from langchain_community.tools.tavily_search import TavilySearchResults

legal_search = TavilySearchResults(
    max_results=10,
    search_depth="advanced",
    include_answer=True,
    # 법률 사이트에 집중
    include_domains=["law.go.kr", "lawnb.com", "scourt.go.kr"],
    # 비관련 사이트 제외
    exclude_domains=["shopping.com", "social.com"],
)
```

---

## 🦆 DuckDuckGoSearchRun 활용

### DuckDuckGo의 특징

- **무료**: API 키 불필요
- **프라이버시 중심**: 사용자 추적 없음
- **제한**: 속도 제한, 결과 품질이 Tavily보다 낮을 수 있음
- **적합한 용도**: 개발/테스트 환경, 간단한 검색

### 설치 및 사용법

```bash
pip install duckduckgo-search
```

```python
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

# 간단한 텍스트 반환
ddg_simple = DuckDuckGoSearchRun(
    backend="text",  # "text", "news", "images"
)
result = ddg_simple.invoke("한국 노동법 최신 개정")

# 구조화된 결과 반환
ddg_structured = DuckDuckGoSearchResults(
    backend="news",  # 뉴스 검색
    num_results=5,
    output_format="list",  # "list" 또는 "json"
)
results = ddg_structured.invoke("법률 개정 2024")
```

### DuckDuckGo 검색 백엔드 비교

```
┌──────────────┬─────────────────────────────────────────────┐
│   백엔드     │  설명                                       │
├──────────────┼─────────────────────────────────────────────┤
│   text       │  일반 웹 검색 (기본값)                       │
│   news       │  최신 뉴스 검색 (실시간 정보)               │
│   images     │  이미지 검색                                 │
│   videos     │  동영상 검색                                 │
│   maps       │  지도/위치 검색                              │
└──────────────┴─────────────────────────────────────────────┘
```

---

## 📖 WikipediaQueryRun 활용

### Wikipedia 도구의 특징

- 백과사전적 지식 제공
- 법률 용어, 개념 설명에 유용
- 한국어 Wikipedia 지원
- 구조화된 정보 제공

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 한국어 Wikipedia 설정
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang="ko",          # 한국어
        top_k_results=3,    # 상위 3개 결과
        doc_content_chars_max=2000,  # 최대 문자 수
        load_all_available_meta=True,  # 메타데이터 포함
    )
)

# 법률 용어 검색
result = wiki_tool.invoke("소멸시효")
print(result)
```

### Wikipedia 결과 예시

```
소멸시효

소멸시효(消滅時效)는 권리자가 그 권리를 행사할 수 있음에도
일정한 기간 동안 권리를 행사하지 않을 경우 그 권리가 소멸되는
제도이다.

== 종류 ==
* 일반 소멸시효: 민법상 10년 (채권)
* 단기 소멸시효: 3년, 2년, 1년 등 특별 규정...
```

---

## 📚 ArxivQueryRun 활용

### Arxiv 도구의 특징

- AI/ML 최신 논문 검색
- Adaptive RAG, Self-RAG 등 학술 논문 접근
- 법학 논문 검색 (일부)
- 무료 사용

```python
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=3,
        load_all_available_meta=True,
        doc_content_chars_max=3000,
    )
)

# RAG 관련 최신 논문 검색
result = arxiv_tool.invoke("Self-RAG Corrective RAG 2023")
```

---

## 🐍 PythonREPLTool 활용

### PythonREPL의 특징과 활용

PythonREPLTool은 Python 코드를 실제로 실행할 수 있는 강력한 도구입니다.

```python
from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()

# 법률 기간 계산 코드 실행
code = """
from datetime import datetime, timedelta

# 부당해고 구제신청 기간 계산
dismissal_date = datetime(2024, 1, 15)
deadline = dismissal_date + timedelta(days=90)

print(f"해고일: {dismissal_date.strftime('%Y년 %m월 %d일')}")
print(f"구제신청 마감일: {deadline.strftime('%Y년 %m월 %d일')}")
print(f"오늘 기준 남은 일수: {(deadline - datetime.now()).days}일")
"""

result = python_tool.invoke(code)
print(result)
```

### PythonREPL 사용 시 주의사항

```
┌─────────────────────────────────────────────────────────┐
│               ⚠️ 보안 경고                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  PythonREPLTool은 임의의 코드를 실행합니다!             │
│                                                         │
│  위험 시나리오:                                         │
│  - os.system("rm -rf /")  # 파일 삭제                   │
│  - subprocess.run(...)    # 외부 명령 실행              │
│  - open("/etc/passwd")    # 민감한 파일 접근            │
│                                                         │
│  안전 조치:                                             │
│  1. 샌드박스 환경에서만 사용                            │
│  2. Docker 컨테이너 격리                                │
│  3. E2B와 같은 전용 코드 실행 환경 사용                 │
│  4. 프로덕션 환경에서 사용 금지                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🖥️ ShellTool 활용 및 주의사항

```python
from langchain_community.tools import ShellTool

shell_tool = ShellTool(
    ask_human_input=True,  # 실행 전 사용자 확인 요청
)

# 파일 목록 조회 (비교적 안전한 예)
result = shell_tool.invoke("ls -la /home/user/documents/legal/")
```

### ShellTool 보안 정책

```
프로덕션 환경에서의 ShellTool 사용 지침:

허용 (읽기 전용):
  ✓ ls, cat, grep, find (파일 검색/조회)
  ✓ date, whoami (시스템 정보)

금지 (변경/실행):
  ✗ rm, mv, chmod (파일 수정)
  ✗ sudo, su (권한 상승)
  ✗ curl, wget (외부 다운로드)
  ✗ apt, pip (패키지 설치)

권장: ShellTool 대신 특화된 도구 사용
  → 파일 읽기: ReadFileTool
  → API 호출: 커스텀 도구
```

---

## 🔗 도구 체이닝 전략

### 전략 1: 순차적 도구 체이닝

```
질문 → [웹 검색] → [Wikipedia 보완] → [요약 생성]
```

```python
from langchain_core.runnables import RunnableSequence

# 체인 구성
search_chain = (
    search_tool
    | (lambda x: {"query": x, "search_results": x})
    | wiki_tool
    | summarize_chain
)
```

### 전략 2: 병렬 도구 실행

```
           ┌─[웹 검색]─────┐
질문 ──────┤               ├──→ [결과 통합] → 답변
           └─[위키피디아]──┘
```

```python
from langchain_core.runnables import RunnableParallel

parallel_search = RunnableParallel(
    web_results=search_tool,
    wiki_results=wiki_tool,
    arxiv_results=arxiv_tool,
)

results = parallel_search.invoke("LangGraph 에이전트 프레임워크")
```

### 전략 3: 폴백 체이닝

```python
from langchain_core.runnables import RunnableWithFallbacks

# Tavily가 실패하면 DuckDuckGo로 폴백
robust_search = tavily_search.with_fallbacks(
    [duckduckgo_search],
    exceptions_to_handle=(Exception,),
)
```

### 전략 4: 도구 선택 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                    도구 선택 흐름도                           │
│                                                             │
│  사용자 질문                                                │
│      │                                                      │
│      ▼                                                      │
│  질문 분류                                                  │
│      │                                                      │
│   ┌──┴──────────────────────────────────┐                  │
│   │          │               │           │                  │
│   ▼          ▼               ▼           ▼                  │
│ 최신 정보  법령/판례        개념 설명   계산 필요            │
│   │          │               │           │                  │
│   ▼          ▼               ▼           ▼                  │
│ Tavily   법제처 API      Wikipedia   Python REPL            │
│ Search                                                      │
│   │          │               │           │                  │
│   └──────────┴───────────────┴───────────┘                 │
│                       │                                     │
│                   결과 통합                                  │
│                       │                                     │
│                  최종 답변 생성                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 검색 결과 파싱 및 활용

### Tavily 결과 파싱

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

def parse_tavily_results(query: str) -> list[Document]:
    """Tavily 검색 결과를 LangChain Document로 변환"""

    search = TavilySearchResults(max_results=5)
    raw_results = search.invoke(query)

    documents = []
    for result in raw_results:
        doc = Document(
            page_content=result.get("content", ""),
            metadata={
                "source": result.get("url", ""),
                "title": result.get("title", ""),
                "score": result.get("score", 0),
                "published_date": result.get("published_date", ""),
            }
        )
        documents.append(doc)

    # 관련도 순으로 정렬
    documents.sort(
        key=lambda d: d.metadata.get("score", 0),
        reverse=True
    )

    return documents
```

### 검색 결과 신뢰도 평가

```python
def evaluate_search_quality(results: list) -> dict:
    """검색 결과 품질 평가"""

    if not results:
        return {"quality": "low", "reason": "결과 없음"}

    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    has_recent = any(
        r.get("published_date", "2020") >= "2023"
        for r in results
    )

    quality_report = {
        "total_results": len(results),
        "avg_relevance_score": round(avg_score, 3),
        "has_recent_content": has_recent,
        "quality": "high" if avg_score > 0.7 else "medium" if avg_score > 0.4 else "low",
        "recommendation": "",
    }

    if quality_report["quality"] == "low":
        quality_report["recommendation"] = "검색어를 더 구체적으로 수정하거나 다른 도구 시도"
    elif not has_recent:
        quality_report["recommendation"] = "최신 정보 부족, 뉴스 검색 추가 권장"

    return quality_report
```

---

## 💡 핵심 포인트

1. **도구 선택 전략**: 법률 정보의 경우 Tavily(최신 뉴스) + 법제처 API(공식 법령) + Wikipedia(개념 설명)를 조합하면 가장 신뢰도 높은 결과를 얻습니다.

2. **무료 vs 유료**: DuckDuckGo는 무료이지만 속도 제한이 있습니다. 프로덕션에서는 Tavily나 다른 유료 API를 권장합니다.

3. **결과 품질 모니터링**: 검색 결과의 relevance score를 항상 확인하고, 낮은 품질의 결과는 재검색하거나 다른 도구로 보완하세요.

4. **도구 캐싱**: 동일한 검색 쿼리가 반복될 경우 캐싱을 통해 비용과 지연을 줄일 수 있습니다.

5. **에러 핸들링**: 외부 API는 언제든 다운될 수 있습니다. 항상 폴백 전략을 준비하세요.

---

## ⚠️ 주의사항

- **API 비용**: Tavily 무료 플랜은 월 1,000회 제한입니다. 교육 환경에서는 DuckDuckGo를 우선 사용하고 중요한 실습만 Tavily로 하세요.
- **Rate Limiting**: 검색 API는 분당 요청 횟수에 제한이 있습니다. 자동화된 에이전트 루프에서는 지수 백오프(exponential backoff)를 구현하세요.
- **저작권**: 검색 결과 내용을 그대로 재사용하면 저작권 문제가 발생할 수 있습니다. 요약/인용 형식으로 활용하세요.
- **PythonREPLTool 보안**: 절대로 프로덕션 환경에서 제한 없이 사용하지 마세요. 반드시 샌드박스 환경에서만 사용하세요.

---

## 🔍 심화학습

### 커스텀 검색 도구 만들기

```python
from langchain_core.tools import BaseTool
from langchain_community.utilities import SearxSearchWrapper

class KoreanLegalSearchTool(BaseTool):
    """한국 법률 특화 검색 도구"""

    name = "korean_legal_search"
    description = (
        "한국 법률 관련 정보를 전문 법률 사이트에서 검색합니다. "
        "법원 판결, 법령 해석, 법률 뉴스 검색에 사용하세요."
    )

    def _run(self, query: str) -> str:
        # 법률 특화 사이트에서 검색
        legal_sites = [
            "site:law.go.kr",
            "site:scourt.go.kr",
            "site:moleg.go.kr",
        ]

        combined_query = f"{query} ({' OR '.join(legal_sites)})"
        # 실제 구현...
        return f"법률 검색 결과: {combined_query}"
```

### 검색 결과 재랭킹

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder로 검색 결과 재랭킹
cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
```

---

## 💻 코드 예제

실습 코드는 `code/02_builtin_tools.py`를 참조하세요.

---

## ❓ 차시별 질문

1. **Tavily와 DuckDuckGo의 차이점은 무엇인가요?** 법률 자문 에이전트에서 각 도구를 어떤 상황에 사용하는 것이 적합한지 설명해보세요.

2. **검색 결과를 어떻게 신뢰할 수 있나요?** 웹 검색 결과의 신뢰도를 평가하는 방법을 제안해보세요.

3. **PythonREPLTool을 법률 자문 시스템에서 어떻게 안전하게 활용할 수 있을까요?** 허용할 코드와 금지할 코드의 기준을 정해보세요.

4. **병렬 도구 실행의 장단점은 무엇인가요?** 법률 검색 시나리오에서 순차 실행이 더 나은 경우와 병렬 실행이 더 나은 경우를 각각 예시로 들어 설명하세요.

5. **검색 결과가 없거나 품질이 낮을 때 에이전트는 어떻게 행동해야 할까요?** 폴백 전략 설계 방안을 제시해보세요.
