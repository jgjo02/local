# Week 8-2: LangGraph 조건부 엣지 및 분기 처리

## 학습 목표
- 조건부 엣지(Conditional Edge)로 동적 분기 구현
- 라우팅 노드 패턴 이해
- 복잡한 워크플로우 설계

---

## 1. 조건부 엣지 기본

### 1.1 add_conditional_edges

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class State(TypedDict):
    query: str
    query_type: str  # "simple" | "complex" | "off_topic"
    answer: str

def classify_query(state: State) -> str:
    """라우팅 함수: 다음 노드 이름 반환"""
    query_type = state.get("query_type", "simple")

    if query_type == "off_topic":
        return "reject"
    elif query_type == "complex":
        return "deep_research"
    else:
        return "quick_answer"

# 노드 등록
builder = StateGraph(State)
builder.add_node("classify", classify_node)
builder.add_node("quick_answer", quick_answer_node)
builder.add_node("deep_research", deep_research_node)
builder.add_node("reject", reject_node)

# 조건부 엣지 추가
builder.add_conditional_edges(
    "classify",
    classify_query,
    {
        "quick_answer": "quick_answer",
        "deep_research": "deep_research",
        "reject": "reject",
    }
)
```

---

## 2. 쿼리 분류 라우팅 그래프

```python
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    query: str
    category: Literal["factual", "analytical", "creative", "off_topic"]
    answer: str
    sources: list

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def route_query(state: RouterState) -> RouterState:
    """쿼리 카테고리 분류"""
    prompt = f"""다음 질문의 유형을 분류하세요:
질문: {state['query']}

유형:
- factual: 사실 확인 (정보 조회)
- analytical: 분석/비교 필요
- creative: 창의적 생성
- off_topic: 무관한 질문

유형만 반환: """

    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()

    if category not in ["factual", "analytical", "creative"]:
        category = "off_topic"

    return {**state, "category": category}


def get_route(state: RouterState) -> str:
    return state["category"]


def handle_factual(state: RouterState) -> RouterState:
    answer = f"팩트 답변: {state['query']}에 대한 정보입니다..."
    return {**state, "answer": answer}


def handle_analytical(state: RouterState) -> RouterState:
    answer = f"분석 결과: {state['query']}를 분석하면..."
    return {**state, "answer": answer}


def handle_creative(state: RouterState) -> RouterState:
    answer = f"창의적 응답: {state['query']}에 대한 새로운 관점..."
    return {**state, "answer": answer}


def handle_off_topic(state: RouterState) -> RouterState:
    answer = "죄송합니다. 해당 주제는 제 전문 영역이 아닙니다."
    return {**state, "answer": answer}


# 그래프 구성
builder = StateGraph(RouterState)
builder.add_node("router", route_query)
builder.add_node("factual", handle_factual)
builder.add_node("analytical", handle_analytical)
builder.add_node("creative", handle_creative)
builder.add_node("off_topic", handle_off_topic)

builder.set_entry_point("router")
builder.add_conditional_edges(
    "router",
    get_route,
    {
        "factual": "factual",
        "analytical": "analytical",
        "creative": "creative",
        "off_topic": "off_topic",
    }
)

for node in ["factual", "analytical", "creative", "off_topic"]:
    builder.add_edge(node, END)

graph = builder.compile()
```

---

## 3. 반복 실행 패턴 (Retry Loop)

```python
from typing import TypedDict, Annotated
import operator

class RetryState(TypedDict):
    query: str
    answer: str
    quality_score: float
    retry_count: int
    max_retries: int

def generate_answer(state: RetryState) -> RetryState:
    """답변 생성"""
    response = llm.invoke([HumanMessage(content=state["query"])])
    return {**state, "answer": response.content}

def evaluate_answer(state: RetryState) -> RetryState:
    """답변 품질 평가"""
    eval_prompt = f"""다음 답변의 품질을 0.0-1.0 점수로만 평가하세요.
질문: {state['query']}
답변: {state['answer']}
점수:"""
    response = llm.invoke([HumanMessage(content=eval_prompt)])
    try:
        score = float(response.content.strip())
    except:
        score = 0.5

    return {**state, "quality_score": score}

def should_retry(state: RetryState) -> str:
    """재시도 여부 결정"""
    if state["quality_score"] >= 0.8:
        return "done"
    if state["retry_count"] >= state["max_retries"]:
        return "done"
    return "retry"

def increment_retry(state: RetryState) -> RetryState:
    return {**state, "retry_count": state["retry_count"] + 1}

# 그래프 구성
builder = StateGraph(RetryState)
builder.add_node("generate", generate_answer)
builder.add_node("evaluate", evaluate_answer)
builder.add_node("increment", increment_retry)

builder.set_entry_point("generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate",
    should_retry,
    {"done": END, "retry": "increment"}
)
builder.add_edge("increment", "generate")

retry_graph = builder.compile()

result = retry_graph.invoke({
    "query": "양자 컴퓨팅이란?",
    "answer": "",
    "quality_score": 0.0,
    "retry_count": 0,
    "max_retries": 3,
})
print(f"최종 품질: {result['quality_score']:.2f}")
print(f"재시도 횟수: {result['retry_count']}")
```

---

## 4. 병렬 실행 패턴

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ParallelState(TypedDict):
    query: str
    web_result: str
    db_result: str
    final_answer: str

def search_web(state: ParallelState) -> ParallelState:
    """웹 검색 (병렬 실행 노드)"""
    result = f"웹 검색 결과: {state['query']}..."
    return {**state, "web_result": result}

def search_db(state: ParallelState) -> ParallelState:
    """DB 검색 (병렬 실행 노드)"""
    result = f"DB 검색 결과: {state['query']}..."
    return {**state, "db_result": result}

def combine_results(state: ParallelState) -> ParallelState:
    """결과 통합"""
    combined = f"""통합 답변:
- 웹: {state['web_result']}
- DB: {state['db_result']}"""
    return {**state, "final_answer": combined}

# 병렬 실행을 위한 팬아웃/팬인 패턴
builder = StateGraph(ParallelState)
builder.add_node("search_web", search_web)
builder.add_node("search_db", search_db)
builder.add_node("combine", combine_results)

builder.set_entry_point("search_web")
# 두 검색 노드를 병렬로 실행
builder.add_edge("search_web", "combine")
builder.add_edge("search_db", "combine")
builder.add_edge("combine", END)
```

---

## 핵심 정리

| 패턴 | 사용 시 | 구현 방법 |
|------|---------|-----------|
| 조건부 라우팅 | 입력 유형별 다른 처리 | `add_conditional_edges` |
| 반복 루프 | 품질 기반 재시도 | 조건부 엣지 → 자기 노드 |
| 병렬 처리 | 독립적인 여러 작업 | 여러 엣지 → 통합 노드 |
| 폴백 | 실패 시 대안 실행 | 조건부 엣지 → 폴백 노드 |
