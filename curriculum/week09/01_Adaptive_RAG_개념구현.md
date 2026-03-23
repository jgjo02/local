# Week 9-1: Adaptive RAG 개념 및 구현

## 학습 목표
- Adaptive RAG의 핵심 아이디어 이해
- 쿼리 복잡도 기반 전략 선택 구현
- LangGraph로 Adaptive RAG 워크플로우 구축

---

## 1. Adaptive RAG란?

### 1.1 기본 개념

Naive RAG는 모든 쿼리에 동일한 전략을 적용합니다. Adaptive RAG는 **쿼리 복잡도와 특성**에 따라 최적의 검색/생성 전략을 동적으로 선택합니다.

```
입력 쿼리
    ↓
쿼리 분류기 (Classifier)
├── Simple → 직접 LLM 응답 (No Retrieval)
├── Factual → 단순 RAG
├── Analytical → 심층 검색 + 분석
└── Iterative → 반복 검색 (Multi-hop)
    ↓
최적화된 전략으로 응답 생성
```

### 1.2 왜 Adaptive RAG인가?

| 쿼리 유형 | 예시 | 최적 전략 |
|-----------|------|-----------|
| 상식 질문 | "파이썬이란?" | 직접 LLM 응답 |
| 단순 사실 | "계약서란?" | 1-hop RAG |
| 복합 분석 | "A와 B의 법적 차이는?" | 다중 검색 + 분석 |
| 다단계 추론 | "사건 X에서 Y가 가능한가?" | Iterative RAG |

---

## 2. 쿼리 분류기 구현

```python
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AdaptiveState(TypedDict):
    query: str
    query_type: Literal["no_retrieval", "single", "multi", "iterative"]
    context: str
    answer: str
    iteration: int
    max_iterations: int

CLASSIFIER_PROMPT = """다음 질문의 유형을 분류하세요.

질문: {query}

유형:
- no_retrieval: 일반 상식, 단순 계산 (검색 불필요)
- single: 단일 문서로 답변 가능
- multi: 여러 문서 종합 필요
- iterative: 여러 단계의 검색과 추론 필요

JSON만 반환: {{"type": "유형", "reason": "이유"}}"""

def classify_query(state: AdaptiveState) -> AdaptiveState:
    prompt = CLASSIFIER_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        data = json.loads(response.content)
        query_type = data.get("type", "single")
    except:
        query_type = "single"

    valid_types = ["no_retrieval", "single", "multi", "iterative"]
    if query_type not in valid_types:
        query_type = "single"

    return {**state, "query_type": query_type}
```

---

## 3. 전략별 노드 구현

### 3.1 직접 응답 (No Retrieval)

```python
def direct_answer(state: AdaptiveState) -> AdaptiveState:
    """검색 없이 LLM 직접 응답"""
    response = llm.invoke([HumanMessage(content=state["query"])])
    return {**state, "answer": response.content, "context": ""}
```

### 3.2 단일 검색 RAG

```python
def single_retrieval(state: AdaptiveState, retriever) -> AdaptiveState:
    """단일 검색 후 답변"""
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    return {**state, "context": context}


def generate_with_context(state: AdaptiveState) -> AdaptiveState:
    """컨텍스트 기반 답변 생성"""
    if not state["context"]:
        return direct_answer(state)

    prompt = f"""다음 자료를 참고하여 질문에 답하세요.

자료:
{state['context']}

질문: {state['query']}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "answer": response.content}
```

### 3.3 다중 검색 (Multi-hop)

```python
def multi_retrieval(state: AdaptiveState, retriever) -> AdaptiveState:
    """여러 관점에서 검색"""
    # 쿼리 분해
    decompose_prompt = f"""다음 질문을 2-3개의 세부 질문으로 분해하세요.
질문: {state['query']}
세부 질문 목록 (줄바꿈으로 구분):"""

    response = llm.invoke([HumanMessage(content=decompose_prompt)])
    sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    all_context = []
    for sub_q in sub_queries[:3]:
        docs = retriever.invoke(sub_q)
        for doc in docs[:2]:
            if doc.page_content not in all_context:
                all_context.append(doc.page_content)

    context = "\n\n".join(all_context[:6])
    return {**state, "context": context}
```

### 3.4 반복 검색 (Iterative)

```python
def iterative_retrieval(state: AdaptiveState, retriever) -> AdaptiveState:
    """반복적으로 검색하며 답변 정제"""
    iteration = state.get("iteration", 0)

    if iteration == 0:
        docs = retriever.invoke(state["query"])
    else:
        # 이전 답변을 바탕으로 추가 검색
        refine_query = f"{state['query']} {state.get('answer', '')[:100]}"
        docs = retriever.invoke(refine_query)

    new_context = "\n\n".join([doc.page_content for doc in docs[:3]])
    combined_context = state.get("context", "") + "\n\n" + new_context

    return {**state, "context": combined_context.strip(), "iteration": iteration + 1}


def should_continue_iterating(state: AdaptiveState) -> str:
    """반복 계속 여부 결정"""
    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    # 답변 충분성 확인
    if len(state.get("answer", "")) > 200:
        return "finalize"

    return "iterate"
```

---

## 4. Adaptive RAG 그래프 완성

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def get_strategy(state: AdaptiveState) -> str:
    return state["query_type"]

memory = MemorySaver()
builder = StateGraph(AdaptiveState)

builder.add_node("classify", classify_query)
builder.add_node("direct_answer", direct_answer)
builder.add_node("single_retrieve", lambda s: single_retrieval(s, retriever))
builder.add_node("multi_retrieve", lambda s: multi_retrieval(s, retriever))
builder.add_node("iterative_retrieve", lambda s: iterative_retrieval(s, retriever))
builder.add_node("generate", generate_with_context)

builder.set_entry_point("classify")

builder.add_conditional_edges(
    "classify",
    get_strategy,
    {
        "no_retrieval": "direct_answer",
        "single": "single_retrieve",
        "multi": "multi_retrieve",
        "iterative": "iterative_retrieve",
    }
)

for node in ["single_retrieve", "multi_retrieve"]:
    builder.add_edge(node, "generate")

builder.add_conditional_edges(
    "iterative_retrieve",
    should_continue_iterating,
    {"iterate": "generate", "finalize": "generate"},
)

builder.add_edge("generate", END)
builder.add_edge("direct_answer", END)

adaptive_graph = builder.compile(checkpointer=memory)

# 테스트
result = adaptive_graph.invoke({
    "query": "임대차보호법에서 묵시적 갱신이란 무엇인가요?",
    "query_type": "",
    "context": "",
    "answer": "",
    "iteration": 0,
    "max_iterations": 3,
})
print(f"전략: {result['query_type']}")
print(f"답변: {result['answer'][:300]}...")
```

---

## 핵심 정리

**Adaptive RAG의 핵심**:
1. **분류기 품질이 전체 성능 결정** - 정확한 쿼리 분류가 핵심
2. **전략별 최적화** - 각 전략은 해당 쿼리 유형에 최적화
3. **계산 비용 절감** - 단순 쿼리에 불필요한 검색 방지
4. **품질 향상** - 복잡한 쿼리에 더 강력한 전략 적용
