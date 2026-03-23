# Week 9-2: Self-RAG 구현

## 학습 목표
- Self-RAG의 자기 반성(Self-Reflection) 메커니즘 이해
- Retrieval, Relevance, Support, Utility 토큰 구현
- LangGraph로 Self-RAG 워크플로우 구축

---

## 1. Self-RAG란?

### 1.1 핵심 아이디어

Self-RAG는 LLM이 스스로 **검색 필요 여부**를 결정하고, 생성된 답변을 **자기 검증**하는 RAG 방식입니다.

**4가지 특수 토큰**:
- `[Retrieve]`: 검색이 필요한가?
- `[ISREL]`: 검색된 문서가 관련있는가?
- `[ISSUP]`: 답변이 문서에 의해 지지되는가?
- `[ISUSE]`: 답변이 쿼리에 유용한가?

### 1.2 Self-RAG 흐름

```
질문 입력
    ↓
[Retrieve] 판단 → No → 직접 생성
    ↓ Yes
문서 검색
    ↓
[ISREL] 판단 → No → 재검색
    ↓ Yes
답변 생성
    ↓
[ISSUP] 판단 → No → 재생성
    ↓ Yes
[ISUSE] 판단 → No → 개선
    ↓ Yes
최종 답변
```

---

## 2. Self-RAG 상태 정의

```python
from typing import TypedDict, Literal, List
from langchain.schema import Document

class SelfRAGState(TypedDict):
    query: str
    # 검색 관련
    should_retrieve: bool
    retrieved_docs: List[Document]
    relevant_docs: List[Document]
    # 생성 관련
    draft_answer: str
    final_answer: str
    # 자기 반성 점수
    relevance_score: float    # [ISREL]
    support_score: float      # [ISSUP]
    utility_score: float      # [ISUSE]
    # 제어
    retry_count: int
    max_retries: int
```

---

## 3. 자기 반성 노드 구현

### 3.1 검색 필요 여부 판단

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def decide_retrieval(state: SelfRAGState) -> SelfRAGState:
    """[Retrieve]: 검색 필요 여부 판단"""
    prompt = f"""다음 질문에 답하기 위해 외부 문서 검색이 필요한가요?

질문: {state['query']}

판단 기준:
- 일반 상식이나 단순 계산이면 검색 불필요
- 특정 사실, 법률, 규정, 최신 정보면 검색 필요

"yes" 또는 "no"만 답하세요:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    should_retrieve = "yes" in response.content.lower()

    return {**state, "should_retrieve": should_retrieve}
```

### 3.2 관련성 평가 [ISREL]

```python
def evaluate_relevance(state: SelfRAGState) -> SelfRAGState:
    """[ISREL]: 검색 문서의 관련성 평가"""
    relevant_docs = []
    total_score = 0

    for doc in state["retrieved_docs"]:
        prompt = f"""다음 문서가 질문에 관련이 있는지 평가하세요.

질문: {state['query']}
문서: {doc.page_content[:400]}

관련성 점수 (0.0-1.0):"""

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
        except:
            score = 0.5

        total_score += score
        if score >= 0.6:
            relevant_docs.append(doc)

    avg_score = total_score / len(state["retrieved_docs"]) if state["retrieved_docs"] else 0

    return {
        **state,
        "relevant_docs": relevant_docs,
        "relevance_score": avg_score,
    }
```

### 3.3 지지 평가 [ISSUP]

```python
def evaluate_support(state: SelfRAGState) -> SelfRAGState:
    """[ISSUP]: 답변이 문서에 의해 지지되는지 평가"""
    if not state["relevant_docs"]:
        return {**state, "support_score": 0.0}

    context = "\n".join([doc.page_content[:300] for doc in state["relevant_docs"][:3]])

    prompt = f"""다음 답변이 제공된 문서에 의해 얼마나 지지되는지 평가하세요.

문서:
{context}

답변: {state['draft_answer'][:500]}

지지 점수 (0.0 = 전혀 지지안됨, 1.0 = 완전히 지지됨):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
    except:
        score = 0.5

    return {**state, "support_score": score}
```

### 3.4 유용성 평가 [ISUSE]

```python
def evaluate_utility(state: SelfRAGState) -> SelfRAGState:
    """[ISUSE]: 답변의 유용성 평가"""
    prompt = f"""다음 답변이 질문자에게 얼마나 유용한지 평가하세요.

질문: {state['query']}
답변: {state['draft_answer'][:500]}

유용성 점수 (0.0-1.0):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        score = float(response.content.strip())
    except:
        score = 0.5

    return {**state, "utility_score": score}
```

---

## 4. 라우팅 로직

```python
def route_retrieval(state: SelfRAGState) -> str:
    return "retrieve" if state["should_retrieve"] else "generate_direct"

def route_after_relevance(state: SelfRAGState) -> str:
    if state["relevance_score"] < 0.4:
        return "re_retrieve" if state["retry_count"] < state["max_retries"] else "generate_no_context"
    return "generate"

def route_after_support(state: SelfRAGState) -> str:
    if state["support_score"] >= 0.7 and state["utility_score"] >= 0.6:
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "regenerate"
```

---

## 5. Self-RAG 그래프

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def retrieve_docs(state: SelfRAGState, retriever) -> SelfRAGState:
    docs = retriever.invoke(state["query"])
    return {**state, "retrieved_docs": docs, "retry_count": state.get("retry_count", 0) + 1}

def generate_answer(state: SelfRAGState) -> SelfRAGState:
    context = "\n\n".join([doc.page_content for doc in state.get("relevant_docs", [])[:3]])
    prompt = f"자료:\n{context}\n\n질문: {state['query']}" if context else state["query"]
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "draft_answer": response.content}

def finalize(state: SelfRAGState) -> SelfRAGState:
    return {**state, "final_answer": state["draft_answer"]}

memory = MemorySaver()
builder = StateGraph(SelfRAGState)

builder.add_node("decide", decide_retrieval)
builder.add_node("retrieve", lambda s: retrieve_docs(s, retriever))
builder.add_node("eval_relevance", evaluate_relevance)
builder.add_node("generate", generate_answer)
builder.add_node("eval_support", evaluate_support)
builder.add_node("eval_utility", evaluate_utility)
builder.add_node("finalize", finalize)

builder.set_entry_point("decide")
builder.add_conditional_edges("decide", route_retrieval,
    {"retrieve": "retrieve", "generate_direct": "generate"})
builder.add_edge("retrieve", "eval_relevance")
builder.add_conditional_edges("eval_relevance", route_after_relevance,
    {"generate": "generate", "re_retrieve": "retrieve", "generate_no_context": "generate"})
builder.add_edge("generate", "eval_support")
builder.add_edge("eval_support", "eval_utility")
builder.add_conditional_edges("eval_utility", route_after_support,
    {"finalize": "finalize", "regenerate": "generate"})
builder.add_edge("finalize", END)

self_rag = builder.compile(checkpointer=memory)
```

---

## 핵심 정리

**Self-RAG vs Naive RAG**:

| 항목 | Naive RAG | Self-RAG |
|------|-----------|----------|
| 검색 여부 | 항상 검색 | 필요시만 검색 |
| 관련성 확인 | X | O ([ISREL]) |
| 지지 확인 | X | O ([ISSUP]) |
| 유용성 확인 | X | O ([ISUSE]) |
| API 비용 | 낮음 | 높음 |
| 답변 품질 | 보통 | 높음 |
