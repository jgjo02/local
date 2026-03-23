# Week 9-3: Corrective RAG 구현

## 학습 목표
- Corrective RAG의 교정 메커니즘 이해
- 웹 검색 폴백 구현
- 지식 정제(Knowledge Refinement) 패턴

---

## 1. Corrective RAG란?

### 1.1 핵심 아이디어

검색된 문서가 불충분하거나 무관할 때 **자동으로 교정**하는 RAG 방식:

1. 문서 관련성 평가
2. 관련성 낮으면 → 웹 검색으로 보완
3. 지식 정제 후 답변 생성

```
검색 → 관련성 평가
    ├── CORRECT (관련 있음) → 직접 생성
    ├── AMBIGUOUS (불확실) → 검색 + 웹 보완
    └── INCORRECT (관련 없음) → 웹 검색으로 대체
```

### 1.2 vs Self-RAG

| 항목 | Self-RAG | Corrective RAG |
|------|----------|----------------|
| 초점 | 전체 파이프라인 자기 반성 | 검색 결과 교정 |
| 웹 검색 | 선택적 | 적극적 활용 |
| 복잡도 | 높음 | 중간 |

---

## 2. Corrective RAG 상태 및 구현

```python
from typing import TypedDict, Literal, List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class CorrectiveState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    web_docs: List[Document]
    relevance_decision: Literal["correct", "ambiguous", "incorrect"]
    refined_knowledge: str
    answer: str

def grade_documents(state: CorrectiveState) -> CorrectiveState:
    """검색 문서 관련성 등급 판정"""
    if not state["retrieved_docs"]:
        return {**state, "relevance_decision": "incorrect"}

    docs_text = "\n---\n".join([
        doc.page_content[:300] for doc in state["retrieved_docs"][:3]
    ])

    prompt = f"""다음 문서들이 질문에 대해 얼마나 관련있는지 평가하세요.

질문: {state['query']}

문서들:
{docs_text}

판정:
- correct: 문서들이 직접적으로 관련있고 답변에 충분
- ambiguous: 부분적으로 관련있으나 추가 정보 필요
- incorrect: 문서들이 질문과 무관함

판정 결과만 반환 (correct/ambiguous/incorrect):"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    if decision not in ["correct", "ambiguous", "incorrect"]:
        decision = "ambiguous"

    return {**state, "relevance_decision": decision}
```

---

## 3. 웹 검색 폴백

```python
from langchain_community.tools import DuckDuckGoSearchRun

def web_search_fallback(state: CorrectiveState) -> CorrectiveState:
    """웹 검색으로 지식 보완"""
    search = DuckDuckGoSearchRun()

    try:
        web_results = search.run(state["query"])
        # 결과를 Document로 변환
        web_doc = Document(
            page_content=web_results,
            metadata={"source": "web_search", "query": state["query"]}
        )
        return {**state, "web_docs": [web_doc]}
    except Exception as e:
        print(f"웹 검색 실패: {e}")
        return {**state, "web_docs": []}
```

---

## 4. 지식 정제 (Knowledge Refinement)

```python
def refine_knowledge(state: CorrectiveState) -> CorrectiveState:
    """검색 결과와 웹 결과를 정제하여 통합"""
    decision = state["relevance_decision"]
    retrieved_docs = state["retrieved_docs"]
    web_docs = state["web_docs"]

    if decision == "correct":
        # 로컬 검색만 사용
        source_docs = retrieved_docs[:3]
    elif decision == "ambiguous":
        # 로컬 + 웹 결합
        source_docs = retrieved_docs[:2] + web_docs[:2]
    else:
        # 웹 검색만 사용
        source_docs = web_docs[:3]

    if not source_docs:
        return {**state, "refined_knowledge": "관련 정보를 찾을 수 없습니다."}

    # 지식 정제: 핵심 정보만 추출
    raw_content = "\n\n".join([doc.page_content[:500] for doc in source_docs])

    refine_prompt = f"""다음 자료에서 '{state['query']}'에 답하는데 필요한 핵심 정보만 추출하세요.

자료:
{raw_content}

핵심 정보 (간결하게):"""

    response = llm.invoke([HumanMessage(content=refine_prompt)])
    return {**state, "refined_knowledge": response.content}


def generate_corrected_answer(state: CorrectiveState) -> CorrectiveState:
    """정제된 지식으로 최종 답변 생성"""
    prompt = f"""다음 정제된 정보를 바탕으로 질문에 답하세요.

정보:
{state['refined_knowledge']}

질문: {state['query']}

상세하고 정확한 답변:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "answer": response.content}
```

---

## 5. Corrective RAG 그래프

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def route_correction(state: CorrectiveState) -> str:
    decision = state["relevance_decision"]
    if decision == "correct":
        return "refine_only"  # 웹 검색 없이 정제
    elif decision == "ambiguous":
        return "web_supplement"  # 웹으로 보완
    else:
        return "web_replace"  # 웹으로 대체

memory = MemorySaver()
builder = StateGraph(CorrectiveState)

builder.add_node("retrieve", lambda s: retrieve_docs(s, retriever))
builder.add_node("grade", grade_documents)
builder.add_node("web_search", web_search_fallback)
builder.add_node("refine", refine_knowledge)
builder.add_node("generate", generate_corrected_answer)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    route_correction,
    {
        "refine_only": "refine",
        "web_supplement": "web_search",
        "web_replace": "web_search",
    }
)
builder.add_edge("web_search", "refine")
builder.add_edge("refine", "generate")
builder.add_edge("generate", END)

corrective_rag = builder.compile(checkpointer=memory)

# 테스트
result = corrective_rag.invoke({
    "query": "2024년 최저임금은 얼마인가요?",
    "retrieved_docs": [],
    "web_docs": [],
    "relevance_decision": "incorrect",
    "refined_knowledge": "",
    "answer": "",
})

print(f"교정 결정: {result['relevance_decision']}")
print(f"웹 검색 사용: {len(result['web_docs']) > 0}")
print(f"답변: {result['answer'][:300]}...")
```

---

## 6. 세 가지 RAG 전략 비교

```python
def select_rag_strategy(query: str, retriever) -> str:
    """쿼리 분석 후 최적 전략 선택"""
    # 빠른 분류
    prompt = f"""다음 질문에 가장 적합한 RAG 전략을 선택하세요.

질문: {query}

전략:
- adaptive: 다양한 복잡도의 일반 질문
- self: 정확성 검증이 중요한 전문 질문
- corrective: 최신 정보나 웹 검색이 필요한 질문

전략명만 반환:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    strategy = response.content.strip().lower()

    if strategy not in ["adaptive", "self", "corrective"]:
        strategy = "adaptive"

    return strategy
```

---

## 핵심 정리

**Corrective RAG 특징**:
1. **자동 교정**: 불량 검색 결과 자동 감지 및 교정
2. **웹 폴백**: 로컬 데이터 부족 시 인터넷 활용
3. **지식 정제**: 여러 소스를 통합한 핵심 정보 추출
4. **적응성**: 데이터 품질에 따른 유연한 전략

**언제 사용?**:
- 로컬 벡터 DB가 불완전할 때
- 최신 정보가 필요할 때
- 도메인 지식이 제한적일 때
