# Week 9-4: RAG 전략 통합 및 법률 시스템 적용

## 학습 목표
- 세 가지 RAG 전략을 통합하는 메타 라우터 구현
- Project 3 법률 상담 시스템에 전략 적용
- 전략별 성능 비교 및 최적 전략 선택

---

## 1. 메타 RAG 라우터

```python
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MetaRAGState(TypedDict):
    query: str
    strategy: Literal["adaptive", "self_rag", "corrective"]
    answer: str
    confidence: float

STRATEGY_SELECTION_PROMPT = """법률 질문에 가장 적합한 RAG 전략을 선택하세요.

질문: {query}

전략 기준:
- adaptive: 명확한 법률 개념, 단순 정보 조회
  예) "계약해지란?", "민사소송 절차는?"

- self_rag: 정확성이 중요, 여러 조건 확인 필요
  예) "A 상황에서 B가 가능한가?", "법적 요건은?"

- corrective: 최신 법령, 판례, 외부 정보 필요
  예) "2024년 최저임금은?", "최근 판례는?"

전략명만 반환 (adaptive/self_rag/corrective):"""

def route_legal_query(state: MetaRAGState) -> MetaRAGState:
    prompt = STRATEGY_SELECTION_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])
    strategy = response.content.strip().lower()

    valid = ["adaptive", "self_rag", "corrective"]
    if strategy not in valid:
        strategy = "adaptive"

    return {**state, "strategy": strategy}
```

---

## 2. 법률 도메인 특화 검색

```python
LEGAL_KEYWORDS = {
    "민사": ["계약", "손해배상", "채권", "물권", "불법행위"],
    "노동": ["해고", "임금", "근로시간", "직장내괴롭힘", "산재"],
    "형사": ["고소", "고발", "구속", "기소", "형사절차"],
    "부동산": ["임대차", "전세", "보증금", "명도", "주택"],
}

def enhance_legal_query(query: str, domain: str) -> str:
    """법률 도메인별 쿼리 강화"""
    keywords = LEGAL_KEYWORDS.get(domain, [])
    if not keywords:
        return query

    # 관련 법률 용어 추가
    enhancement = f"{query} (관련: {', '.join(keywords[:3])})"
    return enhancement


def legal_retrieval_with_domain(query: str, domain: str, retriever) -> list:
    """도메인 특화 법률 문서 검색"""
    enhanced_query = enhance_legal_query(query, domain)

    # 1차 검색
    docs = retriever.invoke(enhanced_query)

    # 법률 조문 우선 정렬
    def legal_priority(doc):
        content = doc.page_content.lower()
        score = 0
        if "제" in content and "조" in content:
            score += 2  # 법조문 포함
        if domain.lower() in content:
            score += 1  # 도메인 관련
        return score

    docs.sort(key=legal_priority, reverse=True)
    return docs[:5]
```

---

## 3. 법률 답변 생성 템플릿

```python
LEGAL_ANSWER_TEMPLATE = """당신은 법률 정보 제공 전문가입니다.

## 참고 법률 자료
{context}

## 답변 형식
1. **핵심 답변**: 질문에 대한 직접적인 답변
2. **법적 근거**: 관련 법조문이나 판례
3. **주의사항**: 예외 상황이나 조건
4. **권고사항**: 전문가 상담 권고 여부

## 면책조항
이 정보는 교육 목적이며 법적 조언이 아닙니다. 실제 법적 문제는 변호사 상담을 권장합니다.

---
질문: {query}

답변:"""

def generate_legal_answer_structured(query: str, context: str) -> str:
    prompt = LEGAL_ANSWER_TEMPLATE.format(query=query, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
```

---

## 4. 통합 법률 상담 시스템

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class LegalChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    domain: str
    strategy: str
    context: str
    answer: str

def classify_domain(state: LegalChatState) -> LegalChatState:
    """법률 도메인 분류"""
    prompt = f"""질문의 법률 도메인을 분류하세요: {state['query']}
도메인 (민사/노동/형사/부동산 중 하나):"""
    response = llm.invoke([HumanMessage(content=prompt)])
    domain = response.content.strip()
    valid_domains = ["민사", "노동", "형사", "부동산"]
    if domain not in valid_domains:
        domain = "민사"
    return {**state, "domain": domain}

def select_strategy(state: LegalChatState) -> LegalChatState:
    """RAG 전략 선택"""
    prompt = STRATEGY_SELECTION_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])
    strategy = response.content.strip().lower()
    if strategy not in ["adaptive", "self_rag", "corrective"]:
        strategy = "adaptive"
    return {**state, "strategy": strategy}

def get_strategy_node(state: LegalChatState) -> str:
    return state["strategy"]

def run_adaptive(state: LegalChatState, retriever) -> LegalChatState:
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content[:400] for d in docs[:3]])
    answer = generate_legal_answer_structured(state["query"], context)
    return {**state, "context": context, "answer": answer,
            "messages": [AIMessage(content=answer)]}

def run_self_rag(state: LegalChatState, retriever) -> LegalChatState:
    # 간단한 Self-RAG 실행
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content[:400] for d in docs[:3]])
    answer = generate_legal_answer_structured(state["query"], context)
    # 검증 단계 생략 (실제 구현에서는 Self-RAG 그래프 호출)
    return {**state, "context": context, "answer": answer,
            "messages": [AIMessage(content=answer)]}

def run_corrective(state: LegalChatState, retriever) -> LegalChatState:
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([d.page_content[:400] for d in docs[:3]])
    answer = generate_legal_answer_structured(state["query"], context)
    return {**state, "context": context, "answer": answer,
            "messages": [AIMessage(content=answer)]}

# 그래프 구성
memory = MemorySaver()
builder = StateGraph(LegalChatState)

builder.add_node("classify_domain", classify_domain)
builder.add_node("select_strategy", select_strategy)
builder.add_node("adaptive", lambda s: run_adaptive(s, retriever))
builder.add_node("self_rag", lambda s: run_self_rag(s, retriever))
builder.add_node("corrective", lambda s: run_corrective(s, retriever))

builder.set_entry_point("classify_domain")
builder.add_edge("classify_domain", "select_strategy")
builder.add_conditional_edges(
    "select_strategy",
    get_strategy_node,
    {"adaptive": "adaptive", "self_rag": "self_rag", "corrective": "corrective"},
)
for node in ["adaptive", "self_rag", "corrective"]:
    builder.add_edge(node, END)

legal_system = builder.compile(checkpointer=memory)
```

---

## 5. 전략별 성능 비교

```python
test_cases = [
    ("임대차보호법에서 묵시적 갱신이란?", "adaptive 예상"),
    ("세입자가 집주인에게 전세금 반환 청구 가능한가?", "self_rag 예상"),
    ("2024년 최저임금은 얼마인가?", "corrective 예상"),
]

print("전략 선택 테스트:")
for query, expected in test_cases:
    state = route_legal_query({"query": query, "strategy": "", "answer": "", "confidence": 0.0})
    match = "✅" if state["strategy"] in expected else "❌"
    print(f"{match} '{query[:30]}...' → {state['strategy']} ({expected})")
```

---

## 핵심 정리

**법률 상담 시스템에서의 전략 선택**:

| 법률 질문 유형 | RAG 전략 | 이유 |
|--------------|----------|------|
| 법률 용어 정의 | Adaptive | 단순 정보 조회 |
| 법적 요건 확인 | Self-RAG | 정확성 검증 필요 |
| 최신 법령/판례 | Corrective | 웹 검색 필요 |
| 복잡한 사례 분석 | Self-RAG | 다단계 검증 |
