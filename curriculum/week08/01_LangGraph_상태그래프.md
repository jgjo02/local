# 1차시: LangGraph StateGraph 핵심 개념

## 학습 목표
- LangGraph의 StateGraph 구조를 이해한다
- 노드, 엣지, 상태를 정의하고 연결할 수 있다
- 그래프를 컴파일하고 실행할 수 있다

---

## 1. LangGraph 소개

```
LangChain (체인 중심) → 선형 파이프라인
LangGraph (그래프 중심) → 순환, 분기, 조건부 흐름 지원

언제 LangGraph?
  - 복잡한 에이전트 워크플로우
  - 순환 (반복 검색)
  - 조건부 분기 (질문 유형별 처리)
  - 상태 유지 (대화 이력)
  - 병렬 처리
```

---

## 2. 기본 StateGraph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    documents: list[str]
    answer: str
    iteration: int


def retrieve_node(state: GraphState) -> dict:
    """검색 노드"""
    query = state["query"]
    docs = retriever.invoke(query)
    return {"documents": [d.page_content for d in docs]}


def generate_node(state: GraphState) -> dict:
    """답변 생성 노드"""
    context = "\n".join(state["documents"])
    answer = llm.invoke(f"컨텍스트: {context}\n질문: {state['query']}")
    return {"answer": answer.content}


def route_after_generate(state: GraphState) -> str:
    """조건부 엣지: 답변 품질에 따라 라우팅"""
    if "알 수 없습니다" in state["answer"] and state["iteration"] < 3:
        return "retry"
    return "done"


# 그래프 구성
graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges(
    "generate",
    route_after_generate,
    {"retry": "retrieve", "done": END}
)

app = graph.compile()

# 실행
result = app.invoke({
    "query": "청약 1순위 조건은?",
    "documents": [],
    "answer": "",
    "iteration": 0
})
print(result["answer"])
```

---

## 3. 그래프 시각화

```python
# Mermaid 다이어그램 출력
print(app.get_graph().draw_mermaid())

# IPython에서 시각화
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

## 4. 스트리밍 실행

```python
# 각 노드 실행마다 중간 결과 스트리밍
for event in app.stream({"query": "청약 조건"}):
    for node_name, state in event.items():
        print(f"[{node_name}] 실행 완료")
        print(f"  상태: {list(state.keys())}")
```

---

## 💡 핵심 포인트
- `TypedDict`로 그래프 상태 스키마 명확히 정의
- `add_messages` reducer로 메시지 누적 관리
- 조건부 엣지(`add_conditional_edges`)로 동적 라우팅
- `START`, `END`는 LangGraph 내장 특수 노드

## ❓ 차시별 질문
1. StateGraph와 MessageGraph의 차이는?
2. 무한 루프 방지를 위해 어떤 설계가 필요한가요?
