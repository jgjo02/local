# Week 8-3: LangGraph 메모리 및 체크포인트

## 학습 목표
- MemorySaver를 활용한 상태 지속성 구현
- 멀티턴 대화에서 컨텍스트 유지
- Human-in-the-Loop 패턴 이해

---

## 1. MemorySaver 기본

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class ConvState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str

# MemorySaver 초기화
memory = MemorySaver()

def chat_node(state: ConvState) -> ConvState:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(ConvState)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

# 체크포인터와 함께 컴파일
graph = builder.compile(checkpointer=memory)

# 세션 기반 대화
config = {"configurable": {"thread_id": "session_001"}}

r1 = graph.invoke(
    {"messages": [HumanMessage(content="내 이름은 김철수야")], "session_id": "001"},
    config=config,
)
print(r1["messages"][-1].content)

r2 = graph.invoke(
    {"messages": [HumanMessage(content="내 이름이 뭐라고 했지?")], "session_id": "001"},
    config=config,
)
print(r2["messages"][-1].content)  # "김철수" 기억
```

---

## 2. 상태 스냅샷 조회

```python
# 현재 상태 확인
snapshot = graph.get_state(config)
print(f"메시지 수: {len(snapshot.values['messages'])}")

# 특정 체크포인트로 복원
history = list(graph.get_state_history(config))
for state in history[:3]:
    print(f"Step {state.metadata.get('step', '?')}: {len(state.values['messages'])} messages")
```

---

## 3. Human-in-the-Loop

### 3.1 interrupt_before 설정

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str

def create_plan(state: ApprovalState) -> ApprovalState:
    plan = f"계획: {state['task']}을 위해 다음 단계를 수행합니다..."
    return {**state, "plan": plan}

def execute_plan(state: ApprovalState) -> ApprovalState:
    result = f"실행 결과: {state['plan']} - 완료"
    return {**state, "result": result}

memory = MemorySaver()
builder = StateGraph(ApprovalState)
builder.add_node("plan", create_plan)
builder.add_node("execute", execute_plan)
builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_edge("execute", END)

# execute 노드 실행 전 중단
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["execute"]
)

config = {"configurable": {"thread_id": "approval_001"}}

# 1단계: 계획 생성 후 중단
state = graph.invoke(
    {"task": "데이터베이스 마이그레이션", "plan": "", "approved": False, "result": ""},
    config=config,
)
print("계획:", state["plan"])
print("승인을 기다리는 중...")

# 2단계: 사람이 승인 후 재개
user_approval = input("계획을 승인하시겠습니까? (y/n): ")
if user_approval.lower() == "y":
    # 중단된 지점에서 재개
    final_state = graph.invoke(None, config=config)
    print("실행 완료:", final_state["result"])
else:
    print("취소되었습니다.")
```

### 3.2 interrupt_after 패턴

```python
# 노드 실행 후 중단 (검토 후 계속 or 수정)
graph_after = builder.compile(
    checkpointer=memory,
    interrupt_after=["plan"],
)

state = graph_after.invoke({"task": "서버 재시작"}, config)

# 계획 수정
current = graph_after.get_state(config)
modified_plan = "수정된 계획: 점검 창 이후에 서버를 재시작합니다."
graph_after.update_state(
    config,
    {"plan": modified_plan}
)

# 수정된 상태로 계속 실행
graph_after.invoke(None, config)
```

---

## 4. 멀티에이전트 메모리 공유

```python
from typing import TypedDict, Annotated, List
import operator

class SharedMemoryState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    agent_notes: Annotated[List[str], operator.add]

def researcher_agent(state: SharedMemoryState) -> SharedMemoryState:
    """연구 에이전트: 정보 수집"""
    note = f"연구 결과: {state['messages'][-1].content}에 관한 정보..."
    return {"agent_notes": [f"[연구] {note}"]}

def analyst_agent(state: SharedMemoryState) -> SharedMemoryState:
    """분석 에이전트: 정보 분석"""
    research = "\n".join(state["agent_notes"])
    note = f"분석: {research}를 바탕으로..."
    return {"agent_notes": [f"[분석] {note}"]}

def summarizer_agent(state: SharedMemoryState) -> SharedMemoryState:
    """요약 에이전트: 최종 정리"""
    all_notes = "\n".join(state["agent_notes"])
    summary = f"종합 요약:\n{all_notes}"
    return {"messages": [AIMessage(content=summary)]}

memory = MemorySaver()
builder = StateGraph(SharedMemoryState)
builder.add_node("researcher", researcher_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("summarizer", summarizer_agent)

builder.set_entry_point("researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "summarizer")
builder.add_edge("summarizer", END)

multi_agent_graph = builder.compile(checkpointer=memory)
```

---

## 5. 상태 영속성 (SQLite 체크포인터)

```python
# 프로덕션에서는 SQLite나 PostgreSQL 체크포인터 사용
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
persistent_memory = SqliteSaver(conn)

graph_persistent = builder.compile(checkpointer=persistent_memory)

# 재시작 후에도 이전 대화 유지
config = {"configurable": {"thread_id": "persistent_session_001"}}
result = graph_persistent.invoke(
    {"messages": [HumanMessage(content="안녕하세요")]},
    config=config,
)
```

---

## 핵심 정리

| 기능 | 클래스/메서드 | 용도 |
|------|-------------|------|
| 인메모리 저장 | `MemorySaver()` | 개발/테스트 |
| 실행 전 중단 | `interrupt_before=["node"]` | 승인 워크플로우 |
| 실행 후 중단 | `interrupt_after=["node"]` | 검토 후 수정 |
| 상태 조회 | `get_state(config)` | 현재 상태 확인 |
| 상태 수정 | `update_state(config, updates)` | 사람이 상태 편집 |
| 재개 | `invoke(None, config)` | 중단점에서 계속 |
