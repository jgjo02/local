# Week 8-4: LangGraph 멀티에이전트 시스템

## 학습 목표
- 멀티에이전트 아키텍처 패턴 이해
- Supervisor 패턴으로 에이전트 오케스트레이션
- 에이전트 간 통신 및 핸드오프 구현

---

## 1. 멀티에이전트 아키텍처

### 1.1 패턴 비교

| 패턴 | 구조 | 특징 |
|------|------|------|
| Sequential | A → B → C | 순차 실행, 간단 |
| Supervisor | Supervisor → (A/B/C) | 중앙 조율, 유연 |
| Network | 각 에이전트가 서로 호출 | 복잡, 강력 |

### 1.2 Supervisor 패턴

```
사용자 질문
    ↓
Supervisor (라우터)
├── Research Agent (검색, 정보 수집)
├── Analysis Agent (분석, 계산)
├── Writing Agent (문서 작성)
└── END (완료 판단)
```

---

## 2. Supervisor 패턴 구현

```python
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 멤버 에이전트 목록
MEMBERS = ["researcher", "analyst", "writer"]
OPTIONS = MEMBERS + ["FINISH"]

class SupervisorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    task_complete: bool

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor: 다음 에이전트 결정"""
    system_prompt = f"""당신은 팀 매니저입니다. 팀원: {', '.join(MEMBERS)}
현재까지 작업 내용을 보고 다음 행동을 결정하세요.

규칙:
- 정보 수집이 필요하면: researcher
- 수집된 정보 분석이 필요하면: analyst
- 최종 보고서 작성이 필요하면: writer
- 모든 작업이 완료되었으면: FINISH

다음 행동 (단어만):"""

    messages = state["messages"] + [HumanMessage(content=system_prompt)]
    response = llm.invoke(messages)
    next_agent = response.content.strip().lower()

    if next_agent not in OPTIONS:
        next_agent = "FINISH"

    return {**state, "next_agent": next_agent}


def get_next(state: SupervisorState) -> str:
    """다음 노드 결정"""
    next_agent = state["next_agent"]
    if next_agent == "finish":
        return END
    return next_agent


def make_agent(name: str, system_prompt: str):
    """에이전트 노드 팩토리"""
    def agent_node(state: SupervisorState) -> SupervisorState:
        agent_llm = ChatOpenAI(model="gpt-4o-mini")
        messages = [HumanMessage(content=system_prompt)] + state["messages"]
        response = agent_llm.invoke(messages)
        result = AIMessage(content=f"[{name}] {response.content}")
        return {"messages": [result]}
    return agent_node


# 에이전트 생성
researcher = make_agent(
    "researcher",
    "당신은 정보 수집 전문가입니다. 주어진 주제에 대해 관련 정보를 찾고 정리하세요."
)
analyst = make_agent(
    "analyst",
    "당신은 데이터 분석가입니다. 수집된 정보를 분석하고 인사이트를 도출하세요."
)
writer = make_agent(
    "writer",
    "당신은 전문 작가입니다. 분석 결과를 바탕으로 명확하고 구조화된 보고서를 작성하세요."
)

# 그래프 구성
memory = MemorySaver()
builder = StateGraph(SupervisorState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher)
builder.add_node("analyst", analyst)
builder.add_node("writer", writer)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    get_next,
    {agent: agent for agent in MEMBERS} | {END: END},
)

for agent in MEMBERS:
    builder.add_edge(agent, "supervisor")

multi_agent = builder.compile(checkpointer=memory)
```

---

## 3. 에이전트 도구 통합

```python
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_database(query: str) -> str:
    """데이터베이스에서 정보를 검색합니다."""
    return f"DB 검색 결과: {query}..."

@tool
def calculate_stats(data: str) -> str:
    """통계 계산을 수행합니다."""
    return f"통계 결과: {data}..."

@tool
def write_report(content: str, title: str) -> str:
    """보고서를 작성합니다."""
    return f"보고서 '{title}'이 작성되었습니다."

# 도구를 가진 에이전트 생성
def make_tool_agent(tools: list, system_prompt: str):
    """도구가 있는 ReAct 에이전트 생성"""
    agent = create_react_agent(llm, tools)
    def node(state):
        result = agent.invoke({
            "messages": [HumanMessage(content=system_prompt)] + state["messages"]
        })
        return {"messages": [result["messages"][-1]]}
    return node
```

---

## 4. 실행 및 모니터링

```python
config = {"configurable": {"thread_id": "multi_agent_001"}}

result = multi_agent.invoke(
    {
        "messages": [HumanMessage(content="2024년 한국 AI 시장 현황 보고서를 작성해주세요")],
        "next_agent": "",
        "task_complete": False,
    },
    config=config,
)

print("최종 보고서:")
for msg in result["messages"]:
    if isinstance(msg, AIMessage):
        print(f"\n{msg.content[:300]}...")

# 실행 이력 조회
state_history = list(multi_agent.get_state_history(config))
print(f"\n총 {len(state_history)} 단계 실행")
```

---

## 5. 스트리밍 실행

```python
for step in multi_agent.stream(
    {"messages": [HumanMessage(content="AI 트렌드 분석")], "next_agent": "", "task_complete": False},
    config=config,
    stream_mode="values",
):
    latest_msg = step["messages"][-1] if step["messages"] else None
    if latest_msg and isinstance(latest_msg, AIMessage):
        print(f"✅ {latest_msg.content[:100]}...")
```

---

## 핵심 정리

**Supervisor 패턴 장단점**:
- ✅ 동적 작업 분배
- ✅ 각 에이전트의 전문화
- ✅ 쉬운 에이전트 추가/제거
- ⚠️ Supervisor의 라우팅 품질이 전체 성능 결정
- ⚠️ 긴 대화에서 컨텍스트 증가
