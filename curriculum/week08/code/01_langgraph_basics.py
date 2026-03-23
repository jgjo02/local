"""Week 8 실습 1: LangGraph 기초"""

import os
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ── 1. 가장 기본적인 그래프 ────────────────────────────────────────────────────

class SimpleState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def chat_node(state: SimpleState) -> SimpleState:
    """단순 챗봇 노드"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(SimpleState)
builder.add_node("chat", chat_node)
builder.set_entry_point("chat")
builder.add_edge("chat", END)
simple_graph = builder.compile()

def demo_simple_graph():
    print("=== 단순 그래프 ===")
    result = simple_graph.invoke({
        "messages": [HumanMessage(content="LangGraph가 뭔가요?")]
    })
    print(f"답변: {result['messages'][-1].content[:200]}")


# ── 2. 메모리가 있는 그래프 ────────────────────────────────────────────────────

memory = MemorySaver()
memory_graph = builder.compile(checkpointer=memory)

def demo_memory_graph():
    print("\n=== 메모리 그래프 ===")
    config = {"configurable": {"thread_id": "thread_001"}}

    r1 = memory_graph.invoke(
        {"messages": [HumanMessage(content="내 이름은 LangChain 개발자입니다.")]},
        config=config,
    )
    print(f"응답 1: {r1['messages'][-1].content[:150]}")

    r2 = memory_graph.invoke(
        {"messages": [HumanMessage(content="제 직업이 뭐라고 했나요?")]},
        config=config,
    )
    print(f"응답 2: {r2['messages'][-1].content[:150]}")


# ── 3. 조건부 엣지가 있는 그래프 ──────────────────────────────────────────────

class RouterState(TypedDict):
    query: str
    category: str
    answer: str

def classify_node(state: RouterState) -> RouterState:
    """쿼리 분류"""
    prompt = f"""다음 질문을 분류하세요: {state['query']}
분류: technical(기술), general(일반), off_topic(무관)
분류만 반환:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()
    if category not in ["technical", "general", "off_topic"]:
        category = "general"
    return {**state, "category": category}

def get_route(state: RouterState) -> str:
    return state["category"]

def technical_node(state: RouterState) -> RouterState:
    response = llm.invoke([
        HumanMessage(content=f"기술 전문가로서 답변: {state['query']}")
    ])
    return {**state, "answer": response.content}

def general_node(state: RouterState) -> RouterState:
    response = llm.invoke([HumanMessage(content=state["query"])])
    return {**state, "answer": response.content}

def off_topic_node(state: RouterState) -> RouterState:
    return {**state, "answer": "관련 없는 주제입니다. 기술적 질문을 해주세요."}

router_builder = StateGraph(RouterState)
router_builder.add_node("classify", classify_node)
router_builder.add_node("technical", technical_node)
router_builder.add_node("general", general_node)
router_builder.add_node("off_topic", off_topic_node)

router_builder.set_entry_point("classify")
router_builder.add_conditional_edges(
    "classify",
    get_route,
    {
        "technical": "technical",
        "general": "general",
        "off_topic": "off_topic",
    }
)
for node in ["technical", "general", "off_topic"]:
    router_builder.add_edge(node, END)

router_graph = router_builder.compile()

def demo_router_graph():
    print("\n=== 라우터 그래프 ===")
    queries = [
        "FAISS와 Chroma의 차이점은?",
        "오늘 날씨가 좋네요",
        "점심 뭐 먹을까요?",
    ]
    for q in queries:
        result = router_graph.invoke({
            "query": q, "category": "", "answer": ""
        })
        print(f"[{result['category']}] {q[:30]}...")
        print(f"  → {result['answer'][:100]}\n")


# ── 4. 그래프 시각화 (Mermaid) ────────────────────────────────────────────────

def visualize_graph(graph):
    """그래프 Mermaid 다이어그램 출력"""
    try:
        print("\n=== 그래프 구조 ===")
        print(graph.get_graph().draw_mermaid())
    except Exception:
        print("(시각화 불가)")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        exit()

    demo_simple_graph()
    demo_memory_graph()
    demo_router_graph()

    print("\n=== 라우터 그래프 구조 ===")
    visualize_graph(router_graph)
