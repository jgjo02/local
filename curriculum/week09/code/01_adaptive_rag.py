"""Week 9 실습 1: Adaptive RAG 구현"""

import os
import json
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AdaptiveState(TypedDict):
    query: str
    query_type: str  # no_retrieval, single, multi, iterative
    context: str
    answer: str
    iteration: int

SAMPLE_DOCS = [
    "임대차보호법: 임차인은 계약 만료 후 묵시적 갱신이 가능합니다. 집주인은 계약 만료 6개월 전~1개월 전 사이에 계약 해지 또는 갱신 거절을 통보해야 합니다.",
    "최저임금법: 최저임금은 매년 최저임금위원회에서 결정합니다. 위반 시 3년 이하 징역 또는 2천만원 이하 벌금에 처합니다.",
    "근로기준법 제23조: 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉 등의 불이익 처우를 할 수 없습니다.",
    "민법 제390조: 채무자가 채무의 내용에 따른 이행을 하지 아니하는 때에는 채권자는 손해배상을 청구할 수 있습니다.",
]

def simple_retriever(query: str, k: int = 3) -> list:
    """간단한 키워드 기반 검색"""
    results = []
    query_words = set(query.split())
    scored = []
    for doc in SAMPLE_DOCS:
        doc_words = set(doc.split())
        score = len(query_words & doc_words)
        scored.append((score, doc))
    scored.sort(reverse=True)
    return [doc for _, doc in scored[:k] if _ > 0]


def classify_query(state: AdaptiveState) -> AdaptiveState:
    prompt = f"""질문을 분류하세요: {state['query']}
- no_retrieval: 일반 상식
- single: 단일 문서로 해결
- multi: 여러 문서 필요
- iterative: 여러 단계 검색 필요
분류명만 반환:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    qtype = response.content.strip().lower()
    if qtype not in ["no_retrieval", "single", "multi", "iterative"]:
        qtype = "single"
    return {**state, "query_type": qtype}


def direct_answer(state: AdaptiveState) -> AdaptiveState:
    response = llm.invoke([HumanMessage(content=state["query"])])
    return {**state, "answer": response.content, "context": ""}


def single_retrieve_generate(state: AdaptiveState) -> AdaptiveState:
    docs = simple_retriever(state["query"], k=2)
    context = "\n\n".join(docs) if docs else "관련 정보 없음"
    prompt = f"참고:\n{context}\n\n질문: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "answer": response.content}


def multi_retrieve_generate(state: AdaptiveState) -> AdaptiveState:
    # 쿼리 분해
    decompose = llm.invoke([HumanMessage(content=f"질문을 2개로 분해: {state['query']}\n각 줄에 하나씩:")])
    sub_queries = [q.strip() for q in decompose.content.split("\n") if q.strip()][:2]

    all_docs = []
    for sq in sub_queries:
        all_docs.extend(simple_retriever(sq, k=2))

    context = "\n\n".join(list(dict.fromkeys(all_docs))[:4])
    prompt = f"참고:\n{context}\n\n질문: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "answer": response.content}


def iterative_retrieve_generate(state: AdaptiveState) -> AdaptiveState:
    docs = simple_retriever(state["query"], k=3)
    context = "\n\n".join(docs) if docs else "관련 정보 없음"
    prompt = f"참고:\n{context}\n\n질문: {state['query']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "answer": response.content, "iteration": 1}


def get_strategy(state: AdaptiveState) -> str:
    return state["query_type"]


memory = MemorySaver()
builder = StateGraph(AdaptiveState)
builder.add_node("classify", classify_query)
builder.add_node("direct", direct_answer)
builder.add_node("single", single_retrieve_generate)
builder.add_node("multi", multi_retrieve_generate)
builder.add_node("iterative", iterative_retrieve_generate)

builder.set_entry_point("classify")
builder.add_conditional_edges(
    "classify",
    get_strategy,
    {
        "no_retrieval": "direct",
        "single": "single",
        "multi": "multi",
        "iterative": "iterative",
    }
)
for node in ["direct", "single", "multi", "iterative"]:
    builder.add_edge(node, END)

adaptive_rag = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        exit()

    test_cases = [
        "파이썬이란 무엇인가요?",
        "임대차 묵시적 갱신 조건은?",
        "계약 해지와 해고의 법적 절차를 비교해주세요",
    ]

    config = {"configurable": {"thread_id": "test_adaptive"}}
    for query in test_cases:
        result = adaptive_rag.invoke(
            {"query": query, "query_type": "", "context": "", "answer": "", "iteration": 0},
            config=config,
        )
        print(f"\n[{result['query_type']}] {query}")
        print(f"답변: {result['answer'][:200]}...")
