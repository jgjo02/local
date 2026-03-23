# Week 8-5: LangGraph 법률 상담 시스템 설계

## 학습 목표
- Project 3 법률 상담 시스템 아키텍처 이해
- Adaptive RAG + LangGraph 통합
- 멀티그래프 전략 선택 패턴

---

## 1. 법률 상담 시스템 개요

### 1.1 시스템 아키텍처

```
사용자 질문
    ↓
쿼리 분류기 (법률 도메인 확인)
    ↓
RAG 전략 선택기
├── Adaptive RAG (일반 법률 질문)
├── Self-RAG (검증이 필요한 질문)
└── Corrective RAG (복잡한 법률 해석)
    ↓
법률 문서 검색 (하이브리드)
    ↓
답변 생성 + 법적 면책조항
    ↓
사용자 응답
```

### 1.2 법률 도메인 특수 고려사항

```python
LEGAL_DOMAINS = {
    "민사법": ["계약", "손해배상", "재산권", "채권"],
    "노동법": ["해고", "임금", "근로계약", "직장내괴롭힘"],
    "형사법": ["범죄", "고소", "고발", "형사절차"],
    "부동산": ["임대차", "전세", "분쟁", "명도"],
}

DISCLAIMER = """
⚠️ 법률 면책조항: 이 정보는 교육 목적으로 제공되며 법적 조언이 아닙니다.
실제 법적 문제는 반드시 변호사와 상담하시기 바랍니다.
"""
```

---

## 2. 쿼리 분류 및 전략 선택

```python
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class LegalState(TypedDict):
    query: str
    domain: str
    rag_strategy: Literal["adaptive", "self", "corrective"]
    retrieved_docs: list
    answer: str
    confidence: float
    requires_verification: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classify_legal_query(state: LegalState) -> LegalState:
    """법률 도메인 및 복잡도 분류"""
    prompt = f"""법률 질문을 분류하세요:

질문: {state['query']}

분류 결과를 JSON으로:
{{
  "domain": "민사법/노동법/형사법/부동산",
  "complexity": "simple/medium/complex",
  "requires_verification": true/false
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    import json
    try:
        data = json.loads(response.content)
        domain = data.get("domain", "민사법")
        complexity = data.get("complexity", "simple")
        requires_verification = data.get("requires_verification", False)
    except:
        domain = "민사법"
        complexity = "simple"
        requires_verification = False

    # 전략 결정
    if complexity == "complex":
        strategy = "corrective"
    elif requires_verification:
        strategy = "self"
    else:
        strategy = "adaptive"

    return {
        **state,
        "domain": domain,
        "rag_strategy": strategy,
        "requires_verification": requires_verification,
    }


def select_rag_strategy(state: LegalState) -> str:
    """전략에 따라 다음 노드 결정"""
    return state["rag_strategy"]
```

---

## 3. 법률 문서 검색

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_legal_retriever(docs_path: str):
    """법률 문서 하이브리드 검색기"""
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = TextLoader(docs_path, encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_ret = faiss_store.as_retriever(search_kwargs={"k": 5})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5

    return EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[0.3, 0.7],
    )


def retrieve_legal_docs(state: LegalState, retriever) -> LegalState:
    """법률 문서 검색"""
    query = f"{state['domain']} {state['query']}"
    docs = retriever.invoke(query)
    return {**state, "retrieved_docs": docs}
```

---

## 4. 답변 생성 및 검증

```python
LEGAL_SYSTEM_PROMPT = """당신은 법률 정보 제공 AI입니다.

## 제공 원칙
1. 검색된 법률 조문과 판례를 근거로 답변
2. 불확실한 내용은 반드시 확실하지 않음을 표시
3. 법적 조언이 아닌 정보 제공임을 항상 명시

## 참고 법률 자료
{context}"""

def generate_legal_answer(state: LegalState) -> LegalState:
    """법률 답변 생성"""
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"][:3]])

    messages = [
        HumanMessage(content=LEGAL_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=state["query"]),
    ]

    response = llm.invoke(messages)
    answer = response.content + "\n\n" + DISCLAIMER

    return {**state, "answer": answer, "confidence": 0.8}


def verify_answer(state: LegalState) -> LegalState:
    """답변 검증 (Self-RAG용)"""
    verify_prompt = f"""다음 법률 답변을 검증하세요:

질문: {state['query']}
답변: {state['answer'][:500]}

다음 기준으로 평가:
- 법적 근거가 있는가?
- 사실과 다른 내용은 없는가?

검증 결과 (pass/fail):"""

    response = llm.invoke([HumanMessage(content=verify_prompt)])
    verified = "pass" in response.content.lower()
    confidence = 0.9 if verified else 0.5

    return {**state, "confidence": confidence}
```

---

## 5. 완성된 법률 상담 그래프

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
builder = StateGraph(LegalState)

builder.add_node("classify", classify_legal_query)
builder.add_node("retrieve", lambda s: retrieve_legal_docs(s, retriever))
builder.add_node("generate", generate_legal_answer)
builder.add_node("verify", verify_answer)

builder.set_entry_point("classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "generate")

builder.add_conditional_edges(
    "classify",
    select_rag_strategy,
    {
        "adaptive": "retrieve",
        "self": "retrieve",
        "corrective": "retrieve",
    }
)

builder.add_conditional_edges(
    "generate",
    lambda s: "verify" if s["requires_verification"] else END,
    {"verify": "verify", END: END},
)
builder.add_edge("verify", END)

legal_graph = builder.compile(checkpointer=memory)
```

---

## 핵심 정리

**Project 3 설계 원칙**:
1. **도메인 특화 분류**: 법률 도메인별 전문화된 검색
2. **전략적 RAG 선택**: 쿼리 복잡도에 맞는 RAG 전략
3. **검증 의무화**: 법률 정보의 정확성 검증
4. **면책조항**: 법적 조언이 아님을 명시
5. **소스 추적**: 어떤 법조문을 근거로 했는지 제시
