# Week 6-3: RAG 체인 설계 및 프롬프트 최적화

## 학습 목표
- ETF 추천에 특화된 RAG 체인 설계
- 금융 도메인 시스템 프롬프트 작성
- 컨텍스트 포맷팅 전략 이해

---

## 1. ETF 추천 시스템 프롬프트

### 1.1 도메인 전문가 페르소나

```python
SYSTEM_PROMPT = """당신은 10년 경력의 ETF 투자 전문 상담사입니다.

## 전문 역량
- 국내외 ETF 상품 특성 및 위험 구조 이해
- 고객 성향별 맞춤 포트폴리오 설계 능력
- 시장 동향과 ETF 성과 분석 능력

## 상담 원칙
1. 고객 우선: 위험 허용도와 투자 목표를 항상 먼저 파악
2. 분산 투자: 단일 ETF보다 포트폴리오 관점 제안
3. 투명성: 수수료, 위험, 한계점을 솔직하게 설명
4. 법적 준수: 투자 권유가 아닌 정보 제공 관점 유지

## 응답 형식
- 추천 ETF: 이름, 티커, 선택 이유
- 위험도: 저위험/중립/고위험 명시
- 포트폴리오 비중 제안 (요청 시)
- 주의사항 반드시 포함

## 참고 ETF 정보
{context}

위 정보 외의 ETF는 언급하지 마세요."""
```

### 1.2 컨텍스트 포맷팅

```python
def format_context(docs: list, max_docs: int = 5) -> str:
    """검색 결과를 프롬프트용 컨텍스트로 변환"""
    if not docs:
        return "관련 ETF를 찾지 못했습니다."

    parts = []
    for i, doc in enumerate(docs[:max_docs], 1):
        meta = doc.metadata
        parts.append(f"""[ETF {i}: {meta.get('name', '')}]
{doc.page_content}
---""")

    return "\n\n".join(parts)
```

---

## 2. LCEL RAG 체인 구현

### 2.1 기본 RAG 체인

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def retrieve_and_format(query: str) -> str:
    docs = ensemble_retriever.invoke(query)
    return format_context(docs)

rag_chain = (
    {
        "context": lambda x: retrieve_and_format(x["question"]),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 실행
answer = rag_chain.invoke({"question": "안전한 ETF 추천해주세요"})
```

### 2.2 소스 포함 체인

```python
from langchain_core.runnables import RunnableParallel

# 답변과 소스를 동시에 반환
chain_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=lambda x: [
        doc.metadata.get("name", "")
        for doc in ensemble_retriever.invoke(x["question"])[:3]
    ],
)

result = chain_with_sources.invoke({"question": "반도체 ETF 추천"})
print(result["answer"])
print("참고 ETF:", result["sources"])
```

---

## 3. 대화 이력 관리

### 3.1 RunnableWithMessageHistory 적용

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chat_chain = CHAT_PROMPT | llm | StrOutputParser()

session_histories = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

chain_with_history = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 연속 대화
config = {"configurable": {"session_id": "user_001"}}

r1 = chain_with_history.invoke(
    {"context": context, "question": "ETF 입문자에게 맞는 상품은?"},
    config=config,
)

r2 = chain_with_history.invoke(
    {"context": context, "question": "방금 추천해준 ETF의 수수료는 얼마인가요?"},
    config=config,
)
```

---

## 4. 스트리밍 응답

```python
import gradio as gr

def stream_response(message: str, history: list):
    """Gradio 스트리밍 응답"""
    docs = ensemble_retriever.invoke(message)
    context = format_context(docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": message},
    ]

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial

demo = gr.ChatInterface(
    fn=stream_response,
    title="ETF 상담 챗봇",
)
demo.launch()
```

---

## 5. 위험도 기반 필터링 체인

```python
def build_risk_aware_chain(risk_profile: str):
    """위험성향을 반영한 맞춤형 체인"""

    risk_context = {
        "저위험": "고객은 원금 보존을 최우선으로 합니다. 채권, 단기채 위주로 추천하세요.",
        "중립": "고객은 적정 수익과 리스크 균형을 원합니다.",
        "고위험": "고객은 높은 수익을 위해 변동성을 수용합니다. 섹터, 해외 성장주 위주로 추천 가능합니다.",
    }

    risk_note = risk_context.get(risk_profile, "")

    enhanced_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + f"\n\n## 고객 위험 성향\n{risk_note}"),
        ("human", "{question}"),
    ])

    return enhanced_prompt | llm | StrOutputParser()
```

---

## 핵심 정리

```
사용자 쿼리
    ↓
하이브리드 검색 (BM25 + FAISS)
    ↓
위험도 필터링
    ↓
컨텍스트 포맷팅
    ↓
시스템 프롬프트 + 대화 이력 + 쿼리
    ↓
LLM 응답 생성 (스트리밍)
    ↓
최종 ETF 추천 답변
```

**프롬프트 최적화 체크리스트**:
- [ ] 도메인 전문가 역할 명확히 정의
- [ ] 응답 형식 구체적으로 지정
- [ ] 제공된 데이터 범위 내에서만 답변하도록 제한
- [ ] 법적/윤리적 주의사항 포함
- [ ] 위험 고지 의무 반영
