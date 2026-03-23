# 5차시: ReAct 에이전트 & 다국어 RAG

## 학습 목표
- LangChain으로 ReAct 에이전트를 구현할 수 있다
- 다국어 RAG 아키텍처를 설계하고 구현할 수 있다
- 쿼리 언어를 감지하고 적절한 경로로 라우팅할 수 있다

---

## 1. ReAct 에이전트 구현

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# ReAct 프롬프트 (LangChain Hub)
prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 도구 리스트
tools = [search_korean_law, calculate_deadline, search_court_cases]

# 에이전트 생성
agent = create_react_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

result = executor.invoke({
    "input": "2022년 3월 15일에 발생한 교통사고의 민사 소멸시효는 언제인가요?"
})
print(result["output"])
```

---

## 2. create_tool_calling_agent (현대적 방식)

```python
from langchain.agents import create_tool_calling_agent

# Tool calling 지원 모델 (GPT-4, Claude 등)
agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

---

## 3. 언어 감지 및 라우팅

```python
from langdetect import detect
from langchain_core.runnables import RunnableLambda, RunnableBranch

def detect_language(inputs: dict) -> dict:
    text = inputs.get("question", "")
    lang = detect(text)
    return {**inputs, "language": lang}

# 언어별 다른 체인으로 라우팅
multilingual_chain = (
    RunnableLambda(detect_language)
    | RunnableBranch(
        (lambda x: x["language"] == "ko", korean_rag_chain),
        (lambda x: x["language"] == "en", english_rag_chain),
        (lambda x: x["language"] == "ja", japanese_rag_chain),
        default_chain  # 기타 언어
    )
)

result = multilingual_chain.invoke({"question": "청약 조건이 뭔가요?"})
```

---

## 4. 다국어 RAG 아키텍처

```
쿼리 (임의 언어)
    │
    ▼
[언어 감지] → 한국어/영어/일본어/...
    │
    ▼
[언어별 벡터저장소 선택]
  또는
[번역 → 한국어 → 검색 → 번역 → 결과]
    │
    ▼
[LLM 답변 생성 (원어로)]
```

```python
# 방식 1: 언어별 별도 벡터저장소
vectorstores = {
    "ko": FAISS.load_local("ko_index", embeddings),
    "en": FAISS.load_local("en_index", embeddings),
}

# 방식 2: 다국어 임베딩 (BAAI/bge-m3) - 단일 저장소
multilingual_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
unified_vectorstore = FAISS.from_documents(all_docs, multilingual_embeddings)
# 한국어/영어 쿼리 모두 처리 가능
```

---

## 💡 핵심 포인트
- `create_react_agent`: 텍스트 기반 ReAct (범용)
- `create_tool_calling_agent`: 도구 호출 기반 (GPT-4, Claude 전용, 더 신뢰성)
- `BAAI/bge-m3` 다국어 임베딩으로 단일 벡터저장소에서 다국어 검색 가능
- 라우팅은 `RunnableBranch` 또는 if/else 로직으로 구현

## ❓ 차시별 질문
1. ReAct와 Tool Calling 방식의 차이는 무엇인가요?
2. 번역 후 검색 vs 다국어 임베딩 방식의 장단점은?
