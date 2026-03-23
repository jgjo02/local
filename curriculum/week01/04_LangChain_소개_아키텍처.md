# 4차시: LangChain 소개 & 아키텍처

## 학습 목표
- LangChain v0.3 아키텍처 구조를 이해한다
- 주요 컴포넌트(Model I/O, Memory, Chains, Agents)를 설명할 수 있다
- ChatModel과 메시지 타입을 활용하여 대화를 구성할 수 있다
- Runnable 인터페이스의 invoke/batch/stream 메서드를 사용할 수 있다

---

## 1. LangChain이란?

### 1.1 탄생 배경

```
문제: LLM을 실제 앱에 통합하려면...
  - API 호출 코드 중복
  - 프롬프트 관리 어려움
  - 여러 도구 통합 복잡
  - 메모리/상태 관리 없음

해결: LangChain (2022년 말 Harrison Chase 개발)
  → LLM 앱 개발을 위한 표준 프레임워크
```

### 1.2 LangChain v0.3 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    langchain (상위 패키지)                │
│  Chains, Agents, 고수준 추상화                            │
└────────────────────────┬────────────────────────────────┘
                         │ depends on
┌────────────────────────▼────────────────────────────────┐
│                   langchain-core                         │
│  Runnable, BaseLanguageModel, BasePromptTemplate        │
│  BaseRetriever, BaseMemory (핵심 추상 클래스)             │
└──────────┬──────────────────────────────────────────────┘
           │                       │
┌──────────▼──────────┐  ┌────────▼────────────────────────┐
│  langchain-openai   │  │  langchain-community            │
│  langchain-anthropic│  │  600+ 서드파티 통합              │
│  langchain-google   │  │  (Chroma, FAISS, etc.)          │
└─────────────────────┘  └─────────────────────────────────┘
```

---

## 2. 주요 컴포넌트

### 2.1 Model I/O

```
LLM (텍스트 입출력)            ChatModel (메시지 입출력)
┌──────────────┐               ┌──────────────────────────┐
│ Input: str   │               │ Input: List[BaseMessage] │
│ Output: str  │               │ Output: AIMessage        │
└──────────────┘               └──────────────────────────┘
                                ↑ 현재 표준 (GPT-4, Claude)
```

**메시지 타입:**
```python
from langchain_core.messages import (
    SystemMessage,    # 시스템 역할 정의
    HumanMessage,     # 사용자 입력
    AIMessage,        # AI 응답
    ToolMessage,      # 도구 실행 결과
    FunctionMessage   # 함수 호출 결과 (레거시)
)
```

### 2.2 Prompts (프롬프트 템플릿)

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role} 전문가입니다."),
    ("human", "{question}")
])

# 변수 채우기
formatted = prompt.format_messages(
    role="파이썬",
    question="리스트 컴프리헨션이란?"
)
```

### 2.3 Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,      # 텍스트 그대로 반환
    JsonOutputParser,     # JSON 파싱
    PydanticOutputParser  # Pydantic 모델로 파싱
)
```

---

## 3. Runnable 인터페이스

모든 LangChain 컴포넌트가 구현하는 표준 인터페이스:

### 3.1 핵심 메서드

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. invoke: 단일 입력 처리
response = llm.invoke("파이썬이란?")

# 2. batch: 여러 입력 병렬 처리
responses = llm.batch(["파이썬?", "자바?", "러스트?"])

# 3. stream: 스트리밍 출력
for chunk in llm.stream("긴 답변을 해줘"):
    print(chunk.content, end="", flush=True)

# 4. ainvoke / astream: 비동기 버전
import asyncio
response = asyncio.run(llm.ainvoke("질문"))
```

### 3.2 LCEL Pipe 연산자 (|)

```
chain = prompt | llm | output_parser

┌──────┐     ┌─────┐     ┌──────────────┐
│Prompt│ ──▶ │ LLM │ ──▶ │OutputParser  │
└──────┘     └─────┘     └──────────────┘
  .invoke()    .invoke()     .invoke()

→ chain.invoke({"role": "파이썬", "question": "lambda란?"})
```

---

## 4. LangChain 실전 구성 예시

### 4.1 기본 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 컴포넌트 준비
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("human", "{input}")
])
parser = StrOutputParser()

# LCEL로 체인 구성
chain = prompt | llm | parser

# 실행
result = chain.invoke({"input": "LangChain이란?"})
print(result)
```

### 4.2 배치 처리

```python
questions = [
    {"input": "파이썬이란?"},
    {"input": "LangChain이란?"},
    {"input": "RAG란?"},
]

results = chain.batch(questions)
for q, r in zip(questions, results):
    print(f"Q: {q['input']}\nA: {r[:100]}...\n")
```

---

## 5. LangSmith 트레이싱

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 이후 모든 체인 실행이 자동으로 LangSmith에 기록됨
result = chain.invoke({"input": "테스트 질문"})
# → https://smith.langchain.com 에서 확인 가능
```

---

## 💡 핵심 포인트
- LangChain v0.3은 `langchain-core` / `langchain-community` / `langchain`으로 분리됨
- 모든 컴포넌트는 `Runnable` 인터페이스 구현 (invoke/batch/stream)
- LCEL의 `|` 연산자로 컴포넌트를 파이프라인처럼 연결
- `ChatModel`이 현재 표준 (LLM 클래스는 레거시)

## ⚠️ 주의사항
- v0.1/v0.2의 `LLMChain`, `ConversationalRetrievalChain` 등은 deprecated
- v0.3에서는 LCEL 방식으로 작성할 것
- `langchain-openai`와 `langchain-anthropic`은 별도 패키지 설치 필요

## 🔍 심화학습
- [LangChain 공식 문서](https://python.langchain.com/docs/introduction/)
- [LCEL 가이드](https://python.langchain.com/docs/expression_language/)

## ❓ 차시별 질문
1. LCEL의 `|` 연산자가 Python의 어떤 기능을 활용하는지 알아보세요
2. `invoke`와 `batch`의 성능 차이를 실험해보세요
3. `StrOutputParser` 없이 체인을 실행하면 어떻게 되나요?
