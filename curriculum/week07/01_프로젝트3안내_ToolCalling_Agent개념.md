# 1차시: [프로젝트 3] 안내 + Tool Calling (Function Calling) + 에이전트(Agent) 개념

## 📌 학습 목표

이번 차시를 마치면 다음을 할 수 있습니다:
1. 프로젝트 3(법률 자문 에이전트)의 전체 구조와 요구사항을 이해한다
2. Tool Calling(Function Calling)의 작동 원리를 API 수준에서 설명할 수 있다
3. JSON Schema로 도구를 정의하고 LangChain Tool 인터페이스와 연결할 수 있다
4. 에이전트(Agent)의 개념과 ReAct 패러다임을 이해하고 구현할 수 있다

---

## 🚀 프로젝트 3: 법률 자문 에이전트

### 프로젝트 소개

법률 자문 에이전트는 한국 법률 시스템에 특화된 AI 어시스턴트입니다. 일반인이 법률 문제에 직면했을 때 전문가 수준의 정보를 신속하게 제공하는 것을 목표로 합니다.

#### 프로젝트 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    법률 자문 에이전트 시스템                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   사용자 입력                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐    ┌──────────────────────────────────────┐   │
│  │  언어   │    │           ReAct 에이전트              │   │
│  │  감지   │───▶│  Thought → Action → Observation 루프  │   │
│  └─────────┘    └──────────────────────────────────────┘   │
│                              │                              │
│              ┌───────────────┼───────────────┐             │
│              ▼               ▼               ▼             │
│         ┌─────────┐   ┌─────────┐   ┌──────────┐          │
│         │법령검색  │   │판례검색  │   │ 웹검색   │          │
│         │  도구   │   │  도구   │   │   도구   │          │
│         └─────────┘   └─────────┘   └──────────┘          │
│              │               │               │             │
│              ▼               ▼               ▼             │
│         ┌─────────┐   ┌─────────┐   ┌──────────┐          │
│         │법제처API│   │종합법률  │   │ Tavily   │          │
│         │         │   │정보API  │   │  Search  │          │
│         └─────────┘   └─────────┘   └──────────┘          │
│                                                             │
│                   최종 법률 자문 답변                        │
└─────────────────────────────────────────────────────────────┘
```

#### 주요 기능 요구사항

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| 법령 조문 검색 | 특정 법률의 조문을 실시간으로 검색 | 필수 |
| 관련 판례 검색 | 유사 사건의 판례를 검색하여 제공 | 필수 |
| 웹 검색 통합 | 최신 법률 해석 및 뉴스 검색 | 필수 |
| 법률 기간 계산 | 소멸시효, 제척기간 등 계산 | 권장 |
| 다국어 지원 | 한국어/영어 질문 처리 | 권장 |
| 대화 이력 관리 | 이전 상담 내용을 참조한 연속 상담 | 고급 |

---

## 🔧 Tool Calling (Function Calling) 개념

### 1. Tool Calling의 등장 배경

초기 LLM은 텍스트 생성만 가능했습니다. 하지만 실제 업무에서는 다음과 같은 한계가 있었습니다:

- **실시간 정보 부재**: 학습 데이터 이후의 정보를 모름
- **계산 능력 제한**: 복잡한 수학 계산 오류 발생
- **외부 시스템 접근 불가**: 데이터베이스, API 조회 불가
- **구조화된 출력 어려움**: 정확한 JSON/XML 형식 보장 어려움

Tool Calling(Function Calling)은 이러한 한계를 극복하기 위해 2023년 OpenAI가 도입한 기능으로, LLM이 사전 정의된 함수를 호출할 수 있게 합니다.

### 2. OpenAI Function Calling API 작동 원리

#### API 수준에서의 동작 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                  Function Calling 작동 흐름                  │
│                                                             │
│  Step 1: 도구 정의와 함께 프롬프트 전송                      │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  클라이언트 ──────────────────────────────▶ OpenAI API      │
│             {                                               │
│               "model": "gpt-4o",                            │
│               "messages": [...],                            │
│               "tools": [                                    │
│                 {                                           │
│                   "type": "function",                       │
│                   "function": {                             │
│                     "name": "search_law",                   │
│                     "description": "법령 검색",              │
│                     "parameters": {...}                     │
│                   }                                         │
│                 }                                           │
│               ]                                             │
│             }                                               │
│                                                             │
│  Step 2: LLM이 도구 호출 결정                                │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  OpenAI API ──────────────────────────────▶ 클라이언트      │
│             {                                               │
│               "finish_reason": "tool_calls",                │
│               "message": {                                  │
│                 "tool_calls": [{                            │
│                   "id": "call_abc123",                      │
│                   "function": {                             │
│                     "name": "search_law",                   │
│                     "arguments": '{"query": "근로기준법"}'  │
│                   }                                         │
│                 }]                                          │
│               }                                             │
│             }                                               │
│                                                             │
│  Step 3: 클라이언트가 실제 함수 실행                         │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  클라이언트: search_law("근로기준법") 실행                   │
│           → 결과: "근로기준법 제1조 목적 ..."               │
│                                                             │
│  Step 4: 함수 결과를 포함하여 다시 API 호출                  │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  클라이언트 ──────────────────────────────▶ OpenAI API      │
│             {                                               │
│               "messages": [                                 │
│                 ...,                                        │
│                 {assistant의 tool_calls 메시지},             │
│                 {                                           │
│                   "role": "tool",                           │
│                   "tool_call_id": "call_abc123",            │
│                   "content": "근로기준법 제1조 ..."          │
│                 }                                           │
│               ]                                             │
│             }                                               │
│                                                             │
│  Step 5: 최종 자연어 답변 생성                               │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  OpenAI API ──────────────────────────────▶ 클라이언트      │
│             "근로기준법에 따르면..."                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. JSON 스키마로 도구 정의하기

도구의 입력 매개변수는 JSON Schema 형식으로 정의합니다. 명확한 스키마 정의는 LLM이 올바른 인수를 생성하는 데 핵심적입니다.

#### JSON Schema 기본 구조

```json
{
  "type": "function",
  "function": {
    "name": "search_korean_law",
    "description": "한국 법령 데이터베이스에서 법령을 검색합니다. 특정 법률의 조문이나 법령명으로 검색이 가능합니다.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "검색할 법령명 또는 키워드 (예: '근로기준법', '형법 제250조')"
        },
        "law_type": {
          "type": "string",
          "enum": ["법률", "대통령령", "부령", "조례"],
          "description": "검색할 법령의 종류"
        },
        "max_results": {
          "type": "integer",
          "description": "반환할 최대 결과 수",
          "default": 5,
          "minimum": 1,
          "maximum": 20
        }
      },
      "required": ["query"]
    }
  }
}
```

#### 💡 핵심 포인트: description의 중요성

도구의 `description` 필드는 LLM이 언제 이 도구를 사용해야 하는지 결정하는 데 가장 중요한 역할을 합니다:

- **나쁜 예**: `"description": "법령 검색"`
- **좋은 예**: `"description": "한국 법령 데이터베이스에서 법률 조문을 검색합니다. 특정 법률의 내용 확인, 관련 조문 탐색, 법률 용어의 정의 확인 시 사용하세요. 최신 법령 개정 내용도 포함됩니다."`

---

## 🔗 LangChain Tool 인터페이스

### 1. Tool의 기본 구조

LangChain에서 Tool은 다음 4가지 핵심 속성을 가집니다:

```
┌─────────────────────────────────────────┐
│           LangChain Tool 구조            │
├─────────────────────────────────────────┤
│                                         │
│  name: "search_korean_law"              │
│    → LLM이 이 이름으로 도구를 호출함    │
│                                         │
│  description: "한국 법령 검색..."       │
│    → LLM이 도구 선택 시 참조하는 설명  │
│                                         │
│  args_schema: Pydantic Model            │
│    → 입력 유효성 검사 및 JSON Schema 생성│
│                                         │
│  func: Callable                         │
│    → 실제 실행되는 Python 함수          │
│                                         │
└─────────────────────────────────────────┘
```

### 2. Tool 생성 방법 3가지

```python
# 방법 1: @tool 데코레이터 (가장 간단)
from langchain_core.tools import tool

@tool
def search_law(query: str) -> str:
    """한국 법령을 검색합니다."""
    return f"법령 검색 결과: {query}"

# 방법 2: StructuredTool (복잡한 입력 스키마)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

class LawSearchInput(BaseModel):
    query: str
    law_type: str = "법률"

structured_tool = StructuredTool.from_function(
    func=search_law_impl,
    name="search_korean_law",
    description="법령 검색",
    args_schema=LawSearchInput
)

# 방법 3: BaseTool 상속 (최대 제어권)
from langchain_core.tools import BaseTool

class LawSearchTool(BaseTool):
    name = "search_korean_law"
    description = "법령 검색"

    def _run(self, query: str) -> str:
        return self._search(query)
```

---

## 🤖 에이전트(Agent) 개념

### 1. 에이전트란 무엇인가?

**에이전트(Agent)**는 목표를 달성하기 위해 스스로 의사결정을 내리고 행동하는 시스템입니다. LLM 기반 에이전트에서 LLM은 "두뇌" 역할을 하며 다음을 결정합니다:

- **무엇을 해야 하는가?** (목표 이해)
- **어떤 순서로 해야 하는가?** (계획 수립)
- **어떤 도구를 사용해야 하는가?** (도구 선택)
- **언제 완료되었는가?** (종료 조건 판단)

### 2. 에이전트의 종류

| 에이전트 타입 | 특징 | LangChain 구현 |
|--------------|------|----------------|
| **ReAct** | 추론+행동을 번갈아 수행 | `create_react_agent` |
| **Tool Calling** | LLM의 native tool calling 활용 | `create_tool_calling_agent` |
| **OpenAI Functions** | OpenAI 특화 버전 | `create_openai_functions_agent` |
| **Structured Chat** | 구조화된 출력 형식 | `create_structured_chat_agent` |
| **XML** | XML 형식 응답 | `create_xml_agent` |

### 3. ReAct (Reasoning + Acting) 패러다임

ReAct는 2022년 Google이 발표한 프레임워크로, LLM이 추론(Reasoning)과 행동(Acting)을 번갈아 수행하도록 합니다.

#### ReAct 논문의 핵심 아이디어
> "언어 모델이 추론 추적과 행동을 동기화하여 생성하도록 하면, 더 나은 성능을 발휘한다." - Yao et al., 2022

#### ReAct 동작 예시 (법률 자문 시나리오)

```
질문: "직장 내 괴롭힘으로 해고된 경우 구제 방법은?"

Thought: 이 질문은 직장 내 괴롭힘과 부당해고에 관한 것입니다.
         먼저 관련 법령을 확인해야 합니다.

Action: search_law({"query": "직장 내 괴롭힘 금지법", "law_type": "법률"})

Observation: "근로기준법 제76조의2 (직장 내 괴롭힘의 금지)
             사용자 또는 근로자는 직장에서의 지위 또는 관계 등의 우위를
             이용하여 업무상 적정범위를 넘어 다른 근로자에게 신체적·정신적
             고통을 주거나 근무환경을 악화시키는 행위를 하여서는 아니 된다."

Thought: 법령을 확인했습니다. 이제 해고와 관련된 구제 방법을 찾아야 합니다.

Action: search_law({"query": "부당해고 구제신청", "law_type": "법률"})

Observation: "근로기준법 제28조 (부당해고등의 구제신청)
             사용자가 근로자에게 부당해고등을 하면 근로자는 노동위원회에
             구제를 신청할 수 있다..."

Thought: 이제 판례도 확인하면 좋겠습니다.

Action: search_case({"query": "직장 내 괴롭힘 해고 구제", "max_results": 3})

Observation: [판례 검색 결과...]

Thought: 충분한 정보를 수집했습니다. 이제 종합적인 답변을 제공할 수 있습니다.

Final Answer: "직장 내 괴롭힘으로 인한 해고는 부당해고에 해당할 수 있으며,
              다음과 같은 구제 방법이 있습니다:

              1. 노동위원회 구제신청 (해고일로부터 3개월 이내)
              2. 법원을 통한 해고무효확인 소송
              3. 고용노동부 신고 (직장 내 괴롭힘 금지 위반)"
```

---

## 🔄 에이전트 루프 작동 방식

### AgentExecutor의 내부 동작

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentExecutor 루프                           │
│                                                                 │
│   ┌─────────────┐                                               │
│   │  사용자 입력 │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                      에이전트 루프                        │  │
│   │                                                         │  │
│   │  ┌─────────────────────────────────────────────────┐   │  │
│   │  │  1. LLM 호출 (현재 상태 + 도구 목록 전달)         │   │  │
│   │  └──────────────────────┬──────────────────────────┘   │  │
│   │                         │                               │  │
│   │            ┌────────────┴────────────┐                 │  │
│   │            │                         │                 │  │
│   │            ▼                         ▼                 │  │
│   │   ┌────────────────┐      ┌────────────────────┐       │  │
│   │   │ tool_calls 반환 │      │  최종 답변 반환     │       │  │
│   │   │  (계속 진행)   │      │  (루프 종료)        │       │  │
│   │   └───────┬────────┘      └─────────┬──────────┘       │  │
│   │           │                          │                  │  │
│   │           ▼                          │                  │  │
│   │  ┌────────────────────────────────┐  │                  │  │
│   │  │  2. 도구 실행                  │  │                  │  │
│   │  │   - 도구 이름 파싱             │  │                  │  │
│   │  │   - 인수 유효성 검사           │  │                  │  │
│   │  │   - 실제 함수 호출             │  │                  │  │
│   │  │   - 오류 처리                  │  │                  │  │
│   │  └────────────┬───────────────────┘  │                  │  │
│   │               │                      │                  │  │
│   │               ▼                      │                  │  │
│   │  ┌────────────────────────────────┐  │                  │  │
│   │  │  3. 관찰 결과 메시지에 추가     │  │                  │  │
│   │  │  (ToolMessage 형식)            │  │                  │  │
│   │  └────────────┬───────────────────┘  │                  │  │
│   │               │                      │                  │  │
│   │               └──────────────────────┘                  │  │
│   │               (다시 LLM 호출 - Step 1로 돌아감)          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │  최종 출력    │                              │
│                  └───────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 루프 종료 조건

```python
# AgentExecutor의 루프 제어 파라미터
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,       # 최대 반복 횟수 (무한 루프 방지)
    max_execution_time=60,   # 최대 실행 시간 (초)
    early_stopping_method="generate",  # 종료 방법
    handle_parsing_errors=True,        # 파싱 오류 처리
    verbose=True             # 중간 과정 출력
)
```

---

## 🏗️ LangChain AgentExecutor

### AgentExecutor 구성요소

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. LLM 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. 도구 정의
tools = [search_law_tool, search_case_tool, web_search_tool]

# 3. 프롬프트 템플릿 (agent_scratchpad 필수)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 한국 법률 전문가 AI 어시스턴트입니다."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # 에이전트 중간 과정 저장
])

# 4. 에이전트 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. AgentExecutor로 래핑
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 6. 실행
result = executor.invoke({"input": "근로기준법상 최저임금은 얼마인가요?"})
```

### 에이전트 실행 흐름 상세

```
input: "근로기준법상 최저임금은?"
          │
          ▼
[AgentExecutor.invoke() 시작]
          │
          ▼
[iteration 1]
  LLM 입력: {messages, tools_schema}
  LLM 출력: tool_call("search_law", {"query": "최저임금 근로기준법"})
          │
          ▼
  Tool 실행: search_law("최저임금 근로기준법")
  Tool 출력: "최저임금법 제6조..."
          │
          ▼
[iteration 2]
  LLM 입력: {messages + tool_result, tools_schema}
  LLM 출력: "현재 최저임금은 시간당 9,860원입니다..."
          │
          ▼
[finish_reason: "stop" → 루프 종료]
          │
          ▼
output: {"output": "현재 최저임금은 시간당 9,860원..."}
```

---

## 💡 핵심 포인트

1. **Tool Calling은 2-phase 프로세스입니다**: LLM이 어떤 함수를 호출할지 결정하는 단계와, 실제로 함수를 실행하는 단계가 분리되어 있습니다. LLM은 함수를 직접 실행하지 않습니다.

2. **JSON Schema의 quality가 성능을 결정합니다**: description이 불명확하면 LLM이 잘못된 도구를 선택하거나 잘못된 인수를 생성합니다.

3. **에이전트는 비결정론적입니다**: 동일한 입력에도 다른 경로를 선택할 수 있습니다. `temperature=0`으로 설정하면 일관성이 높아지지만 완전한 결정론은 아닙니다.

4. **max_iterations 설정이 중요합니다**: 무한 루프를 방지하기 위해 항상 최대 반복 횟수를 설정하세요.

5. **도구 오류는 에이전트에게 알려야 합니다**: 도구 실행 중 오류가 발생하면 에이전트에게 오류 메시지를 반환하여 대안적 행동을 취하도록 해야 합니다.

---

## ⚠️ 주의사항

- **비용 관리**: 에이전트는 여러 번의 LLM 호출을 수행하므로 비용이 빠르게 증가할 수 있습니다. 프로덕션 환경에서는 반드시 비용 모니터링을 설정하세요.
- **프롬프트 인젝션**: 도구에서 반환된 외부 데이터에 악의적인 지시가 포함될 수 있습니다. 입력 검증이 필요합니다.
- **도구 설명 보안**: 도구 설명에 민감한 시스템 정보를 포함시키지 마세요.
- **타임아웃 설정**: 외부 API 호출 시 반드시 타임아웃을 설정하여 에이전트가 멈추지 않도록 하세요.

---

## 🔍 심화학습

### Parallel Tool Calling

OpenAI의 최신 모델은 여러 도구를 동시에 호출할 수 있습니다:

```python
# gpt-4o는 여러 도구를 한 번에 호출 가능
# LLM 응답 예시:
{
    "tool_calls": [
        {"name": "search_law", "arguments": {"query": "근로기준법"}},
        {"name": "search_case", "arguments": {"query": "부당해고 판례"}},
        {"name": "web_search", "arguments": {"query": "최신 노동법 뉴스"}}
    ]
}
# 세 도구가 동시에 실행되어 속도 향상
```

### Tool Choice 제어

```python
# 특정 도구 강제 사용
llm_with_forced_tool = llm.bind_tools(
    tools,
    tool_choice={"type": "function", "function": {"name": "search_law"}}
)

# 도구 사용 없이 직접 답변 강제
llm_no_tools = llm.bind_tools(tools, tool_choice="none")

# 자동 선택 (기본값)
llm_auto = llm.bind_tools(tools, tool_choice="auto")
```

---

## 💻 코드 예제

실습 코드는 `code/01_tool_calling_basics.py`를 참조하세요.

---

## ❓ 차시별 질문

1. **Tool Calling과 단순 프롬프팅의 차이는 무엇인가요?** LLM에게 "계산기처럼 행동해"라고 프롬프팅하는 것과 실제 계산기 도구를 제공하는 것의 차이를 설명해보세요.

2. **에이전트가 무한 루프에 빠지는 경우는 언제인가요?** 실제 사례를 생각해보고, 이를 방지하는 방법을 제안해보세요.

3. **법률 자문 에이전트에서 가장 중요한 도구는 무엇이고, 그 이유는 무엇인가요?** 법령검색, 판례검색, 웹검색 중 우선순위를 정하고 근거를 설명해보세요.

4. **도구의 description을 어떻게 작성하면 LLM이 올바르게 선택할 수 있을까요?** 좋은 description과 나쁜 description의 예를 각각 제시해보세요.

5. **parallel tool calling은 언제 유리하고 언제 불리한가요?** 법률 자문 시나리오에서 구체적인 예를 들어 설명해보세요.
