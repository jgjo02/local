# 📚 7주차: LangChain 도구(Tool) 활용

## 🎯 주차 학습 목표

7주차에서는 LangChain의 도구(Tool) 시스템을 깊이 있게 학습합니다. 단순한 LLM 응답 생성을 넘어, 외부 세계와 상호작용할 수 있는 AI 에이전트를 구축하는 방법을 익힙니다. 법률 자문 에이전트(프로젝트 3)를 실제 구현하면서 Tool Calling, 내장 도구, 커스텀 도구, 외부 API 통합, 그리고 ReAct 패러다임을 종합적으로 학습합니다.

---

## 📋 차시 구성

| 차시 | 주제 | 핵심 내용 |
|------|------|-----------|
| 1차시 | [프로젝트 3] 안내 + Tool Calling + 에이전트 개념 | Function Calling API, JSON Schema, ReAct |
| 2차시 | LangChain 내장 도구 활용 | Tavily, DuckDuckGo, Wikipedia, Arxiv |
| 3차시 | 사용자 정의 도구 (Custom Tool) | @tool 데코레이터, StructuredTool, BaseTool |
| 4차시 | 외부 API 통합 | 법제처 API, Rate Limiting, 캐싱 전략 |
| 5차시 | ReAct 에이전트 + 다국어 RAG | create_react_agent, 라우팅 체인, 다국어 처리 |

---

## 🗂️ 파일 구조

```
week07/
├── README.md                              # 이 파일
├── 01_프로젝트3안내_ToolCalling_Agent개념.md
├── 02_LangChain내장도구활용.md
├── 03_사용자정의도구.md
├── 04_외부API통합.md
├── 05_ReAct에이전트_다국어RAG.md
└── code/
    ├── 01_tool_calling_basics.py
    ├── 02_builtin_tools.py
    ├── 03_custom_tools.py
    ├── 04_external_api_integration.py
    └── 05_react_multilingual_rag.py
```

---

## 🔑 핵심 개념 요약

### Tool Calling이란?
LLM이 자신의 지식 범위를 넘어서는 작업(실시간 정보 검색, 계산, 외부 시스템 호출 등)을 수행하기 위해 미리 정의된 함수(도구)를 호출하는 메커니즘입니다.

```
사용자 질문 → LLM → 도구 선택 결정 → 도구 실행 → 결과 통합 → 최종 답변
```

### 에이전트(Agent)란?
LLM을 "두뇌"로 사용하여 목표 달성을 위해 어떤 행동을 취할지 스스로 결정하고, 도구를 활용하며, 관찰 결과를 바탕으로 다음 행동을 결정하는 시스템입니다.

### ReAct 패러다임
- **Re**asoning (추론): "무엇을 해야 하는가?"
- **Act**ing (행동): 도구를 실제로 호출
- Thought → Action → Observation의 반복 루프

---

## 🛠️ 환경 설정

```bash
# 필수 패키지 설치
pip install langchain langchain-openai langchain-community
pip install tavily-python duckduckgo-search wikipedia
pip install arxiv langchain-experimental

# 환경 변수 설정
export OPENAI_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-key"
```

---

## 📊 주차 학습 로드맵

```
[1차시] Tool Calling 개념 이해
    ↓
[2차시] 내장 도구로 실습 (웹 검색, 위키피디아)
    ↓
[3차시] 직접 도구 만들기 (법률 특화)
    ↓
[4차시] 실제 외부 API 연동 (법제처 API)
    ↓
[5차시] ReAct 에이전트 + 다국어 처리 → 프로젝트 3 기반 완성
```

---

## 📖 프로젝트 3: 법률 자문 에이전트

### 프로젝트 개요
한국 법률에 특화된 AI 자문 에이전트를 구축합니다. 사용자의 법률 관련 질문에 대해:
1. 관련 법령을 검색하고
2. 판례를 찾아보며
3. 전문가적 해석을 제공하는

종합적인 법률 자문 시스템입니다.

### 핵심 도구 목록
| 도구명 | 기능 | 데이터 소스 |
|--------|------|-------------|
| `법령검색` | 특정 법령 조문 검색 | 국가법령정보센터 API |
| `판례검색` | 관련 판례 검색 | 종합법률정보 API |
| `웹검색` | 최신 법률 뉴스/해석 | Tavily Search |
| `법률계산기` | 시효/기간 계산 | 내장 로직 |
| `문서분석` | 계약서/서류 분석 | LLM + RAG |

---

## 💡 이번 주 학습 포인트

1. **도구는 명확한 목적이 있어야 한다**: 이름(name)과 설명(description)이 LLM이 도구를 올바르게 선택하는 데 결정적 역할을 합니다.

2. **에이전트는 반복적으로 추론한다**: 한 번의 호출로 끝나는 체인과 달리, 에이전트는 목표를 달성할 때까지 Think-Act-Observe를 반복합니다.

3. **오류 처리가 핵심이다**: 실제 서비스에서 외부 도구는 언제든 실패할 수 있습니다. 견고한 오류 처리와 폴백 전략이 필수입니다.

4. **다국어 처리의 복잡성**: 한국어 법률 텍스트는 영어 기반 임베딩 모델과의 호환성, 번역 품질 등 고유한 도전이 있습니다.

---

## 🔗 참고 자료

- [LangChain Tools 공식 문서](https://python.langchain.com/docs/modules/tools/)
- [OpenAI Function Calling 가이드](https://platform.openai.com/docs/guides/function-calling)
- [ReAct 논문: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [국가법령정보 Open API](https://open.law.go.kr/LSO/openApi/openApiInfo.do)
- [Tavily Search API](https://tavily.com/)

---

## ⚠️ 주의사항

- 법률 자문 에이전트는 **교육 목적**으로만 활용하며, 실제 법률 조언을 대체하지 않습니다.
- API 키는 절대 코드에 하드코딩하지 말고 환경 변수나 `.env` 파일을 사용하세요.
- 외부 API 호출 시 Rate Limit을 반드시 고려하세요.
- LLM의 법률 해석은 오류가 있을 수 있으므로 전문가 검토가 필요합니다.
