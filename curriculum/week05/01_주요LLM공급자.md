# 1차시: 주요 LLM 공급자 비교

## 학습 목표
- 주요 LLM 공급자(Anthropic, Google, Meta, Groq)의 특성을 비교할 수 있다
- LangChain으로 다양한 모델을 통합하여 활용할 수 있다
- Ollama로 로컬 LLM을 실행할 수 있다

---

## 1. LLM 공급자 현황 (2025)

### 1.1 Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

claude = ChatAnthropic(
    model="claude-opus-4-6",   # 최고 성능
    # model="claude-sonnet-4-6",  # 균형
    # model="claude-haiku-4-5",  # 빠름/저렴
    max_tokens=1024
)

response = claude.invoke("RAG란 무엇인가요?")
print(response.content)
```

**특징:** 긴 컨텍스트(200K), 코드 작성 우수, 안전성 강조

### 1.2 Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",    # 멀티모달, 긴 컨텍스트
    # model="gemini-1.5-flash", # 빠름, 저렴
    temperature=0.7
)

response = gemini.invoke("파이썬 람다란?")
```

**특징:** 1M 컨텍스트, 이미지/영상/음성 지원

### 1.3 Groq - 초고속 추론

```python
from langchain_groq import ChatGroq

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
# Groq LPU: 기존 GPU 대비 10-100배 빠른 추론
```

### 1.4 Ollama - 로컬 LLM

```bash
# 설치 후 모델 다운로드
ollama pull llama3.2
ollama pull qwen2.5:7b
ollama pull gemma3:4b
```

```python
from langchain_ollama import ChatOllama

ollama = ChatOllama(
    model="llama3.2",
    temperature=0.8
)
# 완전 로컬 실행 - 데이터 보안 보장
```

---

## 2. 모델 선택 기준

| 기준 | 추천 모델 |
|------|---------|
| 최고 성능 | Claude Opus, GPT-4o |
| 비용 효율 | Claude Haiku, Gemini Flash |
| 초고속 추론 | Groq + Llama/Mixtral |
| 프라이버시 | Ollama (로컬) |
| 한국어 | Claude, GPT-4o, Qwen2.5 |
| 코딩 | Claude Sonnet, GPT-4o |
| 다국어 RAG | Gemini Flash, Claude Haiku |

---

## 3. LangChain 통합 패턴

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("{question}")

# 모델만 교체하면 동일한 체인 사용
for model_name, model in [
    ("GPT-4o-mini", ChatOpenAI(model="gpt-4o-mini")),
    ("Claude Haiku", ChatAnthropic(model="claude-haiku-4-5-20251001")),
    ("Groq Llama", ChatGroq(model="llama-3.3-70b-versatile")),
]:
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": "RAG란?"})
    print(f"[{model_name}]: {response[:100]}")
```

---

## 💡 핵심 포인트
- LangChain의 추상화로 모델 교체가 코드 한 줄로 가능
- 비용-성능 트레이드오프에 따라 모델 선택
- 로컬 LLM(Ollama)은 데이터 보안이 중요한 경우 필수

## ❓ 차시별 질문
1. 동일 질문을 3개 모델에 보내고 응답 품질/속도/비용을 비교해보세요
2. 한국어 답변 품질이 가장 높은 모델은?
