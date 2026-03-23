# 2차시: Zero-Shot & Few-Shot 프롬프팅

## 학습 목표
- Zero-shot과 Few-shot 프롬프팅의 차이와 활용 방법을 이해한다
- 동적 예시 선택기(ExampleSelector)를 구현할 수 있다

---

## 1. Zero-Shot 프롬프팅

모델에게 예시 없이 직접 작업 수행:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "텍스트를 긍정/부정/중립으로 분류하세요."),
    ("human", "{text}")
])

chain = zero_shot_prompt | llm
result = chain.invoke({"text": "이 제품 정말 만족스럽네요!"})
```

**적합한 케이스:** 단순 분류, 간단한 변환, 범용 작업

---

## 2. Few-Shot 프롬프팅

예시를 제공하여 원하는 출력 형식/스타일 유도:

```python
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate, ChatPromptTemplate
)

examples = [
    {"input": "사과", "output": "🍎 사과 - 빨간 달콤한 과일"},
    {"input": "바나나", "output": "🍌 바나나 - 노란 길쭉한 과일"},
    {"input": "딸기", "output": "🍓 딸기 - 빨간 작은 베리류"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

final = ChatPromptTemplate.from_messages([
    ("system", "과일을 이모지와 함께 설명하세요."),
    few_shot,
    ("human", "{fruit}")
])

chain = final | llm
print(chain.invoke({"fruit": "수박"}))
```

---

## 3. 동적 예시 선택기

모든 예시를 넣으면 토큰 낭비 → 질문과 관련된 예시만 동적 선택:

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

examples = [
    {"input": "파이썬 리스트 정렬", "output": "sorted() 또는 .sort() 사용"},
    {"input": "파이썬 딕셔너리 순회", "output": "items(), keys(), values() 사용"},
    {"input": "에러 처리", "output": "try/except 블록 사용"},
    {"input": "파일 읽기", "output": "with open() as f: f.read()"},
    {"input": "HTTP 요청", "output": "requests 라이브러리 사용"},
]

selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2  # 가장 관련된 2개만 선택
)

# 현재 질문과 유사한 예시 선택
selected = selector.select_examples({"input": "리스트 뒤집기"})
print(selected)  # 파이썬 리스트 관련 예시가 선택됨
```

---

## 4. 예시 수와 성능 관계

```
실험 결과 (분류 태스크 기준):
  0-shot:  정확도 72%
  1-shot:  정확도 81%
  3-shot:  정확도 87%
  5-shot:  정확도 89%
  10-shot: 정확도 90%  (수익 체감)

→ 3~5개 예시가 비용/성능 최적점
→ 다양성이 중요 (같은 패턴 반복 X)
```

---

## 💡 핵심 포인트
- Zero-shot: 간단한 작업, Few-shot: 특정 형식/스타일 필요 시
- 예시 선택 시 다양성과 관련성이 중요
- SemanticSimilarityExampleSelector로 질문별 최적 예시 동적 선택

## ❓ 차시별 질문
1. 3개 예시 vs 10개 예시 성능을 실험해보세요
2. 나쁜 예시가 포함되면 어떻게 될까요?
