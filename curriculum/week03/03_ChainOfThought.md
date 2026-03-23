# 3차시: Chain-of-Thought 프롬프팅

## 학습 목표
- CoT 프롬프팅의 원리와 적용 케이스를 이해한다
- Zero-shot CoT, Few-shot CoT를 구현할 수 있다
- Tree of Thoughts, Self-Consistency 등 고급 기법을 설명할 수 있다

---

## 1. Chain-of-Thought (CoT) 개념

**핵심 아이디어:** 단계별 추론 과정을 프롬프트에 포함시켜 복잡한 문제 해결력 향상

```
일반 프롬프트:
Q: "사과 3개에 바나나 2개를 더하면?" → A: "5"

CoT 프롬프트:
Q: "사과 3개에 바나나 2개를 더하면?"
   "단계별로 생각해봅시다."
A: "사과 3개가 있습니다.
    바나나 2개를 더합니다.
    3 + 2 = 5
    따라서 총 5개입니다."
```

---

## 2. Zero-Shot CoT

"Let's think step by step" 마법 문구:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# Zero-shot CoT
cot_prompt = ChatPromptTemplate.from_template("""
{problem}

단계별로 생각하여 풀어보세요.
""")

chain = cot_prompt | llm
result = chain.invoke({
    "problem": "철수는 사탕 12개를 가지고 있다. 영희에게 1/3을 주고, 민수에게 2개를 주었다. 남은 사탕은?"
})
print(result.content)
```

---

## 3. Few-Shot CoT

복잡한 추론에 단계별 예시 제공:

```python
examples = [
    {
        "question": "5명이 각각 3개의 사탕을 받았다. 총 사탕은?",
        "reasoning": "5명이 있습니다.\n각자 3개를 받았습니다.\n5 × 3 = 15",
        "answer": "총 15개입니다."
    },
    {
        "question": "20개 중 1/4을 사용했다. 남은 수는?",
        "reasoning": "전체 20개입니다.\n1/4 사용 = 20 × 1/4 = 5개 사용\n20 - 5 = 15",
        "answer": "15개가 남았습니다."
    }
]

few_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "수학 문제를 단계별로 풀어주세요."),
    *[msg for ex in examples for msg in [
        ("human", f"Q: {ex['question']}"),
        ("ai", f"풀이: {ex['reasoning']}\n답: {ex['answer']}")
    ]],
    ("human", "Q: {question}")
])
```

---

## 4. 고급 CoT 기법

### Tree of Thoughts (ToT)
```
단일 경로 CoT:      A → B → C → 답
ToT (다중 경로):    A → B1 → C1 → 답1 (선택!)
                       ↘ B2 → C2 → 답2
                       ↘ B3 → X (가지치기)

적합: 복잡한 전략 게임, 창의적 글쓰기
```

### Self-Consistency (다수결)
```python
# 같은 질문을 여러 번 실행 후 가장 많이 나온 답 선택
answers = []
for _ in range(5):
    result = chain.invoke({"problem": problem})
    answers.append(extract_answer(result.content))

from collections import Counter
final_answer = Counter(answers).most_common(1)[0][0]
```

### ReAct (Reasoning + Acting)
LLM이 추론하면서 도구를 사용:
```
생각: "날씨 정보가 필요하다"
행동: search_weather("서울 오늘 날씨")
관찰: "서울 맑음, 15도"
생각: "내일 비가 올지 확인해야 한다"
행동: search_weather("서울 내일 날씨")
...
최종 답변: "오늘은 맑지만 내일은 비가 옵니다"
```

---

## 💡 핵심 포인트
- CoT는 수학적 추론, 논리, 다단계 문제에 효과적
- Zero-shot CoT: "단계별로 생각해봅시다" 한 문장으로도 효과
- Self-Consistency: 여러 번 실행 후 다수결로 신뢰도 향상
- ReAct = LangChain Agent의 기반 원리

## ❓ 차시별 질문
1. CoT 없이 vs CoT 있이 복잡한 수학 문제를 풀어보세요
2. Temperature=1.0으로 Self-Consistency를 실험해보세요
