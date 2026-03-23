# 3차시: LLM-as-Judge & QA 평가

## 학습 목표
- LLM-as-Judge 개념과 전통적 평가 방법의 차이를 이해한다
- QAEvalChain을 활용하여 RAG 답변을 평가할 수 있다

---

## 1. LLM-as-Judge 개념

```
전통적 평가의 한계:
  BLEU, ROUGE → 단어 중복 기반, 의미 파악 불가
  사람 평가    → 시간 소요, 비용 높음, 일관성 부족

LLM-as-Judge:
  LLM이 평가자 역할 → 의미적 이해 가능
  일관성 있는 대규모 평가 가능
  비용 대비 효율 우수
```

---

## 2. QAEvalChain 활용

```python
from langchain.evaluation import QAEvalChain
from langchain_openai import ChatOpenAI

eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# QA 쌍 준비
examples = [
    {
        "query": "청약 1순위 조건은?",
        "answer": "투기과열지구 기준 통장 2년 이상 납입, 무주택 세대구성원"
    }
]

# RAG가 생성한 예측 답변
predictions = [
    {"result": "1순위가 되려면 청약통장을 2년 이상 납입하고 무주택자여야 합니다."}
]

eval_chain = QAEvalChain.from_llm(eval_llm)
graded_outputs = eval_chain.evaluate(examples, predictions)

for i, grade in enumerate(graded_outputs):
    print(f"Q: {examples[i]['query']}")
    print(f"등급: {grade['results']}")
```

---

## 3. ContextQAEvalChain

컨텍스트(검색 문서)를 기반으로 답변이 정확한지 평가:

```python
from langchain.evaluation import ContextQAEvalChain

examples = [
    {
        "query": "청약 1순위 조건은?",
        "context": "투기과열지구: 청약통장 2년 이상 납입 + 무주택",
        "answer": "2년 이상 납입 + 무주택"
    }
]

predictions = [
    {"result": "1순위는 무주택자로 2년 이상 납입해야 합니다."}
]

eval_chain = ContextQAEvalChain.from_llm(eval_llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
```

---

## 4. 배치 평가 파이프라인

```python
def evaluate_rag_system(rag_chain, test_cases):
    """RAG 시스템 전체 평가"""
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_chain = QAEvalChain.from_llm(eval_llm)

    predictions = []
    for case in test_cases:
        result = rag_chain.invoke(case["query"])
        predictions.append({"result": result["answer"]})

    grades = eval_chain.evaluate(test_cases, predictions)

    correct = sum(1 for g in grades if "CORRECT" in g.get("results", "").upper())
    accuracy = correct / len(grades)

    return {
        "accuracy": accuracy,
        "total": len(grades),
        "correct": correct,
        "grades": grades
    }
```

---

## 💡 핵심 포인트
- LLM-as-Judge는 의미적 이해가 가능하여 전통적 지표보다 RAG 평가에 적합
- `QAEvalChain`: 질문+정답 기반, `ContextQAEvalChain`: 컨텍스트 포함 평가
- 평가 LLM은 답변 생성 LLM과 분리하는 것이 좋음

## ❓ 차시별 질문
1. 평가 LLM 자체의 편향(bias)은 어떻게 줄일 수 있을까요?
2. LLM-as-Judge 결과와 사람 평가 결과가 다를 때 어떻게 처리할까요?
