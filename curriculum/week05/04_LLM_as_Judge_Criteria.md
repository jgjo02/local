# 4차시: LLM-as-Judge & Criteria 평가

## 학습 목표
- 기준(Criteria) 기반 평가 방법을 이해하고 적용할 수 있다
- 사용자 정의 기준으로 맞춤형 평가 시스템을 구축할 수 있다

---

## 1. Criteria 평가란?

특정 기준(기준)에 따라 답변을 평가:

```python
from langchain.evaluation import load_evaluator

# 내장 기준 평가
evaluator = load_evaluator(
    "criteria",
    criteria="conciseness",  # 간결성
    llm=ChatOpenAI(model="gpt-4o-mini")
)

result = evaluator.evaluate_strings(
    input="청약 1순위 조건은?",
    prediction="1순위가 되려면 청약통장을 가입한 후 투기과열지구는 2년 이상, 수도권은 1년 이상, 기타 지역은 6개월 이상 납입하고 무주택 세대구성원이어야 합니다."
)
print(result)  # {'score': 0, 'value': 'N', 'reasoning': '...너무 길다...'}
```

### 내장 기준 목록

| 기준 | 설명 |
|------|------|
| conciseness | 간결성 |
| relevance | 관련성 |
| correctness | 정확성 |
| coherence | 일관성 |
| harmfulness | 유해성 |
| maliciousness | 악의성 |
| helpfulness | 도움됨 |
| controversiality | 논란성 |

---

## 2. 사용자 정의 기준

```python
# RAG 시스템에 특화된 커스텀 기준
custom_criteria = {
    "source_citation": """
    답변이 출처 문서를 명시적으로 인용하거나 언급합니까?
    다음 기준으로 판단하세요:
    Y: 출처를 명확히 언급함
    N: 출처 언급 없음
    """,

    "legal_disclaimer": """
    법률/의료 관련 답변에 적절한 면책 조항이 포함되어 있습니까?
    Y: 전문가 상담 권유 포함
    N: 면책 조항 없음
    """,

    "factual_accuracy": """
    답변의 사실 정확성을 평가하세요.
    Y: 주요 사실이 정확함
    N: 부정확하거나 오해의 소지가 있음
    """
}

# 복합 기준 평가
evaluator = load_evaluator(
    "criteria",
    criteria=custom_criteria,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

result = evaluator.evaluate_strings(
    input="청약 당첨 후 계약 포기하면 어떻게 되나요?",
    prediction="""청약 당첨 후 계약을 포기하면 투기과열지구 주택의 경우
    최대 10년간 재청약이 제한될 수 있습니다.
    정확한 사항은 청약홈이나 전문가에게 문의하세요."""
)
print(result)
```

---

## 3. LabeledCriteriaEvalChain (정답 포함)

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator(
    "labeled_criteria",
    criteria="correctness",
    llm=ChatOpenAI(model="gpt-4o-mini")
)

result = evaluator.evaluate_strings(
    input="청약 1순위 가점 최대점은?",
    prediction="84점",
    reference="가점제 최대점은 84점입니다"
)
```

---

## 4. 다차원 평가 프레임워크

```python
def comprehensive_evaluation(question: str, prediction: str, reference: str = None):
    """다차원 RAG 답변 평가"""
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    criteria_list = ["conciseness", "relevance", "helpfulness"]
    results = {}

    for criteria in criteria_list:
        evaluator = load_evaluator("criteria", criteria=criteria, llm=eval_llm)
        result = evaluator.evaluate_strings(input=question, prediction=prediction)
        results[criteria] = {
            "score": result.get("score", 0),
            "reasoning": result.get("reasoning", "")[:100]
        }

    total_score = sum(r["score"] for r in results.values()) / len(results)
    results["total_score"] = total_score
    return results
```

---

## 💡 핵심 포인트
- 내장 기준(conciseness, relevance, correctness 등)으로 빠른 평가 가능
- 도메인 특화 사용자 정의 기준으로 세밀한 평가 가능
- 다차원 평가로 답변의 여러 측면을 종합 판단

## ❓ 차시별 질문
1. 청약 챗봇에 필요한 가장 중요한 평가 기준 3개를 선택하고 이유를 설명하세요
2. 평가 기준의 설명(prompt)이 모호하면 어떤 문제가 생길까요?
