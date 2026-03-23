# 2차시: RAG 답변 평가 정량 지표

## 학습 목표
- BLEU, ROUGE, BERTScore 등 자동 평가 지표를 이해하고 계산할 수 있다
- RAGAS 프레임워크로 RAG 파이프라인을 평가할 수 있다

---

## 1. 전통적 자동 평가 지표

### 1.1 BLEU Score (BiLingual Evaluation Understudy)

```
기본 아이디어: 생성 텍스트 n-gram이 참조 텍스트와 얼마나 겹치는가

예시:
  참조: "청약 1순위는 통장 2년 이상 납입한 무주택자입니다"
  생성: "1순위 청약은 무주택자로서 2년 이상 통장을 납입해야 합니다"

1-gram 정밀도: 6/9 = 0.67
2-gram 정밀도: 4/8 = 0.50
...

문제: 단어 순서 무시, 동의어 처리 안 됨
```

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["청약", "1순위는", "통장", "2년", "이상", "납입한", "무주택자"]]
hypothesis = ["1순위", "청약은", "무주택자로서", "2년", "이상", "통장을", "납입"]

smoothing = SmoothingFunction().method1
score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
print(f"BLEU: {score:.4f}")
```

### 1.2 ROUGE Score

```
ROUGE-1: 1-gram 재현율
ROUGE-2: 2-gram 재현율
ROUGE-L: 가장 긴 공통 부분 수열(LCS)

재현율 중심 (얼마나 많은 참조 내용을 포함했는가)
```

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(
    target="청약 1순위는 통장 2년 이상 납입한 무주택자입니다",
    prediction="1순위 청약은 무주택자로서 2년 이상 통장을 납입해야 합니다"
)

for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f}, R={score.recall:.3f}, F1={score.fmeasure:.3f}")
```

---

## 2. RAGAS - RAG 특화 평가 프레임워크

```
RAGAS 4대 지표:

1. Faithfulness (충실도): 답변이 검색 문서에 근거하는가?
   → 0.0 (완전 환각) ~ 1.0 (완전 충실)

2. Answer Relevancy (답변 관련성): 질문과 답변이 관련있는가?
   → 역방향: 답변에서 질문을 역생성하여 유사도 측정

3. Context Recall (컨텍스트 재현율): 정답에 필요한 정보를 검색했는가?
   → 정답의 각 문장이 검색 컨텍스트에 있는지 확인

4. Context Precision (컨텍스트 정밀도): 검색 결과가 얼마나 정확한가?
   → 검색된 문서 중 실제 유용한 것의 비율
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

from datasets import Dataset

data = {
    "question": ["청약 1순위 조건은?"],
    "answer": ["청약 1순위는 통장 2년 이상 납입한 무주택자입니다."],
    "contexts": [["투기과열지구 기준 2년 이상 납입한 무주택자가 1순위입니다."]],
    "ground_truth": ["1순위는 무주택자로 통장을 2년 이상 납입해야 합니다."]
}

dataset = Dataset.from_dict(data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
print(result)
```

---

## 3. 지표 해석 가이드

| 지표 | 낮은 경우 의미 | 개선 방법 |
|------|-------------|---------|
| Faithfulness < 0.7 | 환각 발생 | 프롬프트 강화, 컨텍스트 보강 |
| Answer Relevancy < 0.8 | 무관한 답변 | 프롬프트 개선 |
| Context Recall < 0.7 | 검색 실패 | 청크 전략 개선, 쿼리 확장 |
| Context Precision < 0.6 | 노이즈 많음 | 재순위화, 필터링 |

---

## 💡 핵심 포인트
- BLEU/ROUGE는 단어 중복 기반, 의미적 유사도 포착 못함
- RAGAS는 RAG 특화 지표로 실서비스 평가에 적합
- Faithfulness가 가장 중요 - 환각 탐지의 핵심 지표

## ❓ 차시별 질문
1. Faithfulness가 낮으면 어떤 대응 방법이 있을까요?
2. 평가 데이터셋을 직접 만들 때 주의사항은?
