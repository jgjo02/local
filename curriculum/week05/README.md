# Week 5: RAG 답변 성능 평가

## 주요 학습 내용

다양한 LLM 공급자 비교 및 RAG 시스템의 답변 품질을 정량적으로 평가하는 방법을 학습합니다.

## 세션 구성

| 차시 | 주제 |
|------|------|
| 1차시 | 주요 LLM 공급자 비교 (Claude, Gemini, Groq, Ollama) |
| 2차시 | RAG 답변 평가 정량 지표 (BLEU, ROUGE, BERTScore) |
| 3차시 | LLM-as-Judge & QA 평가 |
| 4차시 | LLM-as-Judge & Criteria 평가 |
| 5차시 | 텍스트 분할 전략 & 임베딩 모델 비교 |

## 핵심 개념

```
RAG 평가 프레임워크 (RAGAS):

Faithfulness:   답변이 검색된 문서에 근거하는가?
Answer Relevancy: 답변이 질문과 관련있는가?
Context Recall: 필요한 정보가 검색되었는가?
Context Precision: 검색 결과가 정확한가?
```
