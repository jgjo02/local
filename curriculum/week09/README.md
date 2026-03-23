# Week 9: Agent 기반 RAG 구현

## 주요 학습 내용

LangGraph를 활용하여 Adaptive RAG, Self-RAG, Corrective RAG를 구현합니다.

## 세션 구성

| 차시 | 주제 |
|------|------|
| 1차시 | Adaptive RAG 1부 (라우팅, 검색) |
| 2차시 | Adaptive RAG 2부 (완성) |
| 3차시 | Self-RAG 1부 (검색 필요성, 관련성 평가) |
| 4차시 | Self-RAG 2부 (환각 검증, 완성) |
| 5차시 | Corrective RAG & Gradio 연동 |

## 학습 목표

- Adaptive RAG: 질문 유형에 따른 동적 검색 전략
- Self-RAG: 자가 반성으로 답변 품질 향상
- Corrective RAG: 검색 실패 시 대안 경로 탐색

## Advanced RAG 비교

```
Naive RAG:       질문 → 검색 → LLM → 답변 (단순)
Adaptive RAG:    질문 분류 → [검색/웹/직접답변] → LLM
Self-RAG:        검색 → 관련성 체크 → 생성 → 환각 체크 → 출력
Corrective RAG:  검색 → 신뢰도 평가 → [그대로/수정/웹검색] → 생성
```
