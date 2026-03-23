# Week 12: Graph RAG

## 주요 학습 내용

그래프 데이터베이스(Neo4j)와 지식 그래프를 활용한 Graph RAG를 구현합니다.

## 세션 구성

| 차시 | 주제 |
|------|------|
| 1차시 | 그래프 데이터베이스 이해 |
| 2차시 | Neo4j & Cypher 쿼리 |
| 3차시 | LangChain + Neo4j 통합 |
| 4차시 | 지식 그래프 구축 |
| 5차시 | Graph RAG 구현 & 최종 프로젝트 완성 |

## 학습 목표

- Neo4j 설치 및 Cypher 쿼리 작성
- LangChain Neo4jGraph로 자연어 → Cypher 변환
- LLMGraphTransformer로 텍스트에서 지식 그래프 자동 추출
- Graph RAG로 복잡한 관계 쿼리 처리

## Graph RAG vs 일반 RAG

```
일반 RAG:    문서 → 벡터 검색 → LLM
              장점: 단순, 빠름
              단점: 관계 파악 불가 ("삼성전자 CEO의 전 직장은?")

Graph RAG:   지식 그래프 → 그래프 순회 + 벡터 검색 → LLM
              장점: 복잡한 관계, 멀티홉 질의 가능
              단점: 구축 비용, 복잡성
```
