# 📚 Week 4: RAG 검색 성능 평가 및 개선

## 🎯 학습 목표

이번 주차에서는 RAG(Retrieval-Augmented Generation) 시스템의 **검색 성능을 체계적으로 평가하고 개선**하는 방법을 학습합니다. 단순한 벡터 검색을 넘어 하이브리드 검색, 쿼리 확장, 재순위화 등 Advanced RAG 기법을 통해 실무 수준의 검색 시스템을 구축합니다.

---

## 📋 주차 개요

| 차시 | 주제 | 핵심 내용 |
|------|------|-----------|
| 1차시 | [프로젝트 2] 안내 + RAG 성능평가 개요 | 프로젝트 2 소개, 평가 프레임워크 이해 |
| 2차시 | 정보 검색 평가지표 | HitRate, MRR, NDCG 이론 및 구현 |
| 3차시 | 키워드 검색 / 하이브리드 검색 | BM25, RRF, EnsembleRetriever |
| 4차시 | 쿼리 확장 | MultiQuery, HyDE, RAG-Fusion |
| 5차시 | 재순위화 + 맥락 압축 | Cross-Encoder, Cohere Reranker |

---

## 🗺️ 학습 로드맵

```
Week 4: RAG 검색 성능 평가 및 개선
│
├── 1차시: 프로젝트 안내 & 평가 개요
│   ├── 프로젝트 2 (ETF 추천 시스템) 소개
│   ├── Advanced RAG 개요
│   ├── 평가 프레임워크 (RAGAS, DeepEval)
│   └── 테스트셋 구축 방법
│
├── 2차시: 검색 평가 지표
│   ├── Precision, Recall, F1
│   ├── Hit Rate (HR@K)
│   ├── MRR (Mean Reciprocal Rank)
│   └── NDCG, MAP
│
├── 3차시: 검색 성능 향상 I
│   ├── BM25 키워드 검색
│   ├── Semantic Search
│   ├── Hybrid Search (BM25 + Vector)
│   └── RRF (Reciprocal Rank Fusion)
│
├── 4차시: 검색 성능 향상 II
│   ├── Multi-Query Retriever
│   ├── HyDE (가상 문서 임베딩)
│   ├── RAG-Fusion
│   └── Query Decomposition
│
└── 5차시: 검색 성능 향상 III
    ├── Re-ranking (Cross-Encoder)
    ├── Cohere Reranker
    ├── Contextual Compression
    └── LLMChainExtractor/Filter
```

---

## 🏗️ Advanced RAG 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   Advanced RAG Pipeline                  │
│                                                          │
│  사용자 쿼리                                              │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────────┐                                     │
│  │   쿼리 확장      │ ← Multi-Query, HyDE, RAG-Fusion    │
│  │  (Week4 4차시)  │                                     │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │   하이브리드     │ ← BM25 + Vector Search             │
│  │     검색        │    RRF Fusion                       │
│  │  (Week4 3차시)  │                                     │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │   재순위화       │ ← Cross-Encoder, Cohere            │
│  │  (Week4 5차시)  │                                     │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │   맥락 압축      │ ← LLMChainExtractor                │
│  │  (Week4 5차시)  │                                     │
│  └────────┬────────┘                                     │
│           │                                              │
│           ▼                                              │
│  ┌─────────────────┐                                     │
│  │   LLM 답변 생성  │                                    │
│  └─────────────────┘                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 성능 평가 체계

```
RAG 성능 평가
├── 검색(Retrieval) 평가        ← Week 4
│   ├── Hit Rate @ K
│   ├── MRR
│   ├── NDCG
│   └── MAP
│
└── 답변(Generation) 평가       ← Week 5
    ├── BLEU / ROUGE
    ├── BERTScore
    ├── Faithfulness
    ├── Answer Relevancy
    └── LLM-as-Judge
```

---

## 💻 실습 파일 구조

```
week04/
├── README.md (현재 파일)
├── 01_프로젝트2안내_RAG성능평가개요.md
├── 02_정보검색_평가지표.md
├── 03_키워드검색_하이브리드검색.md
├── 04_쿼리확장.md
├── 05_재순위화_맥락압축.md
└── code/
    ├── 02_retrieval_metrics.py
    ├── 03_hybrid_search.py
    ├── 04_query_expansion.py
    └── 05_reranking_compression.py
```

---

## 🔧 필요 패키지

```bash
# 기본 LangChain 패키지
pip install langchain langchain-openai langchain-community

# 검색 관련 패키지
pip install rank-bm25 faiss-cpu chromadb

# 재순위화 패키지
pip install cohere flashrank sentence-transformers

# 한국어 처리
pip install konlpy  # 형태소 분석기

# 평가 프레임워크
pip install ragas deepeval

# 데이터 처리
pip install pandas numpy scikit-learn
```

---

## 🎯 주차별 학습 성과

이번 주차를 마치면 다음을 수행할 수 있습니다:

1. **검색 성능 정량 측정**: HR@K, MRR, NDCG 지표를 계산하고 해석
2. **하이브리드 검색 구현**: BM25와 벡터 검색을 결합한 강력한 검색 시스템 구축
3. **쿼리 확장 기법 적용**: Multi-Query, HyDE 등으로 검색 재현율 향상
4. **재순위화 파이프라인**: Cross-Encoder를 활용한 검색 정확도 향상
5. **맥락 압축 적용**: 불필요한 컨텍스트를 제거하여 LLM 효율 향상

---

## 📚 참고 자료

- [LangChain Retrievers Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [RAGAS: RAG Evaluation Framework](https://docs.ragas.io/)
- [BM25 Paper: Okapi BM25](https://dl.acm.org/doi/10.1145/2682862.2682863)
- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank)
