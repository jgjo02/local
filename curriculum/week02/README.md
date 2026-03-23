# Week 2: 문서 처리 및 벡터 데이터베이스

## 주요 학습 내용

2주차에서는 RAG(Retrieval Augmented Generation) 시스템의 핵심인 문서 처리와 벡터 검색을 학습합니다.

## 세션 구성

| 차시 | 주제 | 핵심 개념 |
|------|------|---------|
| 1차시 | NLP 기초 & 토크나이제이션 & 임베딩 | BPE, TF-IDF, Word2Vec, 코사인 유사도 |
| 2차시 | RAG 기본 개념 & 문서 전처리 | RAG 파이프라인, Indexing vs Querying |
| 3차시 | 문서 로더 & 텍스트 분할 | RecursiveCharacterTextSplitter, 청크 전략 |
| 4차시 | 임베딩 & 벡터 저장소 | FAISS, Chroma, OpenAI Embeddings |
| 5차시 | RAG 체인 & Naive RAG 구현 | LCEL RAG, Retriever, 소스 인용 |

## 학습 목표

- NLP의 기초 개념(토큰, 임베딩, 유사도)을 이해한다
- LangChain으로 다양한 문서를 로드하고 처리할 수 있다
- 적절한 청크 전략을 선택하고 적용할 수 있다
- 벡터 저장소(FAISS, Chroma)에 문서를 저장하고 검색할 수 있다
- 완전한 Naive RAG 파이프라인을 구현할 수 있다

## 실습 환경 설치

```bash
pip install langchain langchain-openai langchain-community
pip install faiss-cpu chromadb
pip install pypdf tiktoken
pip install rank-bm25 sentence-transformers
```

## 주요 코드 파일

- `code/01_nlp_basics.py` - NLP 기초 실습
- `code/02_rag_pipeline_basic.py` - RAG 파이프라인 기초
- `code/03_document_loaders.py` - 문서 로더 & 텍스트 분할
- `code/04_embeddings_vectorstore.py` - 임베딩 & 벡터저장소
- `code/05_naive_rag.py` - Naive RAG 완성 구현

## 핵심 개념 미리보기

```
RAG 파이프라인:

[문서 준비 단계 - Indexing]
원본 문서 → 문서 로더 → 텍스트 분할 → 임베딩 → 벡터 저장소

[질의 응답 단계 - Querying]
사용자 질문 → 임베딩 → 유사도 검색 → 관련 문서 → LLM → 답변
```
