"""
2주차 4차시: 임베딩 & 벡터 저장소 실습
- OpenAI 임베딩
- FAISS 벡터 저장소
- Chroma DB
- 메타데이터 필터링
- MMR 검색
"""

import os
import math
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# 샘플 문서들
SAMPLE_DOCS = [
    Document(page_content="파이썬은 간결한 문법의 프로그래밍 언어입니다.", metadata={"category": "tech", "source": "doc1"}),
    Document(page_content="머신러닝은 데이터를 통해 패턴을 학습합니다.", metadata={"category": "ai", "source": "doc2"}),
    Document(page_content="청약통장은 주택청약 신청에 필요합니다.", metadata={"category": "housing", "source": "doc3"}),
    Document(page_content="딥러닝은 신경망 기반 머신러닝 기법입니다.", metadata={"category": "ai", "source": "doc4"}),
    Document(page_content="주택 청약 1순위 조건은 2년 이상 납입입니다.", metadata={"category": "housing", "source": "doc5"}),
    Document(page_content="자연어 처리는 NLP라고도 불립니다.", metadata={"category": "ai", "source": "doc6"}),
    Document(page_content="RAG는 검색 증강 생성 기법입니다.", metadata={"category": "ai", "source": "doc7"}),
    Document(page_content="분양가 상한제는 주택 가격을 규제합니다.", metadata={"category": "housing", "source": "doc8"}),
]


def demo_openai_embeddings():
    """OpenAI 임베딩 기본 사용"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = ["파이썬이란?", "Python programming", "자동차 엔진"]
    vectors = embeddings.embed_documents(texts)

    print("OpenAI 임베딩:")
    for text, vec in zip(texts, vectors):
        print(f"  '{text}': {len(vec)}차원, 앞 3개=[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]")

    # 코사인 유사도 계산
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a))
        nb = math.sqrt(sum(x**2 for x in b))
        return dot / (na * nb)

    print(f"\n  '파이썬이란?'와 'Python programming' 유사도: {cosine_sim(vectors[0], vectors[1]):.4f}")
    print(f"  '파이썬이란?'와 '자동차 엔진' 유사도: {cosine_sim(vectors[0], vectors[2]):.4f}")


def demo_faiss():
    """FAISS 벡터 저장소"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 생성
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)
    print(f"\nFAISS 벡터 저장소 ({len(SAMPLE_DOCS)}개 문서)")

    # 기본 유사도 검색
    results = vectorstore.similarity_search("머신러닝 딥러닝", k=3)
    print("\n유사도 검색 결과 ('머신러닝 딥러닝'):")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")

    # 점수 포함 검색
    results_score = vectorstore.similarity_search_with_score("주택청약 조건", k=3)
    print("\n점수 포함 검색 ('주택청약 조건'):")
    for doc, score in results_score:
        print(f"  스코어={score:.4f}: {doc.page_content}")

    # 저장/로드
    vectorstore.save_local("/tmp/faiss_demo")
    loaded = FAISS.load_local("/tmp/faiss_demo", embeddings, allow_dangerous_deserialization=True)
    print(f"\n저장 후 로드 성공: {loaded.index.ntotal}개 벡터")

    return vectorstore


def demo_chroma():
    """Chroma DB 벡터 저장소"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 영구 저장
    vectorstore = Chroma.from_documents(
        SAMPLE_DOCS,
        embeddings,
        persist_directory="/tmp/chroma_demo",
        collection_name="sample"
    )

    count = vectorstore._collection.count()
    print(f"\nChroma DB ({count}개 문서)")

    # 메타데이터 필터 검색
    ai_results = vectorstore.similarity_search(
        "학습 알고리즘",
        k=3,
        filter={"category": "ai"}
    )
    print("\n메타데이터 필터 (category=ai):")
    for doc in ai_results:
        print(f"  [{doc.metadata['category']}] {doc.page_content}")

    return vectorstore


def demo_retriever_types(vectorstore):
    """다양한 Retriever 타입 비교"""
    query = "인공지능 학습 방법"

    # 기본 유사도
    sim_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # MMR (다양성 보장)
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 6}
    )

    print(f"\nQuery: '{query}'")

    print("\n기본 유사도 검색:")
    for doc in sim_retriever.invoke(query):
        print(f"  {doc.page_content}")

    print("\nMMR 검색 (다양성):")
    for doc in mmr_retriever.invoke(query):
        print(f"  {doc.page_content}")


if __name__ == "__main__":
    print("2주차 4차시: 임베딩 & 벡터 저장소\n")
    demo_openai_embeddings()
    faiss_store = demo_faiss()
    demo_chroma()
    demo_retriever_types(faiss_store)
