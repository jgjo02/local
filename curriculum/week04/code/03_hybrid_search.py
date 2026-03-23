"""
4주차 3차시: 하이브리드 검색 (BM25 + 벡터) 구현
"""

import os
import math
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

SAMPLE_DOCS = [
    Document(page_content="주택청약 1순위는 청약통장 2년 이상 납입한 무주택자입니다.", metadata={"source": "faq1"}),
    Document(page_content="청약 가점제는 무주택기간 부양가족 통장가입기간으로 계산합니다.", metadata={"source": "faq2"}),
    Document(page_content="특별공급에는 신혼부부 생애최초 다자녀 장애인 등이 있습니다.", metadata={"source": "faq3"}),
    Document(page_content="분양가 상한제는 주택 가격 상한을 규제하는 제도입니다.", metadata={"source": "faq4"}),
    Document(page_content="청약 당첨 후 계약 포기시 최대 10년 재청약 제한이 있습니다.", metadata={"source": "faq5"}),
    Document(page_content="공공분양은 LH SH에서 공급하며 소득 기준이 있습니다.", metadata={"source": "faq6"}),
    Document(page_content="청약홈에서 온라인으로 청약 신청이 가능합니다.", metadata={"source": "faq7"}),
    Document(page_content="전매 제한은 분양권을 일정 기간 타인에게 양도할 수 없는 규제입니다.", metadata={"source": "faq8"}),
]


def demo_bm25():
    """BM25 검색 데모"""
    bm25_retriever = BM25Retriever.from_documents(SAMPLE_DOCS)
    bm25_retriever.k = 3

    query = "청약 1순위 조건"
    results = bm25_retriever.invoke(query)

    print("BM25 검색 결과:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")


def demo_dense_search():
    """밀집 벡터 검색 데모"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)

    query = "청약 점수 계산 방법"
    results = vectorstore.similarity_search(query, k=3)

    print("\n벡터 검색 결과:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")

    return vectorstore


def demo_hybrid_search(vectorstore):
    """하이브리드 검색 (BM25 + 벡터) 데모"""
    # BM25 리트리버
    bm25_retriever = BM25Retriever.from_documents(SAMPLE_DOCS)
    bm25_retriever.k = 3

    # 벡터 리트리버
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 앙상블 리트리버 (EnsembleRetriever with RRF)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # BM25 50%, 벡터 50%
    )

    query = "청약 자격 조건"
    results = ensemble.invoke(query)

    print("\n하이브리드 검색 결과 (BM25 50% + 벡터 50%):")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")

    # 가중치 조정 비교
    print("\n가중치 변경 (BM25 30% + 벡터 70%):")
    ensemble2 = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    results2 = ensemble2.invoke(query)
    for i, doc in enumerate(results2, 1):
        print(f"  [{i}] {doc.page_content}")


def compare_search_methods(vectorstore, query: str):
    """세 가지 검색 방법 비교"""
    bm25 = BM25Retriever.from_documents(SAMPLE_DOCS, k=3)
    vector = vectorstore.as_retriever(search_kwargs={"k": 3})
    hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])

    print(f"\n=== '{query}' 검색 비교 ===")

    methods = [("BM25", bm25), ("벡터", vector), ("하이브리드", hybrid)]
    for name, retriever in methods:
        results = retriever.invoke(query)
        print(f"\n{name} 검색:")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:60]}")


if __name__ == "__main__":
    print("4주차 3차시: 하이브리드 검색\n")
    demo_bm25()
    vectorstore = demo_dense_search()
    demo_hybrid_search(vectorstore)
    compare_search_methods(vectorstore, "특별공급 신혼부부 조건")
    compare_search_methods(vectorstore, "1순위 가점 계산")
