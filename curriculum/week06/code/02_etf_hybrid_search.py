"""Week 6 실습 2: ETF 하이브리드 검색 구현"""

import json
import os
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "../../projects/project2_etf_recommendation/data/sample_etf_data.json"


def load_documents(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        etfs = json.load(f)["etfs"]

    docs = []
    for etf in etfs:
        text = f"{etf['name']} {etf['category']} {etf['region']} {etf['risk_level']} "
        text += f"{etf['description']} {' '.join(etf.get('theme', []))}"

        docs.append(Document(
            page_content=text,
            metadata=etf,
        ))
    return docs


def build_retriever(documents: List[Document]) -> EnsembleRetriever:
    """하이브리드 검색기 구축"""
    print("🔧 BM25 인덱스 구축...")
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 10

    print("🔧 FAISS 인덱스 구축...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_store = FAISS.from_documents(documents, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 10})

    print("🔧 앙상블 검색기 구성...")
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble


def search_and_display(retriever: EnsembleRetriever, query: str, top_k: int = 5) -> None:
    """검색 결과 출력"""
    print(f"\n🔍 검색: '{query}'")
    print("-" * 50)
    results = retriever.invoke(query)[:top_k]

    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"{i}. {m['name']} ({m['ticker']})")
        print(f"   카테고리: {m['category']} | 위험도: {m['risk_level']}")
        print(f"   1년 수익률: {m['1y_return']:+.1f}% | 수수료: {m['expense_ratio']}%")


def compare_search_methods(documents: List[Document], query: str) -> None:
    """BM25 vs Dense vs Hybrid 검색 비교"""
    print(f"\n{'='*60}")
    print(f"검색어: {query}")
    print(f"{'='*60}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_store = FAISS.from_documents(documents, embeddings)

    # BM25 검색
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 5
    bm25_results = bm25.invoke(query)
    print("\n[BM25 키워드 검색]")
    for doc in bm25_results[:3]:
        print(f"  - {doc.metadata['name']}")

    # Dense 검색
    dense_results = faiss_store.similarity_search(query, k=5)
    print("\n[Dense 벡터 검색]")
    for doc in dense_results[:3]:
        print(f"  - {doc.metadata['name']}")

    # 하이브리드 검색
    faiss_ret = faiss_store.as_retriever(search_kwargs={"k": 5})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[0.4, 0.6],
    )
    hybrid_results = ensemble.invoke(query)
    print("\n[하이브리드 검색]")
    for doc in hybrid_results[:3]:
        print(f"  - {doc.metadata['name']}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        exit(1)

    print("📚 ETF 데이터 로드...")
    documents = load_documents(DATA_PATH)
    print(f"  ✅ {len(documents)}개 Document 생성")

    retriever = build_retriever(documents)
    print("✅ 하이브리드 검색기 준비 완료\n")

    test_queries = [
        "노후 대비 안전한 채권 ETF",
        "AI 반도체 고성장 투자",
        "미국 나스닥 기술주",
        "배당 수익 월 배당",
    ]

    for q in test_queries:
        search_and_display(retriever, q, top_k=3)

    # 검색 방법 비교
    compare_search_methods(documents, "저위험 안전 채권")
