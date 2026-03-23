"""
4주차 5차시: 재순위화 & 맥락 압축
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    LLMChainExtractor, LLMChainFilter, EmbeddingsFilter
)
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

DOCS = [
    Document(page_content="""주택청약 1순위 조건
투기과열지구: 청약통장 가입 2년 이상 + 납입 24회 이상
수도권: 1년 이상 + 12회 이상
기타 지역: 6개월 이상 + 6회 이상
모든 세대구성원이 무주택자여야 합니다.""", metadata={"source": "doc1"}),
    Document(page_content="""가점제 점수 계산
무주택기간: 1년미만 0점 ~ 15년이상 32점
부양가족수: 0명 5점 ~ 6명이상 35점
통장가입기간: 6개월미만 1점 ~ 15년이상 17점
합계 최대 84점""", metadata={"source": "doc2"}),
    Document(page_content="""특별공급 자격
신혼부부: 혼인 7년 이내, 무주택, 소득 140%이하
생애최초: 세대원 모두 주택 소유 이력 없음
다자녀: 미성년 자녀 3명 이상
노부모부양: 65세 이상 직계존속 3년 이상 부양""", metadata={"source": "doc3"}),
    Document(page_content="""청약 당첨 후 절차
1. 당첨자 발표 확인
2. 서류 제출 (당첨일로부터 보통 7일~14일)
3. 계약 체결 (분양가의 10% 계약금)
4. 중도금 납부 (보통 6회 분할)
5. 잔금 납부 및 입주""", metadata={"source": "doc4"}),
]

vectorstore = FAISS.from_documents(DOCS, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def demo_llm_extractor():
    """LLMChainExtractor: 관련 부분만 추출"""
    extractor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )

    query = "무주택기간 점수는?"
    print(f"LLMChainExtractor 결과 ('{query}'):")
    docs = compression_retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] 원본 소스: {doc.metadata.get('source')}")
        print(f"       추출 내용: {doc.page_content}")


def demo_embeddings_filter():
    """EmbeddingsFilter: 유사도 기반 빠른 필터"""
    emb_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=0.76
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=emb_filter,
        base_retriever=base_retriever
    )

    query = "1순위 납입 횟수"
    print(f"\nEmbeddingsFilter 결과 ('{query}'):")
    docs = compression_retriever.invoke(query)
    print(f"필터링 후 {len(docs)}개 (전체 {4}개 중):")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:80]}")


def demo_pipeline():
    """압축 파이프라인 결합"""
    from langchain.retrievers.document_compressors import DocumentCompressorPipeline
    from langchain_text_splitters import CharacterTextSplitter

    pipeline = DocumentCompressorPipeline(
        transformers=[
            CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n"),
            EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7),
        ]
    )

    pipeline_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=base_retriever
    )

    query = "특별공급 신혼부부 조건"
    print(f"\n압축 파이프라인 결과 ('{query}'):")
    docs = pipeline_retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:80]}")


if __name__ == "__main__":
    print("4주차 5차시: 재순위화 & 맥락 압축\n")
    demo_llm_extractor()
    demo_embeddings_filter()
    demo_pipeline()
