"""Week 2 실습 2: 기본 RAG 파이프라인 구현"""

import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# ── 샘플 문서 ─────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    """주택청약 1순위 조건
국민주택에 청약하려면 청약통장 가입 기간이 지역에 따라 다릅니다.
투기과열지구에서는 2년 이상, 청약과열지역에서는 1년 이상이어야 합니다.
일반 지역의 경우 6개월 이상 납입 실적이 필요합니다.
납입 횟수는 24회 이상이어야 1순위 자격이 됩니다.""",

    """특별공급 자격 조건
특별공급은 일반 경쟁을 거치지 않고 특정 계층에게 우선 공급하는 제도입니다.
신혼부부 특별공급: 혼인 기간 7년 이내, 무주택 세대구성원
다자녀 특별공급: 미성년 자녀 3명 이상
생애최초 특별공급: 세대원 전원 주택을 소유한 적 없는 경우
노부모 부양 특별공급: 65세 이상 직계존속 3년 이상 부양""",

    """청약 당첨 후 절차
당첨 발표일 이후 당첨자 서류 제출 기간이 있습니다.
서류 심사 후 계약 체결을 진행합니다.
계약금은 통상 분양가의 10%입니다.
중도금 대출은 사전에 금융기관과 협의하세요.
잔금은 입주 지정 기간 내에 납부해야 합니다.""",
]


# ── RAG 파이프라인 구축 ────────────────────────────────────────────────────────

def create_documents(texts: List[str]) -> List[Document]:
    """텍스트를 Document 리스트로 변환"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs = []
    for i, text in enumerate(texts):
        docs = splitter.create_documents([text], metadatas=[{"source": f"doc_{i}"}])
        all_docs.extend(docs)

    print(f"✅ {len(all_docs)}개 청크 생성")
    return all_docs


def build_vector_store(docs: List[Document]) -> FAISS:
    """FAISS 벡터 스토어 구축"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = FAISS.from_documents(docs, embeddings)
    print(f"✅ 벡터 스토어 구축 완료")
    return store


def build_rag_chain(vector_store: FAISS):
    """LCEL RAG 체인 구성"""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 주택청약 전문 상담사입니다.
아래 참고 자료를 바탕으로 정확하게 답변하세요.
참고 자료에 없는 내용은 모른다고 답변하세요.

참고 자료:
{context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        return

    print("🔧 RAG 파이프라인 구축 중...")
    docs = create_documents(SAMPLE_DOCS)
    vector_store = build_vector_store(docs)
    chain, retriever = build_rag_chain(vector_store)

    print("✅ RAG 파이프라인 준비 완료\n")

    test_queries = [
        "청약 1순위 조건이 무엇인가요?",
        "신혼부부 특별공급 자격이 궁금합니다",
        "당첨 후 어떤 절차가 있나요?",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"❓ 질문: {query}")

        # 검색 결과 확인
        retrieved = retriever.invoke(query)
        print(f"\n📄 검색된 문서 ({len(retrieved)}개):")
        for i, doc in enumerate(retrieved, 1):
            print(f"  [{i}] {doc.page_content[:80]}...")

        # 답변 생성
        answer = chain.invoke(query)
        print(f"\n💬 답변:\n{answer}")


if __name__ == "__main__":
    main()
