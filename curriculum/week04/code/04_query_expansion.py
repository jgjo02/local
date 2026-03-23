"""
4주차 4차시: 쿼리 확장 (Multi-Query, HyDE, RAG-Fusion)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

SAMPLE_DOCS = [
    Document(page_content="주택청약 1순위는 투기과열지구 2년, 수도권 1년, 기타 6개월 이상 납입 무주택자입니다.", metadata={"id": 1}),
    Document(page_content="청약 가점제: 무주택기간(32점) + 부양가족수(35점) + 통장가입기간(17점) = 최대 84점", metadata={"id": 2}),
    Document(page_content="특별공급: 신혼부부(혼인 7년 이내), 생애최초, 다자녀(3명 이상), 노부모부양", metadata={"id": 3}),
    Document(page_content="분양가 상한제: 공공택지 및 규제지역 민간 아파트 가격 상한 제도", metadata={"id": 4}),
    Document(page_content="전매 제한: 투기과열지구 소유권이전등기까지 분양권 양도 금지", metadata={"id": 5}),
    Document(page_content="LH 공공분양: 소득 기준, 자산 기준 충족 시 신청 가능, 저렴한 분양가", metadata={"id": 6}),
]

vectorstore = FAISS.from_documents(SAMPLE_DOCS, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def demo_multi_query():
    """Multi-Query Retriever 데모"""
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    query = "청약 점수 올리는 방법"
    print(f"Multi-Query 검색: '{query}'")
    docs = multi_retriever.invoke(query)
    print(f"결과 {len(docs)}개:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:80]}")


def demo_hyde():
    """HyDE (Hypothetical Document Embeddings) 데모"""
    hyde_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대한 답변 문단을 작성하세요. 실제 문서처럼 작성하세요.

질문: {question}

답변 문단:""")

    generate_doc_chain = hyde_prompt | llm | StrOutputParser()

    def hyde_search(question: str, k: int = 3):
        hypothetical_doc = generate_doc_chain.invoke({"question": question})
        print(f"  가상 문서: {hypothetical_doc[:100]}...")
        return vectorstore.similarity_search(hypothetical_doc, k=k)

    query = "청약 당첨 확률을 높이려면?"
    print(f"\nHyDE 검색: '{query}'")
    docs = hyde_search(query)
    print(f"결과 {len(docs)}개:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:80]}")


def demo_rag_fusion():
    """RAG-Fusion: Multi-Query + RRF 결합"""
    from collections import defaultdict

    queries_prompt = ChatPromptTemplate.from_template("""
주어진 질문을 다양한 관점으로 재작성하여 4개의 검색 쿼리를 생성하세요.
각 쿼리를 새 줄에 작성하고 번호를 붙이지 마세요.

원래 질문: {question}""")

    generate_queries_chain = queries_prompt | llm | StrOutputParser()

    def rag_fusion(question: str, k: int = 4):
        queries_text = generate_queries_chain.invoke({"question": question})
        queries = [question] + [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
        print(f"  생성된 쿼리들: {queries}")

        # 각 쿼리로 검색
        all_results = defaultdict(lambda: {"doc": None, "score": 0})
        for q in queries:
            docs = base_retriever.invoke(q)
            for rank, doc in enumerate(docs):
                key = doc.page_content[:50]
                all_results[key]["doc"] = doc
                all_results[key]["score"] += 1 / (rank + 60)

        reranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in reranked[:k] if item["doc"]]

    query = "청약 1순위 조건과 가점 계산"
    print(f"\nRAG-Fusion: '{query}'")
    docs = rag_fusion(query)
    print(f"결과 {len(docs)}개:")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.page_content[:80]}")


if __name__ == "__main__":
    print("4주차 4차시: 쿼리 확장\n")
    demo_multi_query()
    demo_hyde()
    demo_rag_fusion()
