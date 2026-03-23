"""
2주차 5차시: Naive RAG 완성 구현
- 문서 로드 → 청킹 → 임베딩 → FAISS → RAG 체인
- 소스 인용 포함
- 대화형 CLI 인터페이스
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document

load_dotenv()

# ============================================================
# 샘플 데이터 (실제 파일 없을 경우 사용)
# ============================================================

SAMPLE_FAQ = """
Q: 주택청약 저축 가입 자격은?
A: 만 19세 이상이면 누구나 가입할 수 있습니다. 단, 청약을 위해서는 무주택 세대구성원이어야 합니다.

Q: 청약 가점제란?
A: 무주택기간, 부양가족수, 청약통장 가입기간 세 가지 항목으로 최대 84점을 계산합니다.

Q: 특별공급 종류에는 무엇이 있나요?
A: 신혼부부, 생애최초, 다자녀, 노부모부양, 장애인, 국가유공자 특별공급이 있습니다.

Q: 청약 당첨 후 계약 포기하면 불이익이 있나요?
A: 일반적으로 당첨 후 계약을 포기하면 일정 기간(최대 10년) 청약이 제한됩니다.

Q: 민간분양과 공공분양의 차이는?
A: 민간분양은 건설사가 공급하며 분양가 상한제 미적용 단지가 많습니다. 공공분양은 LH, SH 등이 공급하며 분양가가 저렴합니다.

Q: 청약 신청은 어디서 하나요?
A: 청약홈(applyhome.co.kr) 또는 은행 앱에서 신청할 수 있습니다.

Q: 청약통장 납입 횟수가 중요한가요?
A: 공공분양의 경우 납입 횟수(24회 이상 권장)가 당첨에 영향을 줍니다. 민간분양은 주로 납입금액(1,500만 원 이상)이 기준입니다.

Q: 전용면적 85㎡ 이하란?
A: 전용면적은 방, 거실, 주방, 욕실 등 실제 거주 면적으로, 85㎡ 이하가 국민주택 규모입니다.

Q: 청약 1순위 조건은?
A: 청약저축 가입 후 2년 이상(투기과열지구), 수도권은 1년 이상, 기타 지역은 6개월 이상입니다.

Q: 분양가 상한제란?
A: 주택 분양 가격의 상한을 규제하는 제도로, 적용 지역에서는 주변 시세보다 낮게 공급됩니다.
"""


def create_sample_documents() -> list:
    """샘플 Document 리스트 생성"""
    docs = []
    qa_pairs = SAMPLE_FAQ.strip().split("\n\n")

    for i, qa in enumerate(qa_pairs):
        if qa.strip():
            docs.append(Document(
                page_content=qa,
                metadata={
                    "source": "sample_faq.txt",
                    "chunk_id": i,
                    "category": "주택청약"
                }
            ))
    return docs


# ============================================================
# Naive RAG 클래스
# ============================================================

class NaiveRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.rag_chain = None

    def build_from_file(self, file_path: str):
        """파일에서 인덱스 구축"""
        path = Path(file_path)
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        raw_docs = loader.load()
        self._build_index(raw_docs)

    def build_from_documents(self, docs: list):
        """Document 리스트에서 인덱스 구축"""
        self._build_index(docs)

    def _build_index(self, docs: list):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        if len(docs) > 0 and len(docs[0].page_content) > 500:
            chunks = splitter.split_documents(docs)
        else:
            chunks = docs  # 이미 작은 청크

        print(f"청크 수: {len(chunks)}")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("벡터 인덱스 구축 완료")

        self._setup_chain()

    def _setup_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        PROMPT = ChatPromptTemplate.from_template("""
당신은 주택청약 전문 상담사입니다.
아래 제공된 참고 문서만을 사용하여 질문에 답변하세요.
참고 문서에 없는 내용은 "제공된 정보에 해당 내용이 없습니다"라고 답하세요.

참고 문서:
{context}

질문: {question}

답변:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        answer_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        self.rag_chain = RunnableParallel({
            "answer": answer_chain,
            "sources": retriever
        })

    def ask(self, question: str) -> dict:
        if not self.rag_chain:
            raise ValueError("먼저 build_from_file() 또는 build_from_documents() 호출 필요")

        result = self.rag_chain.invoke(question)
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    def print_answer(self, result: dict):
        print(f"\n질문: {result['question']}")
        print(f"\n답변:\n{result['answer']}")
        print("\n참고 문서:")
        for i, doc in enumerate(result["sources"], 1):
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i}] {source}: {doc.page_content[:80]}...")


# ============================================================
# 대화형 CLI
# ============================================================

def interactive_rag():
    """대화형 RAG 인터페이스"""
    print("=" * 50)
    print("Naive RAG 데모 - 주택청약 Q&A")
    print("=" * 50)

    rag = NaiveRAG()

    # 샘플 데이터로 인덱스 구축
    sample_docs = create_sample_documents()
    rag.build_from_documents(sample_docs)

    print("\n준비 완료! 질문을 입력하세요 (종료: 'q')\n")
    print("예시 질문:")
    print("  - 청약 1순위 조건은?")
    print("  - 특별공급 종류를 알려줘")
    print("  - 청약통장 납입 횟수가 중요한가요?\n")

    while True:
        question = input("질문: ").strip()
        if question.lower() in ['q', 'quit', '종료', 'exit']:
            print("종료합니다.")
            break
        if not question:
            continue

        result = rag.ask(question)
        rag.print_answer(result)
        print()


# ============================================================
# 배치 평가
# ============================================================

def batch_evaluation():
    """여러 질문으로 RAG 성능 평가"""
    rag = NaiveRAG()
    sample_docs = create_sample_documents()
    rag.build_from_documents(sample_docs)

    test_questions = [
        "청약 가점제 점수 계산 방법은?",
        "민간분양과 공공분양 차이가 뭔가요?",
        "청약 당첨 후 계약을 취소하면 어떻게 되나요?",
        "청약 신청은 어디서 하나요?",
        "전용면적 85㎡가 뭔가요?",
    ]

    print("\n=" * 50)
    print("배치 평가")
    print("=" * 50)

    for q in test_questions:
        result = rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer'][:200]}...")
        print(f"소스: {[doc.metadata.get('source') for doc in result['sources']]}")


if __name__ == "__main__":
    import sys
    if "--batch" in sys.argv:
        batch_evaluation()
    else:
        interactive_rag()
