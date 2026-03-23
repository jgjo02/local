# 5차시: RAG 체인 & Naive RAG 구현

## 학습 목표
- LCEL로 완전한 RAG 체인을 구성할 수 있다
- Retriever의 다양한 설정을 활용할 수 있다
- 소스 문서 인용 기능을 구현할 수 있다
- Naive RAG의 한계를 이해하고 개선 방향을 설명할 수 있다

---

## 1. RAG 체인 구성 방법

### 1.1 구성 요소

```
RAG 체인 = Retriever + Prompt + LLM + OutputParser

질문 ──→ [Retriever] ──→ 관련 문서들
   └──────────────────→ [Prompt] ──→ [LLM] ──→ 답변
```

### 1.2 LCEL 방식 (v0.3 표준)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답하세요.
컨텍스트에 없는 내용은 "알 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

# 3. 문서 포맷터
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 체인 구성
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 실행
answer = rag_chain.invoke("청약 자격 조건은?")
```

### 1.3 소스 문서 인용 포함

```python
from langchain_core.runnables import RunnableParallel

# 답변과 소스 문서를 함께 반환
rag_chain_with_source = RunnableParallel({
    "answer": rag_chain,
    "sources": retriever
})

result = rag_chain_with_source.invoke("청약 자격 조건은?")
print(f"답변: {result['answer']}")
print("\n소스 문서:")
for i, doc in enumerate(result['sources']):
    print(f"[{i+1}] {doc.metadata.get('source', 'Unknown')} - "
          f"p.{doc.metadata.get('page', '?')}")
    print(f"    {doc.page_content[:100]}...")
```

---

## 2. 완전한 Naive RAG 구현

```python
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


class NaiveRAG:
    """Naive RAG 구현 클래스"""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4
    ):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.vectorstore = None
        self.chain = None

    def load_documents(self, file_paths: list) -> list:
        """다양한 포맷 문서 로드"""
        docs = []
        for path in file_paths:
            path = Path(path)
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            elif path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                print(f"지원하지 않는 형식: {path.suffix}")
                continue
            docs.extend(loader.load())
        print(f"로드된 문서: {len(docs)}개")
        return docs

    def build_index(self, file_paths: list):
        """RAG 인덱스 구축"""
        # 1. 문서 로드
        raw_docs = self.load_documents(file_paths)

        # 2. 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(raw_docs)
        print(f"생성된 청크: {len(chunks)}개")

        # 3. 벡터 저장소 구축
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("벡터 저장소 구축 완료")

        # 4. 체인 구성
        self._build_chain()

    def _build_chain(self):
        """RAG 체인 구성"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

        prompt = ChatPromptTemplate.from_template(
            """당신은 유능한 AI 어시스턴트입니다.
주어진 컨텍스트만을 사용하여 질문에 답변하세요.
컨텍스트에 관련 정보가 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
추측이나 외부 지식을 사용하지 마세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        )

        def format_docs(docs):
            return "\n\n---\n\n".join([
                f"[출처: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ])

        base_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        self.chain = RunnableParallel({
            "answer": base_chain,
            "source_docs": retriever
        })

    def query(self, question: str) -> dict:
        """질문에 답변"""
        if self.chain is None:
            raise ValueError("먼저 build_index()를 호출하세요")

        result = self.chain.invoke(question)
        return {
            "question": question,
            "answer": result["answer"],
            "sources": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in result["source_docs"]
            ]
        }

    def save_index(self, path: str = "rag_index"):
        """인덱스 저장"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"인덱스 저장: {path}")

    def load_index(self, path: str = "rag_index"):
        """저장된 인덱스 불러오기"""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self._build_chain()
        print(f"인덱스 로드: {path}")
```

---

## 3. Naive RAG의 한계와 개선 방향

### 3.1 주요 실패 케이스

```
1. 어휘 불일치 (Vocabulary Mismatch)
   질문: "전세 보증금 반환 기한은?"
   문서: "임대차 보증금 반환 의무 기간"
   → 같은 의미지만 다른 표현 → 검색 실패

2. 복합 질문 (Multi-hop)
   질문: "A가 B의 대표인 회사의 매출은?"
   → 단일 검색으로 답변 불가, 2단계 검색 필요

3. 최근성 편향
   → 오래된 문서는 임베딩 공간에서 멀어질 수 있음

4. 청크 경계 문제
   → 핵심 정보가 두 청크에 나뉨
```

### 3.2 Advanced RAG 예고

```
Naive RAG → Advanced RAG 개선 포인트:

검색 단계:
  ✗ 단순 의미 검색
  ✓ 하이브리드 검색 (BM25 + 의미)
  ✓ 쿼리 확장 (Multi-Query, HyDE)
  ✓ 재순위화 (Re-ranking)

답변 단계:
  ✗ 단순 컨텍스트 + 질문
  ✓ Self-RAG (환각 검증)
  ✓ Corrective RAG (검색 수정)
  ✓ LangGraph 에이전트
```

---

## 💡 핵심 포인트
- LCEL 파이프라인으로 RAG 체인 구성 (v0.3 표준)
- `RunnableParallel`로 답변과 소스 문서를 동시에 반환
- 시스템 프롬프트에서 "컨텍스트 외 답변 금지"를 명시하여 환각 억제
- Naive RAG는 출발점이며, 실서비스에는 Advanced RAG 필수

## ⚠️ 주의사항
- Retriever와 LLM 사이에 포맷팅 함수(`format_docs`) 반드시 삽입
- k 값이 너무 크면 컨텍스트 창 초과, 너무 작으면 정보 누락
- 인덱스 구축 후 임베딩 모델 변경 불가 (재구축 필요)

## ❓ 차시별 질문
1. 동일 질문에 k=2 vs k=8 비교 실험을 해보세요
2. 컨텍스트에 없는 질문을 하면 어떻게 동작하나요?
3. 소스 인용 기능이 왜 중요한가요?
