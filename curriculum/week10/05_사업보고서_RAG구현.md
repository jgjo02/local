# 5차시: 사업보고서 RAG 구현

## 학습 목표
- DART API로 사업보고서를 자동 수집할 수 있다
- 기업 특화 RAG 시스템을 완성할 수 있다
- 재무/비재무 통합 질의응답 시스템을 구현할 수 있다

---

## 1. DART API 활용

```python
import requests
import os

def fetch_dart_report(corp_code: str, year: str = "2023") -> str:
    """DART API로 사업보고서 다운로드"""
    api_key = os.getenv("DART_API_KEY")

    # 보고서 목록 조회
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": f"{year}0101",
        "end_de": f"{year}1231",
        "pblntf_ty": "A",  # 사업보고서
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] == "000":
        reports = data.get("list", [])
        if reports:
            return reports[0]["rcept_no"]  # 최신 보고서 접수번호
    return None
```

---

## 2. 기업 특화 RAG 시스템

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class BusinessReportRAG:
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None

    def build_index(self, documents: list):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_template("""
당신은 {company}의 사업보고서 분석 전문가입니다.
아래 사업보고서 내용을 바탕으로 질문에 답변하세요.

사업보고서 내용:
{context}

질문: {question}

답변 (구체적 수치와 근거 포함):""")

        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
                "company": lambda _: self.company_name
            }
            | prompt | self.llm | StrOutputParser()
        )
        return chain

    def analyze_financials(self, year: str = "2023") -> str:
        """재무 분석"""
        chain = self.create_chain()
        questions = [
            f"{year}년 매출액과 영업이익은?",
            f"{year}년 부채비율은?",
            "주요 사업 부문별 매출 비중은?"
        ]
        results = []
        for q in questions:
            answer = chain.invoke(q)
            results.append(f"Q: {q}\nA: {answer}")
        return "\n\n".join(results)
```

---

## 3. 연도별 비교 분석

```python
def compare_annual_reports(company_name: str, years: list) -> str:
    """연도별 실적 비교"""
    comparisons = []

    for year in years:
        rag = BusinessReportRAG(company_name)
        # 해당 연도 보고서 로드
        docs = load_year_report(company_name, year)
        rag.build_index(docs)
        chain = rag.create_chain()

        metrics = chain.invoke(f"{year}년 매출액, 영업이익, 순이익을 정확한 수치로 알려주세요")
        comparisons.append(f"[{year}년]\n{metrics}")

    return "\n\n".join(comparisons)
```

---

## 4. 메타데이터 기반 필터링

```python
# 섹션별 검색
def search_by_section(rag: BusinessReportRAG, query: str, section: str):
    """특정 섹션에서만 검색"""
    retriever = rag.vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {"section": section}
        }
    )
    return retriever.invoke(query)

# 재무 섹션에서만 검색
financial_docs = search_by_section(rag, "영업이익", "재무에 관한 사항")
```

---

## 💡 핵심 포인트
- DART API로 실제 사업보고서를 자동 수집 가능
- 섹션 메타데이터로 "재무" vs "사업 내용" 구분 검색
- 연도별 데이터를 비교 분석하면 트렌드 파악 가능
- 재무 수치는 LLM이 환각하기 쉬우므로 프롬프트에 정확성 강조

## ❓ 차시별 질문
1. 재무제표의 수치를 직접 DB에 저장하고 RAG를 보완하는 방법은?
2. 여러 회사 보고서를 동시에 비교 분석하는 쿼리를 어떻게 설계할까요?
