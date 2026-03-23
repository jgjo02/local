# Week 11-4: Project 4 기업 분석 시스템 완성

## 학습 목표
- Project 4 전체 아키텍처 통합
- 멀티모달 + Graph RAG 결합
- 기업 종합 분석 리포트 자동 생성

---

## 1. Project 4 아키텍처

```
project4_company_analysis/
├── app.py                          # Gradio 웹 UI
├── src/
│   ├── config.py                   # 설정
│   ├── document_parser.py          # unstructured 문서 파싱
│   ├── knowledge_graph_builder.py  # Neo4j 지식 그래프
│   ├── multimodal_vector_store.py  # 멀티모달 벡터 스토어
│   ├── multimodal_processor.py     # 이미지/표 처리
│   ├── graph_rag.py               # Graph RAG 체인
│   └── company_analyzer.py        # 종합 분석 에이전트
└── data/
    └── sample_company_info.json    # 샘플 기업 데이터
```

---

## 2. 통합 CompanyAnalyzer

```python
# src/company_analyzer.py 핵심 구조

class CompanyAnalyzer:
    """기업 종합 분석 에이전트"""

    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.graph_rag = None
        self.multimodal_store = None

    def analyze(self, company_name: str, analysis_type: str) -> dict:
        """종합 분석 실행"""
        results = {}

        # 1. 텍스트 기반 RAG 분석
        text_result = self._text_rag_analysis(company_name, analysis_type)
        results["text_analysis"] = text_result

        # 2. 그래프 기반 관계 분석
        if self.graph_rag:
            graph_result = self._graph_analysis(company_name)
            results["graph_analysis"] = graph_result

        # 3. 멀티모달 분석 (차트/이미지)
        if self.multimodal_store:
            visual_result = self._visual_analysis(company_name)
            results["visual_analysis"] = visual_result

        # 4. 종합 리포트 생성
        report = self._generate_report(company_name, results, analysis_type)
        results["report"] = report

        return results

    def _generate_report(self, company: str, analyses: dict, analysis_type: str) -> str:
        """종합 분석 리포트 생성"""
        context = ""
        if "text_analysis" in analyses:
            context += f"\n텍스트 분석:\n{analyses['text_analysis']}"
        if "graph_analysis" in analyses:
            context += f"\n관계 분석:\n{analyses['graph_analysis']}"

        prompt = f"""다음 분석 결과를 바탕으로 {company}에 대한 종합 {analysis_type} 리포트를 작성하세요.

{context}

리포트 형식:
## {company} {analysis_type} 리포트

### 1. 기업 개요
### 2. 핵심 분석 결과
### 3. 경쟁 환경
### 4. 투자 포인트 및 리스크
### 5. 종합 의견"""

        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

---

## 3. Gradio 기업 분석 UI

```python
# app.py 핵심 UI 구조

import gradio as gr
from src.company_analyzer import CompanyAnalyzer

analyzer = CompanyAnalyzer(config)

ANALYSIS_TYPES = [
    "재무 분석",
    "경쟁사 비교",
    "사업 전략 분석",
    "리스크 분석",
    "투자 매력도 평가",
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 AI 기업 분석 시스템")
    gr.Markdown("> 사업보고서, 지식 그래프, 멀티모달 분석 통합 플랫폼")

    with gr.Tabs():
        # 탭 1: 기업 분석
        with gr.TabItem("🏢 기업 분석"):
            with gr.Row():
                company_input = gr.Textbox(
                    label="기업명",
                    placeholder="예) 삼성전자, SK하이닉스",
                )
                analysis_type = gr.Dropdown(
                    choices=ANALYSIS_TYPES,
                    value="재무 분석",
                    label="분석 유형",
                )
            analyze_btn = gr.Button("분석 시작", variant="primary")
            analysis_output = gr.Markdown(label="분석 결과")

            analyze_btn.click(
                fn=lambda company, atype: analyzer.analyze(company, atype)["report"],
                inputs=[company_input, analysis_type],
                outputs=analysis_output,
            )

        # 탭 2: 기업 비교
        with gr.TabItem("⚖️ 기업 비교"):
            companies_input = gr.Textbox(
                label="비교 기업 (쉼표로 구분)",
                placeholder="삼성전자, SK하이닉스, TSMC",
            )
            compare_btn = gr.Button("비교 분석", variant="primary")
            compare_output = gr.Markdown()

        # 탭 3: 챗봇
        with gr.TabItem("💬 기업 Q&A"):
            chatbot = gr.Chatbot(height=400)
            chat_input = gr.Textbox(placeholder="기업에 대해 질문하세요...")
            chat_btn = gr.Button("질문하기")

demo.launch(server_port=7864)
```

---

## 4. 샘플 기업 데이터

```python
# data/sample_company_info.json 구조
sample_data = {
    "companies": [
        {
            "name": "삼성전자",
            "ticker": "005930",
            "sector": "반도체/전자",
            "employees": 270372,
            "headquarters": "수원",
            "founded": 1969,
            "description": "메모리 반도체, 스마트폰, 가전 등 글로벌 전자기업",
            "key_products": ["DRAM", "NAND Flash", "갤럭시", "파운드리"],
            "competitors": ["SK하이닉스", "TSMC", "Apple", "Micron"],
            "customers": ["Apple", "Qualcomm", "Google", "Amazon"],
        },
        # ...
    ]
}
```

---

## 5. 분석 파이프라인 최적화

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelAnalyzer:
    """병렬 분석으로 속도 최적화"""

    def __init__(self, analyzer: CompanyAnalyzer):
        self.analyzer = analyzer
        self.executor = ThreadPoolExecutor(max_workers=3)

    async def parallel_analysis(self, company: str) -> dict:
        """여러 분석을 병렬로 실행"""
        loop = asyncio.get_event_loop()

        # 병렬로 3가지 분석 실행
        financial_task = loop.run_in_executor(
            self.executor,
            self.analyzer._text_rag_analysis,
            company, "재무 분석"
        )
        graph_task = loop.run_in_executor(
            self.executor,
            self.analyzer._graph_analysis,
            company
        )
        risk_task = loop.run_in_executor(
            self.executor,
            self.analyzer._text_rag_analysis,
            company, "리스크 분석"
        )

        financial, graph, risk = await asyncio.gather(
            financial_task, graph_task, risk_task
        )

        return {
            "financial": financial,
            "relationships": graph,
            "risks": risk,
        }
```

---

## 핵심 정리

**Project 4 기술 스택**:
| 기능 | 기술 |
|------|------|
| 문서 파싱 | unstructured (PDF/HTML) |
| 지식 그래프 | Neo4j + Cypher |
| 이미지 분석 | CLIP + GPT-4o Vision |
| 벡터 검색 | FAISS 멀티모달 |
| 그래프 쿼리 | LangChain GraphCypherQAChain |
| UI | Gradio Blocks |
