"""
Gradio Web Application for Company Analysis Agent.

Features:
- PDF upload for Korean business reports (사업보고서)
- Analysis type selection (재무분석, 사업분석, 리스크분석)
- Multimodal Q&A (text + image input)
- Knowledge graph visualization (pyvis)
- Company comparison
- Full report generation
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr
from loguru import logger

from src.company_analyzer import CompanyAnalyzer
from src.config import config

# -----------------------------------------------------------------------
# Initialize analyzer (lazy)
# -----------------------------------------------------------------------
_analyzer: Optional[CompanyAnalyzer] = None
_analyzed_companies: list[str] = []


def get_analyzer() -> CompanyAnalyzer:
    global _analyzer
    if _analyzer is None:
        logger.info("Initializing CompanyAnalyzer...")
        _analyzer = CompanyAnalyzer()
    return _analyzer


# -----------------------------------------------------------------------
# Gradio callback functions
# -----------------------------------------------------------------------

def upload_and_analyze_pdf(
    pdf_file,
    company_name: str,
    extract_images: bool,
    build_graph: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Handle PDF upload and run analysis pipeline."""
    if pdf_file is None:
        return "PDF 파일을 선택해 주세요.", ""

    if not company_name.strip():
        return "기업명을 입력해 주세요.", ""

    analyzer = get_analyzer()
    company_name = company_name.strip()

    progress(0.05, desc="분석 준비 중...")

    try:
        # Save uploaded file to temp location
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)

        progress(0.15, desc="PDF 파싱 중...")
        session = analyzer.analyze_business_report(
            pdf_path=pdf_path,
            company_name=company_name,
            extract_images=extract_images,
            build_graph=build_graph,
            build_vector_store=True,
        )

        progress(1.0, desc="완료!")

        if company_name not in _analyzed_companies:
            _analyzed_companies.append(company_name)

        # Build status summary
        doc = session.parsed_doc
        stats = {
            "총 페이지": doc.total_pages if doc else "N/A",
            "총 요소": len(doc.all_elements) if doc else "N/A",
            "테이블 수": len(doc.tables) if doc else "N/A",
            "이미지 수": len(doc.images) if doc else "N/A",
            "섹션 수": len(doc.sections) if doc else "N/A",
            "그래프 구축": "완료" if session.graph_built else "건너뜀",
            "벡터 스토어": "완료" if session.vector_store_built else "오류",
            "오류": ", ".join(session.errors) if session.errors else "없음",
        }

        stats_text = "\n".join(f"• {k}: {v}" for k, v in stats.items())
        status_msg = (
            f"✅ {company_name} 분석 완료!\n\n"
            f"📊 문서 통계:\n{stats_text}\n\n"
            f"이제 아래 탭에서 질문하거나 보고서를 생성하세요."
        )

        vs_stats = analyzer.get_vector_store_stats()
        vs_text = json.dumps(vs_stats, ensure_ascii=False, indent=2)

        return status_msg, vs_text

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"❌ 오류 발생: {str(e)}", ""


def load_sample_company(company_name: str) -> tuple[str, str]:
    """Load sample company data from JSON without PDF."""
    sample_path = Path("data/sample_company_info.json")
    if not sample_path.exists():
        return "샘플 데이터 파일을 찾을 수 없습니다.", ""

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            companies = json.load(f)

        analyzer = get_analyzer()
        target = None
        for company in companies:
            if company_name in company.get("name", ""):
                target = company
                break

        if not target:
            available = ", ".join(c.get("name", "") for c in companies)
            return f"'{company_name}'을 찾을 수 없습니다. 사용 가능: {available}", ""

        try:
            analyzer.graph_builder.connect()
            company_id = analyzer.graph_builder.create_company_node(target)
        except Exception as e:
            logger.warning(f"Graph load failed: {e}")

        if company_name not in _analyzed_companies:
            _analyzed_companies.append(company_name)

        return (
            f"✅ {target['name']} 샘플 데이터 로드 완료!\n\n"
            f"CEO: {target.get('ceo', 'N/A')}\n"
            f"시장: {target.get('market', 'N/A')}\n"
            f"섹터: {target.get('sector', 'N/A')}\n\n"
            f"그래프 탭 또는 Q&A 탭에서 질문하세요.",
            json.dumps(target.get("financials", {}), ensure_ascii=False, indent=2),
        )

    except Exception as e:
        return f"❌ 로드 실패: {str(e)}", ""


def answer_question(
    question: str,
    company_name: str,
    analysis_type: str,
    use_graph: bool,
    use_vector: bool,
) -> tuple[str, str]:
    """Answer a text question about a company."""
    if not question.strip():
        return "질문을 입력해 주세요.", ""

    analyzer = get_analyzer()

    try:
        result = analyzer.query(
            question=question,
            analysis_type=analysis_type,
            company_name=company_name.strip() if company_name.strip() else None,
            use_graph=use_graph,
            use_vector=use_vector,
        )

        answer = result["answer"]
        meta = (
            f"검색 방식: {result['search_type']}\n"
            f"신뢰도: {result['confidence']:.1%}\n"
            f"사용된 엔티티: {', '.join(result['entities_used'][:5]) or '없음'}\n"
            f"사용된 커뮤니티: {result['communities_used'] or '없음'}"
        )

        return answer, meta

    except Exception as e:
        logger.error(f"Query error: {e}")
        return f"오류 발생: {str(e)}", ""


def answer_multimodal_question(
    question: str,
    image_file,
    company_name: str,
    analysis_type: str,
) -> tuple[str, str]:
    """Answer a question with image context."""
    if not question.strip():
        return "질문을 입력해 주세요.", ""

    analyzer = get_analyzer()

    if image_file is None:
        # Fallback to text-only
        return answer_question(question, company_name, analysis_type, True, True)

    try:
        image_path = image_file.name if hasattr(image_file, "name") else str(image_file)
        result = analyzer.query_with_image(
            question=question,
            image_path=image_path,
            analysis_type=analysis_type,
        )

        meta = (
            f"검색 방식: 멀티모달\n"
            f"신뢰도: {result['confidence']:.1%}\n\n"
            f"이미지 설명 (요약):\n{result.get('image_description', '')[:300]}"
        )

        return result["answer"], meta

    except Exception as e:
        logger.error(f"Multimodal query error: {e}")
        return f"오류 발생: {str(e)}", ""


def compare_companies_fn(
    company1: str,
    company2: str,
    aspect: str,
) -> tuple[str, str]:
    """Compare two companies."""
    if not company1.strip() or not company2.strip():
        return "두 기업명을 모두 입력해 주세요.", ""

    if company1.strip() == company2.strip():
        return "서로 다른 기업을 선택해 주세요.", ""

    analyzer = get_analyzer()

    try:
        result = analyzer.compare_companies(
            company1=company1.strip(),
            company2=company2.strip(),
            aspect=aspect.strip() or "전반적인 사업 경쟁력",
        )

        scores = result.get("scores", {})
        score_text = (
            f"{company1}: {scores.get('company1_score', 'N/A')}점\n"
            f"{company2}: {scores.get('company2_score', 'N/A')}점\n"
            f"우위: {scores.get('winner', 'N/A')}"
        )

        return result["comparison"], score_text

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return f"오류 발생: {str(e)}", ""


def generate_report_fn(
    company_name: str,
    include_financial: bool,
    include_business: bool,
    include_risk: bool,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Generate a full analysis report."""
    if not company_name.strip():
        return "기업명을 입력해 주세요.", ""

    analyzer = get_analyzer()

    sections = []
    if include_financial:
        sections.append("재무분석")
    if include_business:
        sections.append("사업분석")
    if include_risk:
        sections.append("리스크분석")

    if not sections:
        sections = ["재무분석", "사업분석", "리스크분석"]

    progress(0.1, desc="보고서 생성 중...")

    try:
        result = analyzer.generate_analysis_report(
            company_name=company_name.strip(),
            include_sections=sections,
        )
        progress(1.0, desc="완료!")

        return (
            result["report"],
            f"보고서 저장 경로: {result['report_path']}"
        )

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return f"오류 발생: {str(e)}", ""


def visualize_graph_fn(company_name: str) -> str:
    """Generate an HTML knowledge graph visualization using pyvis."""
    if not company_name.strip():
        return "<p>기업명을 입력해 주세요.</p>"

    analyzer = get_analyzer()

    try:
        from pyvis.network import Network

        traversal = analyzer.graph_rag.traverse_relationships(
            company_name.strip(), depth=2
        )

        nodes = traversal.get("nodes", [])
        edges = traversal.get("edges", [])

        if not nodes:
            return f"<p>'{company_name}' 관련 그래프 데이터가 없습니다. 먼저 분석을 실행하세요.</p>"

        net = Network(
            height="600px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True,
        )
        net.set_options("""
        {
          "nodes": {
            "borderWidth": 2,
            "size": 20,
            "font": {"size": 12}
          },
          "edges": {
            "color": {"inherit": true},
            "smooth": {"type": "continuous"}
          },
          "physics": {
            "stabilization": {"iterations": 100}
          }
        }
        """)

        # Color scheme by node type
        color_map = {
            "Company": "#4ecdc4",
            "Executive": "#45b7d1",
            "Subsidiary": "#96ceb4",
            "Product": "#ffeaa7",
            "FinancialMetric": "#fd79a8",
            "Risk": "#e17055",
            "Market": "#a29bfe",
        }

        added_nodes = set()
        for node in nodes:
            name = node.get("name", "Unknown")
            ntype = node.get("type", "Unknown")
            if name and name not in added_nodes:
                color = color_map.get(ntype, "#dfe6e9")
                size = 30 if ntype == "Company" else 20
                net.add_node(
                    name,
                    label=name[:20],
                    title=f"{ntype}: {name}",
                    color=color,
                    size=size,
                )
                added_nodes.add(name)

        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            rels = edge.get("relationships", [])
            if from_node in added_nodes and to_node in added_nodes:
                label = rels[0] if rels else ""
                net.add_edge(from_node, to_node, label=label, title=label)

        # Save to temp file and return HTML
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            html_content = open(f.name, encoding="utf-8").read()

        os.unlink(f.name)
        return html_content

    except ImportError:
        # Fallback: simple networkx visualization description
        try:
            traversal = analyzer.graph_rag.traverse_relationships(
                company_name.strip(), depth=2
            )
            nodes = traversal.get("nodes", [])
            edges = traversal.get("edges", [])

            node_list = "\n".join(
                f"• {n['name']} ({n['type']})" for n in nodes[:20]
            )
            edge_list = "\n".join(
                f"• {e['from']} → {e['to']} [{', '.join(e.get('relationships', []))}]"
                for e in edges[:20]
            )

            return (
                f"<pre style='color:white; background:#1a1a2e; padding:16px;'>"
                f"<b>노드 ({len(nodes)}개):</b>\n{node_list}\n\n"
                f"<b>엣지 ({len(edges)}개):</b>\n{edge_list}"
                f"</pre>"
            )
        except Exception as e2:
            return f"<p>그래프 시각화 오류: {str(e2)}</p>"

    except Exception as e:
        logger.error(f"Graph visualization error: {e}")
        return f"<p>그래프 시각화 오류: {str(e)}</p>"


def get_graph_stats_fn() -> str:
    """Return graph database statistics."""
    try:
        analyzer = get_analyzer()
        stats = analyzer.get_graph_stats()
        return json.dumps(stats, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"통계 조회 오류: {str(e)}"


# -----------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="기업 분석 AI 에이전트",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        ),
        css="""
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status-box { font-family: monospace; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>🏢 기업 분석 AI 에이전트</h1>
            <p>Multimodal RAG + Graph RAG 기반 한국 상장기업 사업보고서 분석 시스템</p>
        </div>
        """)

        # ---------------------------------------------------------------
        # Tab 1: Document Upload & Analysis
        # ---------------------------------------------------------------
        with gr.Tab("📄 문서 업로드 & 분석"):
            gr.Markdown("### 사업보고서 PDF 업로드")

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.File(
                        label="사업보고서 PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    company_name_input = gr.Textbox(
                        label="기업명",
                        placeholder="예: 삼성전자, SK하이닉스, NAVER",
                    )
                    with gr.Row():
                        extract_images_cb = gr.Checkbox(
                            label="이미지/차트 추출", value=True
                        )
                        build_graph_cb = gr.Checkbox(
                            label="지식 그래프 구축", value=True
                        )
                    analyze_btn = gr.Button(
                        "🚀 분석 시작", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 샘플 데이터 로드")
                    gr.Markdown("PDF 없이 샘플 데이터로 테스트:")
                    sample_company = gr.Dropdown(
                        label="샘플 기업 선택",
                        choices=[
                            "삼성전자", "SK하이닉스", "NAVER", "카카오", "LG에너지솔루션"
                        ],
                        value="삼성전자",
                    )
                    load_sample_btn = gr.Button("📥 샘플 로드", variant="secondary")

            with gr.Row():
                with gr.Column():
                    analysis_status = gr.Textbox(
                        label="분석 상태",
                        lines=10,
                        elem_classes=["status-box"],
                    )
                with gr.Column():
                    vector_stats = gr.Textbox(
                        label="벡터 스토어 통계",
                        lines=10,
                        elem_classes=["status-box"],
                    )

            analyze_btn.click(
                fn=upload_and_analyze_pdf,
                inputs=[
                    pdf_input,
                    company_name_input,
                    extract_images_cb,
                    build_graph_cb,
                ],
                outputs=[analysis_status, vector_stats],
            )

            load_sample_btn.click(
                fn=load_sample_company,
                inputs=[sample_company],
                outputs=[analysis_status, vector_stats],
            )

        # ---------------------------------------------------------------
        # Tab 2: Text Q&A
        # ---------------------------------------------------------------
        with gr.Tab("💬 텍스트 Q&A"):
            gr.Markdown("### 기업 분석 질문 답변")
            gr.Markdown(
                "분석된 기업에 대해 재무, 사업, 리스크 관련 질문을 하세요."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    qa_question = gr.Textbox(
                        label="질문",
                        placeholder="예: 삼성전자의 2023년 영업이익은 얼마인가요?",
                        lines=3,
                    )
                    with gr.Row():
                        qa_company = gr.Textbox(
                            label="기업명 (선택)",
                            placeholder="특정 기업에 집중하려면 입력",
                        )
                        qa_analysis_type = gr.Dropdown(
                            label="분석 유형",
                            choices=["재무분석", "사업분석", "리스크분석", "종합분석"],
                            value="종합분석",
                        )
                    with gr.Row():
                        qa_use_graph = gr.Checkbox(label="그래프 RAG 사용", value=True)
                        qa_use_vector = gr.Checkbox(label="벡터 RAG 사용", value=True)
                    qa_btn = gr.Button("🔍 질문하기", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 예시 질문")
                    example_questions = [
                        "삼성전자의 반도체 사업 경쟁력은?",
                        "SK하이닉스의 주요 리스크 요인은 무엇인가요?",
                        "NAVER의 광고 매출 비중은 얼마나 되나요?",
                        "카카오의 주요 자회사는 어디인가요?",
                        "LG에너지솔루션의 배터리 시장 점유율은?",
                    ]
                    for eq in example_questions:
                        gr.Button(eq, size="sm").click(
                            fn=lambda q=eq: q,
                            outputs=[qa_question],
                        )

            qa_answer = gr.Textbox(label="답변", lines=12)
            qa_meta = gr.Textbox(label="검색 메타데이터", lines=5)

            qa_btn.click(
                fn=answer_question,
                inputs=[qa_question, qa_company, qa_analysis_type, qa_use_graph, qa_use_vector],
                outputs=[qa_answer, qa_meta],
            )

        # ---------------------------------------------------------------
        # Tab 3: Multimodal Q&A
        # ---------------------------------------------------------------
        with gr.Tab("🖼️ 멀티모달 Q&A"):
            gr.Markdown("### 이미지 + 텍스트 질문 답변")
            gr.Markdown(
                "차트, 표, 그래프 이미지를 업로드하고 관련 질문을 하세요."
            )

            with gr.Row():
                with gr.Column():
                    mm_image = gr.Image(
                        label="이미지 업로드 (차트, 표 등)",
                        type="filepath",
                    )
                    mm_question = gr.Textbox(
                        label="질문",
                        placeholder="이 차트에서 어떤 트렌드를 볼 수 있나요?",
                        lines=2,
                    )
                    with gr.Row():
                        mm_company = gr.Textbox(
                            label="기업명 (선택)",
                        )
                        mm_analysis_type = gr.Dropdown(
                            label="분석 유형",
                            choices=["재무분석", "사업분석", "리스크분석", "종합분석"],
                            value="재무분석",
                        )
                    mm_btn = gr.Button("🔍 분석하기", variant="primary")

                with gr.Column():
                    mm_answer = gr.Textbox(label="답변", lines=12)
                    mm_meta = gr.Textbox(label="이미지 분석 상세", lines=8)

            mm_btn.click(
                fn=answer_multimodal_question,
                inputs=[mm_question, mm_image, mm_company, mm_analysis_type],
                outputs=[mm_answer, mm_meta],
            )

        # ---------------------------------------------------------------
        # Tab 4: Knowledge Graph
        # ---------------------------------------------------------------
        with gr.Tab("🕸️ 지식 그래프"):
            gr.Markdown("### 기업 지식 그래프 시각화")

            with gr.Row():
                with gr.Column(scale=2):
                    graph_company = gr.Textbox(
                        label="기업명",
                        placeholder="예: 삼성전자",
                        value="삼성전자",
                    )
                    graph_btn = gr.Button("그래프 시각화", variant="primary")
                    graph_stats_btn = gr.Button("그래프 통계", variant="secondary")

                with gr.Column(scale=1):
                    graph_stats_out = gr.Textbox(
                        label="그래프 통계",
                        lines=10,
                        elem_classes=["status-box"],
                    )

            graph_html = gr.HTML(label="지식 그래프")

            graph_btn.click(
                fn=visualize_graph_fn,
                inputs=[graph_company],
                outputs=[graph_html],
            )
            graph_stats_btn.click(
                fn=get_graph_stats_fn,
                outputs=[graph_stats_out],
            )

            gr.Markdown("""
            **그래프 범례:**
            - 🟦 Company (기업)
            - 🔵 Executive (임원)
            - 🟢 Subsidiary (자회사)
            - 🟡 Product (제품)
            - 🩷 FinancialMetric (재무지표)
            - 🟥 Risk (리스크)
            - 🟣 Market (시장)
            """)

        # ---------------------------------------------------------------
        # Tab 5: Company Comparison
        # ---------------------------------------------------------------
        with gr.Tab("⚖️ 기업 비교"):
            gr.Markdown("### 기업 비교 분석")

            with gr.Row():
                comp1_input = gr.Textbox(
                    label="기업 1",
                    placeholder="예: 삼성전자",
                    value="삼성전자",
                )
                comp2_input = gr.Textbox(
                    label="기업 2",
                    placeholder="예: SK하이닉스",
                    value="SK하이닉스",
                )

            comp_aspect = gr.Textbox(
                label="비교 측면",
                placeholder="예: 반도체 사업 경쟁력, 재무 건전성, 성장 가능성",
                value="반도체 사업 경쟁력 및 재무 성과",
            )

            with gr.Row():
                preset_comparisons = [
                    ("반도체 경쟁력", "삼성전자", "SK하이닉스"),
                    ("플랫폼 성장성", "NAVER", "카카오"),
                    ("배터리 사업", "LG에너지솔루션", "삼성전자"),
                ]
                for label, c1, c2 in preset_comparisons:
                    btn = gr.Button(f"{label}: {c1} vs {c2}", size="sm")
                    btn.click(
                        fn=lambda c1=c1, c2=c2, label=label: (c1, c2, label),
                        outputs=[comp1_input, comp2_input, comp_aspect],
                    )

            compare_btn = gr.Button("⚖️ 비교 분석", variant="primary", size="lg")

            with gr.Row():
                comp_result = gr.Textbox(label="비교 분석 결과", lines=20)
                comp_scores = gr.Textbox(label="상대 점수", lines=5)

            compare_btn.click(
                fn=compare_companies_fn,
                inputs=[comp1_input, comp2_input, comp_aspect],
                outputs=[comp_result, comp_scores],
            )

        # ---------------------------------------------------------------
        # Tab 6: Report Generation
        # ---------------------------------------------------------------
        with gr.Tab("📋 보고서 생성"):
            gr.Markdown("### 종합 분석 보고서 생성")

            with gr.Row():
                with gr.Column():
                    report_company = gr.Textbox(
                        label="기업명",
                        placeholder="예: 삼성전자",
                    )
                    gr.Markdown("**포함할 섹션:**")
                    include_financial = gr.Checkbox(
                        label="재무 분석", value=True
                    )
                    include_business = gr.Checkbox(
                        label="사업 분석", value=True
                    )
                    include_risk = gr.Checkbox(
                        label="리스크 분석", value=True
                    )

                    report_btn = gr.Button(
                        "📋 보고서 생성", variant="primary", size="lg"
                    )

                with gr.Column():
                    gr.Markdown("""
                    **보고서 구조:**
                    1. 기업 개요
                    2. 재무 분석 (매출, 이익, 건전성)
                    3. 사업 분석 (부문, 경쟁, 전략)
                    4. 리스크 분석 (주요 위험 요인)
                    5. 투자 시사점

                    생성된 보고서는 `output/reports/` 디렉토리에 저장됩니다.
                    """)

            report_output = gr.Textbox(
                label="분석 보고서",
                lines=30,
                elem_classes=["status-box"],
            )
            report_path_out = gr.Textbox(label="저장 경로", lines=1)

            report_btn.click(
                fn=generate_report_fn,
                inputs=[
                    report_company,
                    include_financial,
                    include_business,
                    include_risk,
                ],
                outputs=[report_output, report_path_out],
            )

        # ---------------------------------------------------------------
        # Footer
        # ---------------------------------------------------------------
        gr.HTML("""
        <div style="text-align:center; color: #666; padding: 20px; margin-top: 20px;">
            <p>기업 분석 AI 에이전트 | Multimodal RAG + Graph RAG</p>
            <p>기술 스택: OpenAI GPT-4o · CLIP · Neo4j · FAISS · Unstructured · Gradio</p>
        </div>
        """)

    return demo


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    config.ensure_directories()

    demo = build_ui()
    demo.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.gradio_share,
        auth=(
            tuple(config.app.gradio_auth.split(":"))
            if config.app.gradio_auth
            else None
        ),
        show_error=True,
    )
