"""
CompanyAnalyzer: Main orchestrator for company analysis pipeline.

Coordinates:
- Document parsing
- Multimodal processing
- Vector store indexing
- Knowledge graph construction
- Graph RAG queries
- Report generation
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .document_parser import ParsedDocument, UnstructuredDocumentParser
from .graph_rag import GraphRAG, GraphRAGAnswer
from .knowledge_graph_builder import KnowledgeGraphBuilder
from .multimodal_processor import MultimodalDocument, MultimodalProcessor
from .multimodal_vector_store import MultimodalVectorStore, SearchResult


ANALYSIS_TYPE_PROMPTS = {
    "재무분석": (
        "재무제표, 매출, 영업이익, 순이익, 부채비율, 유동비율, ROE, ROA 등 "
        "재무 지표를 중심으로 분석하세요. 전년 대비 변화와 업계 평균과의 비교를 포함하세요."
    ),
    "사업분석": (
        "핵심 사업 부문, 제품/서비스 포트폴리오, 시장 점유율, 경쟁 환경, "
        "성장 전략, R&D 투자, 글로벌 사업 현황을 중심으로 분석하세요."
    ),
    "리스크분석": (
        "사업 리스크, 시장 리스크, 규제 리스크, 운영 리스크, ESG 리스크, "
        "지정학적 리스크, 기술 리스크 등을 식별하고 심각도를 평가하세요."
    ),
    "종합분석": (
        "재무 건전성, 사업 경쟁력, 성장 가능성, 리스크 요인을 종합적으로 "
        "평가하고 투자 관점에서의 인사이트를 제공하세요."
    ),
}

COMPARISON_PROMPT_TEMPLATE = """
다음 두 기업을 {aspect} 측면에서 비교 분석하세요.

{company1} 정보:
{context1}

{company2} 정보:
{context2}

비교 분석 요소:
1. 핵심 지표 비교
2. 강점과 약점
3. 경쟁 우위 요소
4. 향후 전망
5. 투자자 관점에서의 시사점

구체적인 수치와 사실을 포함하여 전문적으로 분석하세요.
"""

REPORT_GENERATION_PROMPT = """
{company_name}에 대한 종합 분석 보고서를 작성하세요.

컨텍스트:
{context}

보고서 구조:
1. 기업 개요
   - 핵심 사업 영역
   - 시장 지위
   - 최근 주요 이슈

2. 재무 분석
   - 매출 및 수익성 추이
   - 재무 건전성 지표
   - 현금흐름 분석

3. 사업 분석
   - 사업 부문별 현황
   - 경쟁 환경
   - 성장 동력

4. 리스크 분석
   - 주요 리스크 요인
   - 리스크 완화 방안

5. 투자 시사점
   - 긍정적 요인
   - 부정적 요인
   - 종합 의견

전문적이고 객관적인 투자 리포트 형식으로 작성하세요. 한국어로 작성하세요.
"""


@dataclass
class AnalysisSession:
    """Tracks an analysis session for a company."""
    session_id: str
    company_name: str
    pdf_path: Optional[str] = None
    status: str = "initialized"
    parsed_doc: Optional[ParsedDocument] = None
    graph_built: bool = False
    vector_store_built: bool = False
    start_time: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)


class CompanyAnalyzer:
    """
    Main orchestrator for Korean company analysis.

    Manages the full pipeline from PDF parsing to Q&A and report generation.
    Supports multiple companies simultaneously through session management.
    """

    def __init__(self):
        config.ensure_directories()

        self.parser = UnstructuredDocumentParser()
        self.processor = MultimodalProcessor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.vector_store = MultimodalVectorStore(processor=self.processor)
        self.graph_rag = GraphRAG(
            graph_builder=self.graph_builder,
            vector_store=self.vector_store,
        )

        self._sessions: dict[str, AnalysisSession] = {}
        self._openai_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=config.openai.api_key)
        return self._openai_client

    # --- Main Pipeline --------------------------------------------------

    def analyze_business_report(
        self,
        pdf_path: Union[str, Path],
        company_name: str,
        extract_images: bool = True,
        build_graph: bool = True,
        build_vector_store: bool = True,
    ) -> AnalysisSession:
        """
        Run the full analysis pipeline on a business report PDF.

        Steps:
        1. Parse PDF with unstructured
        2. Process images with CLIP/GPT-4V
        3. Build multimodal vector store
        4. Build Neo4j knowledge graph
        5. Build community summaries

        Args:
            pdf_path: Path to the business report PDF
            company_name: Name of the company
            extract_images: Extract and process images
            build_graph: Build Neo4j knowledge graph
            build_vector_store: Build FAISS vector store

        Returns:
            AnalysisSession with status and parsed data
        """
        session_id = f"session_{company_name}_{int(time.time())}"
        session = AnalysisSession(
            session_id=session_id,
            company_name=company_name,
            pdf_path=str(pdf_path),
        )
        self._sessions[session_id] = session

        try:
            logger.info(f"Starting analysis for {company_name}: {pdf_path}")
            session.status = "parsing"

            # Step 1: Parse PDF
            output_dir = config.paths.parsed_docs_dir / company_name
            parsed_doc = self.parser.parse_business_report(
                pdf_path=pdf_path,
                company_name=company_name,
                extract_images=extract_images,
                output_image_dir=output_dir / "images",
            )
            self.parser.save_parsed(parsed_doc, output_dir)
            session.parsed_doc = parsed_doc
            session.status = "parsed"

            # Step 2: Build vector store
            if build_vector_store:
                session.status = "building_vector_store"
                self._build_vector_store(parsed_doc, company_name)
                session.vector_store_built = True

            # Step 3: Build knowledge graph
            if build_graph:
                session.status = "building_graph"
                try:
                    self.graph_builder.connect()
                    self.graph_builder.build_from_document(parsed_doc, company_name)
                    session.graph_built = True

                    # Build community summaries
                    self.graph_rag.build_community_summaries()
                except Exception as e:
                    logger.warning(f"Graph build failed (non-fatal): {e}")
                    session.errors.append(f"Graph build: {str(e)}")

            session.status = "ready"
            elapsed = time.time() - session.start_time
            logger.info(
                f"Analysis complete for {company_name} in {elapsed:.1f}s. "
                f"Session: {session_id}"
            )

        except Exception as e:
            session.status = "error"
            session.errors.append(str(e))
            logger.error(f"Analysis failed for {company_name}: {e}")
            raise

        return session

    def _build_vector_store(
        self,
        parsed_doc: ParsedDocument,
        company_name: str,
    ) -> None:
        """Build FAISS indexes from parsed document."""
        # Create text documents
        text_docs = []
        for elem in parsed_doc.all_elements:
            if elem.text and len(elem.text) > 50:
                doc = self.processor.create_multimodal_document(
                    text=elem.text,
                    metadata={
                        "company": company_name,
                        "page": elem.page_number,
                        "element_type": elem.element_type.value,
                        "section": (
                            " > ".join(elem.section_hierarchy[-2:])
                            if elem.section_hierarchy else ""
                        ),
                    },
                )
                text_docs.append(doc)

        # Process table elements as text
        for table in parsed_doc.tables:
            if table.table_data:
                table_text = self._table_to_text(table.table_data)
                doc = self.processor.create_multimodal_document(
                    text=table_text,
                    metadata={
                        "company": company_name,
                        "page": table.page_number,
                        "element_type": "table",
                        "is_financial": table.element_type.value == "financial_table",
                    },
                )
                text_docs.append(doc)

        # Process images
        image_docs = []
        for img_elem in parsed_doc.images:
            if img_elem.image_path and Path(img_elem.image_path).exists():
                caption = img_elem.text or f"{company_name} 이미지 (p.{img_elem.page_number})"
                image_docs.append((img_elem.image_path, caption))

        logger.info(
            f"Building vector store: {len(text_docs)} text docs, "
            f"{len(image_docs)} image docs"
        )

        self.vector_store.build(text_docs, image_docs)

        # Save vector store
        vs_path = config.paths.faiss_index_path / company_name
        self.vector_store.save(vs_path)
        logger.info(f"Vector store saved: {vs_path}")

    @staticmethod
    def _table_to_text(table_data: list[list[str]]) -> str:
        """Convert table data to a readable text format."""
        if not table_data:
            return ""
        rows = [" | ".join(str(c) for c in row) for row in table_data]
        return "\n".join(rows)

    # --- Query Interface ------------------------------------------------

    def query(
        self,
        question: str,
        analysis_type: str = "종합분석",
        company_name: Optional[str] = None,
        use_graph: bool = True,
        use_vector: bool = True,
    ) -> dict:
        """
        Answer a question about analyzed companies.

        Selects the appropriate search strategy based on parameters.

        Args:
            question: User question
            analysis_type: One of 재무분석, 사업분석, 리스크분석, 종합분석
            company_name: Optional company to focus on
            use_graph: Use knowledge graph context
            use_vector: Use vector store context

        Returns:
            Dict with answer and supporting context
        """
        # Augment question with analysis type context
        type_context = ANALYSIS_TYPE_PROMPTS.get(analysis_type, "")
        augmented_question = (
            f"{question}\n\n분석 관점: {type_context}" if type_context else question
        )

        if company_name:
            augmented_question = f"[{company_name}] {augmented_question}"

        try:
            if use_graph and use_vector:
                result = self.graph_rag.hybrid_search(augmented_question, k=5)
            elif use_graph:
                result = self.graph_rag.local_search(
                    augmented_question, entity_name=company_name
                )
            elif use_vector:
                results = self.vector_store.text_search(augmented_question, k=5)
                context = "\n\n".join(r.text for r in results)
                answer = self.graph_rag.generate_answer(
                    augmented_question, context
                )
                result = GraphRAGAnswer(
                    answer=answer,
                    search_type="vector_only",
                    entities_used=[],
                    communities_used=[],
                    context_snippets=[context[:500]],
                    confidence=0.7,
                )
            else:
                result = GraphRAGAnswer(
                    answer="검색 방법을 지정해 주세요.",
                    search_type="none",
                    entities_used=[],
                    communities_used=[],
                    context_snippets=[],
                    confidence=0.0,
                )

            return {
                "answer": result.answer,
                "search_type": result.search_type,
                "entities_used": result.entities_used,
                "communities_used": result.communities_used,
                "confidence": result.confidence,
                "analysis_type": analysis_type,
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                "search_type": "error",
                "entities_used": [],
                "communities_used": [],
                "confidence": 0.0,
                "analysis_type": analysis_type,
            }

    def query_with_image(
        self,
        question: str,
        image_path: Union[str, Path],
        analysis_type: str = "종합분석",
    ) -> dict:
        """
        Answer a question with an image as additional context.

        Uses GPT-4V to understand the image and combines with graph/vector context.

        Args:
            question: User question
            image_path: Path to image file
            analysis_type: Analysis focus

        Returns:
            Dict with answer
        """
        image_path = Path(image_path)

        # Get image description
        try:
            img_description = self.processor.describe_image_with_vlm(
                image_path,
                prompt=(
                    f"이 이미지와 관련하여 다음 질문에 답하는 데 도움이 될 "
                    f"정보를 설명하세요: {question}"
                ),
            )
        except Exception as e:
            logger.warning(f"Image description failed: {e}")
            img_description = ""

        # Find similar content in vector store
        try:
            img_emb = self.processor.encode_image_clip(image_path)
            similar_docs = self.vector_store.image_to_text_search(img_emb, k=3)
            vector_context = "\n".join(d.text for d in similar_docs)
        except Exception:
            vector_context = ""

        # Build combined context
        combined_context = ""
        if img_description:
            combined_context += f"이미지 내용:\n{img_description}\n\n"
        if vector_context:
            combined_context += f"관련 문서 내용:\n{vector_context}"

        if not combined_context:
            combined_context = "관련 컨텍스트를 찾을 수 없습니다."

        type_context = ANALYSIS_TYPE_PROMPTS.get(analysis_type, "")
        full_question = (
            f"{question}\n\n분석 관점: {type_context}" if type_context else question
        )

        answer = self.graph_rag.generate_answer(full_question, combined_context)

        return {
            "answer": answer,
            "image_description": img_description,
            "search_type": "multimodal",
            "confidence": 0.8 if img_description else 0.4,
            "analysis_type": analysis_type,
        }

    # --- Company Comparison --------------------------------------------

    def compare_companies(
        self,
        company1: str,
        company2: str,
        aspect: str = "전반적인 사업 경쟁력",
    ) -> dict:
        """
        Compare two analyzed companies on a given aspect.

        Retrieves context for both companies and uses LLM for comparison.

        Args:
            company1: First company name
            company2: Second company name
            aspect: Aspect to compare (e.g., 재무 건전성, 성장성, 수익성)

        Returns:
            Dict with comparison text and structured scores
        """
        logger.info(f"Comparing {company1} vs {company2} on: {aspect}")

        # Get context for each company
        def get_company_context(name: str) -> str:
            parts = []

            # Graph context
            try:
                ctx = self.graph_rag.get_entity_context(name)
                props_text = json.dumps(ctx.properties, ensure_ascii=False, indent=2)
                neighbors_text = "\n".join(
                    f"- {n['name']} ({n['type']}): {n['relationship']}"
                    for n in ctx.neighbors[:8]
                )
                parts.append(f"기업 정보:\n{props_text}")
                parts.append(f"관계:\n{neighbors_text}")
            except Exception as e:
                logger.warning(f"Graph context failed for {name}: {e}")

            # Vector context
            try:
                results = self.vector_store.text_search(
                    f"{name} {aspect}", k=3
                )
                if results:
                    parts.append(
                        "문서 내용:\n" + "\n".join(r.text[:300] for r in results)
                    )
            except Exception as e:
                logger.warning(f"Vector context failed for {name}: {e}")

            return "\n\n".join(parts) if parts else f"{name}에 대한 정보 없음"

        context1 = get_company_context(company1)
        context2 = get_company_context(company2)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            aspect=aspect,
            company1=company1,
            company2=company2,
            context1=context1[:2000],
            context2=context2[:2000],
        )

        response = self.openai_client.chat.completions.create(
            model=config.openai.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국 주식시장 전문 애널리스트입니다. "
                        "두 기업을 객관적으로 비교 분석하세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )

        comparison_text = response.choices[0].message.content or ""

        # Extract structured scores if possible
        scores = self._extract_comparison_scores(
            comparison_text, company1, company2
        )

        return {
            "comparison": comparison_text,
            "company1": company1,
            "company2": company2,
            "aspect": aspect,
            "scores": scores,
        }

    def _extract_comparison_scores(
        self,
        comparison_text: str,
        company1: str,
        company2: str,
    ) -> dict:
        """Extract numerical comparison scores from the comparison text."""
        try:
            score_prompt = f"""다음 비교 분석 텍스트에서 두 기업의 상대적 강점을 점수화하세요.
0-100점 기준으로, JSON으로만 응답:
{{"company1_score": 숫자, "company2_score": 숫자, "winner": "회사명 또는 동등"}}

분석 텍스트 (앞부분):
{comparison_text[:1000]}"""

            response = self.openai_client.chat.completions.create(
                model=config.openai.llm_model,
                messages=[{"role": "user", "content": score_prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception:
            return {
                "company1_score": 50,
                "company2_score": 50,
                "winner": "동등",
            }

    # --- Report Generation ---------------------------------------------

    def generate_analysis_report(
        self,
        company_name: str,
        include_sections: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate a comprehensive analysis report for a company.

        Args:
            company_name: Target company name
            include_sections: Optional list of sections to include.
                              Defaults to all sections.

        Returns:
            Dict with 'report' (str), 'metadata' (dict), 'sections' (dict)
        """
        logger.info(f"Generating analysis report for: {company_name}")

        sections_to_include = include_sections or [
            "재무분석", "사업분석", "리스크분석"
        ]

        # Gather comprehensive context
        context_parts = []

        # 1. Graph context
        try:
            traversal = self.graph_rag.traverse_relationships(company_name, depth=2)
            node_summary = ", ".join(
                f"{n['name']}({n['type']})"
                for n in traversal.get("nodes", [])[:20]
            )
            if node_summary:
                context_parts.append(f"관련 엔티티: {node_summary}")
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")

        # 2. Section-specific contexts
        section_contexts = {}
        for section in sections_to_include:
            section_query = f"{company_name} {ANALYSIS_TYPE_PROMPTS.get(section, section)}"
            try:
                results = self.vector_store.text_search(section_query, k=4)
                section_text = "\n".join(r.text[:400] for r in results)
                section_contexts[section] = section_text
                context_parts.append(f"[{section}]\n{section_text}")
            except Exception as e:
                logger.warning(f"Vector search failed for {section}: {e}")

        # 3. Community reports
        if self.graph_rag._community_reports:
            relevant = self.graph_rag._rank_communities_for_query(
                company_name, top_k=2
            )
            for rep in relevant:
                context_parts.append(
                    f"[커뮤니티 분석: {rep.title}]\n{rep.summary}"
                )

        full_context = "\n\n---\n\n".join(context_parts)

        # Generate report
        prompt = REPORT_GENERATION_PROMPT.format(
            company_name=company_name,
            context=full_context[:6000],
        )

        response = self.openai_client.chat.completions.create(
            model=config.openai.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 선임 기업 애널리스트입니다. "
                        "전문적이고 구체적인 기업 분석 보고서를 작성하세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        report_text = response.choices[0].message.content or ""

        # Save report
        report_path = config.paths.reports_dir / f"{company_name}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# {company_name} 기업 분석 보고서\n\n")
            f.write(report_text)

        logger.info(f"Report saved: {report_path}")

        return {
            "report": report_text,
            "company_name": company_name,
            "sections_included": sections_to_include,
            "report_path": str(report_path),
            "metadata": {
                "context_length": len(full_context),
                "sections": list(section_contexts.keys()),
            },
        }

    # --- Session Management --------------------------------------------

    def load_company(
        self,
        company_name: str,
        vector_store_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Load a previously analyzed company from disk.

        Args:
            company_name: Company to load
            vector_store_path: Optional custom path for vector store

        Returns:
            True if loaded successfully
        """
        vs_path = vector_store_path or (
            config.paths.faiss_index_path / company_name
        )

        try:
            self.vector_store.load(vs_path)
            logger.info(f"Loaded vector store for {company_name}")

            # Rebuild graph community summaries if graph is connected
            try:
                self.graph_builder.connect()
                self.graph_rag.build_community_summaries()
            except Exception as e:
                logger.warning(f"Could not load graph context: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to load {company_name}: {e}")
            return False

    def list_sessions(self) -> list[dict]:
        """Return summary of all analysis sessions."""
        return [
            {
                "session_id": s.session_id,
                "company": s.company_name,
                "status": s.status,
                "graph_built": s.graph_built,
                "vector_store_built": s.vector_store_built,
                "elapsed_s": round(time.time() - s.start_time, 1),
                "errors": s.errors,
            }
            for s in self._sessions.values()
        ]

    def get_vector_store_stats(self) -> dict:
        return self.vector_store.stats()

    def get_graph_stats(self) -> dict:
        try:
            if not self.graph_builder._driver:
                self.graph_builder.connect()
            return self.graph_builder.get_company_stats()
        except Exception as e:
            return {"error": str(e)}
