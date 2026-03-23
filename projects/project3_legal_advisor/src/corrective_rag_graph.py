"""
Corrective RAG LangGraph for the Legal Advisor Agent.

Corrective RAG (CRAG, Yan et al., 2024) evaluates the quality of retrieved
documents and triggers a corrective action (web search) when local documents
are insufficient. It then refines the combined knowledge before generating.

Key decisions:
  - CORRECT: Retrieved docs are relevant → generate from docs
  - AMBIGUOUS: Some docs relevant → combine docs with web search
  - INCORRECT: No relevant docs → fall back entirely to web search
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.agent_state import (
    CorrectiveRAGState,
    DocumentGrade,
    LegalEvidence,
    make_initial_corrective_rag_state,
)
from src.config import get_config
from src.graders import GraderSuite
from src.legal_vector_store import LegalVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CORRECTIVE_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
아래 제공된 법령 문서와 웹 검색 결과를 종합하여 사용자의 법률 질문에 답변하세요.

[법령 문서]
{doc_context}

[웹 검색 결과]
{web_context}

답변 지침:
1. 공식 법령 조항을 우선으로 인용하세요
2. 웹 검색 결과는 최신 정보나 판례 보충에 활용하세요
3. 각 정보의 출처를 명시하세요
4. 상충되는 정보가 있으면 공식 법령을 기준으로 하되, 차이점을 언급하세요
5. 법적 불확실성이 있는 경우 명시하세요
6. 답변은 한국어로 작성하세요""",
        ),
        ("human", "질문: {query}"),
    ]
)

KNOWLEDGE_REFINEMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 법률 정보 정제 전문가입니다.
수집된 문서들에서 핵심적이고 신뢰할 수 있는 법률 정보만 추출하세요.

다음을 수행하세요:
1. 중복 정보 제거
2. 서로 모순되는 정보 식별 및 표시
3. 가장 최신이고 권위 있는 정보 우선
4. 불필요한 절차적 내용 제거

정제된 정보를 구조화된 형태로 제시하세요.""",
        ),
        (
            "human",
            """원본 문서들:
{raw_documents}

웹 검색 결과:
{web_results}

질문: {query}

정제된 핵심 정보:""",
        ),
    ]
)

WEB_ONLY_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
아래 웹 검색 결과를 바탕으로 법률 질문에 답변하세요.
검색 결과에서 법령명, 조문 번호, 판례 번호 등을 구체적으로 인용하세요.

[웹 검색 결과]
{web_context}

주의: 로컬 법령 데이터베이스에서 관련 문서를 찾지 못했으므로
웹 검색 결과에 의존합니다. 정보의 정확성을 확인하시기 바랍니다.""",
        ),
        ("human", "질문: {query}"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_docs(documents: list[Document]) -> str:
    if not documents:
        return "관련 법령 문서 없음"
    parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        header = (
            f"[{i}] {meta.get('law_name', '미상')} "
            f"{meta.get('article_label', '')} "
            f"{meta.get('article_title', '')}".strip()
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_web(results: list[dict[str, Any]]) -> str:
    if not results:
        return "웹 검색 결과 없음"
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")[:500]
        parts.append(f"[웹 {i}] {title}\n출처: {url}\n{content}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(
    documents: list[Document], web_results: list[dict]
) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        meta = doc.metadata
        cite = f"{meta.get('law_name', '')} {meta.get('article_label', '')}".strip()
        if cite and cite not in seen:
            citations.append(cite)
            seen.add(cite)
    for r in web_results:
        url = r.get("url", "")
        if url and url not in seen:
            citations.append(f"{r.get('title', 'Web')} ({url})")
            seen.add(url)
    return citations


def _docs_to_evidence(
    documents: list[Document], web_results: list[dict]
) -> list[LegalEvidence]:
    evidence: list[LegalEvidence] = []
    for doc in documents:
        meta = doc.metadata
        evidence.append(
            LegalEvidence(
                doc_id=meta.get("doc_id", ""),
                law_name=meta.get("law_name", ""),
                article_number=meta.get("article_number", ""),
                article_title=meta.get("article_title", ""),
                content_snippet=doc.page_content[:300],
                source_type="vector_store",
                relevance_score=meta.get("relevance_score", 0.5),
                url=None,
            )
        )
    for r in web_results:
        evidence.append(
            LegalEvidence(
                doc_id="",
                law_name="",
                article_number="",
                article_title=r.get("title", ""),
                content_snippet=r.get("content", "")[:300],
                source_type="web_search",
                relevance_score=r.get("score", 0.5),
                url=r.get("url"),
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# CorrectiveRAGGraph class
# ---------------------------------------------------------------------------


class CorrectiveRAGGraph:
    """
    Corrective RAG LangGraph implementation.

    Retrieves documents, evaluates their quality, performs web search if
    needed, refines knowledge, then generates the final answer.

    Quality assessment outcomes:
    - CORRECT (high relevance):  use docs → generate
    - AMBIGUOUS (mixed):         combine docs + web → refine → generate
    - INCORRECT (low relevance): web search only → generate
    """

    def __init__(
        self,
        vector_store: Optional[LegalVectorStore] = None,
        grader_suite: Optional[GraderSuite] = None,
    ) -> None:
        self._config = get_config()
        self._llm = ChatOpenAI(**self._config.get_llm_kwargs())
        self._vector_store = vector_store or LegalVectorStore()
        self._graders = grader_suite or GraderSuite()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _node_retrieve(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """Initial retrieval from the vector store."""
        node_name = "retrieve_node"
        logger.info(f"[CorrectiveRAG] {node_name}: query='{state['query'][:60]}…'")

        docs: list[Document] = []
        try:
            docs = self._vector_store.hybrid_search(
                query=state["query"],
                k=self._config.retriever_k,
            )
            logger.debug(f"[CorrectiveRAG] Retrieved {len(docs)} documents")
        except Exception as exc:
            logger.error(f"[CorrectiveRAG] Retrieval failed: {exc}")

        return {
            **state,
            "retrieved_documents": docs,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_grade_documents(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """
        Assess quality of retrieved documents.

        Determines CORRECT / AMBIGUOUS / INCORRECT outcome
        by grading each document for relevance.
        """
        node_name = "grade_documents_node"
        logger.info(f"[CorrectiveRAG] {node_name}")

        documents = state.get("retrieved_documents", [])
        query = state["query"]

        if not documents:
            return {
                **state,
                "filtered_documents": [],
                "document_grades": [],
                "relevance_scores": [],
                "needs_web_search": True,
                "node_trace": state.get("node_trace", []) + [node_name],
            }

        filtered: list[Document] = []
        grades: list[DocumentGrade] = []
        scores: list[float] = []

        for doc in documents:
            result = self._graders.relevance.grade(document=doc, query=query)
            scores.append(result.confidence)
            if result.score == "yes":
                grades.append(DocumentGrade.RELEVANT)
                filtered.append(doc)
                doc.metadata["relevance_score"] = result.confidence
            else:
                grades.append(DocumentGrade.NOT_RELEVANT)

        total = len(documents)
        relevant_count = len(filtered)
        relevance_ratio = relevant_count / total if total > 0 else 0.0

        # Determine CRAG decision
        if relevance_ratio >= 0.6:
            # CORRECT: mostly relevant, no web search needed
            needs_web = False
            logger.info(
                f"[CorrectiveRAG] Decision=CORRECT "
                f"({relevant_count}/{total} relevant)"
            )
        elif relevance_ratio >= 0.2:
            # AMBIGUOUS: supplement with web search
            needs_web = True
            logger.info(
                f"[CorrectiveRAG] Decision=AMBIGUOUS "
                f"({relevant_count}/{total} relevant) → web supplement"
            )
        else:
            # INCORRECT: web search required
            needs_web = True
            filtered = []  # don't trust the docs
            logger.info(
                f"[CorrectiveRAG] Decision=INCORRECT "
                f"({relevant_count}/{total} relevant) → web fallback"
            )

        return {
            **state,
            "filtered_documents": filtered,
            "document_grades": grades,
            "relevance_scores": scores,
            "needs_web_search": needs_web,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_web_search(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """
        Web search fallback using Tavily.

        Triggered when INCORRECT or AMBIGUOUS document quality is detected.
        """
        node_name = "web_search_node"
        logger.info(f"[CorrectiveRAG] {node_name}")

        query = state["query"]
        web_results: list[dict[str, Any]] = []

        try:
            from tavily import TavilyClient  # type: ignore

            client = TavilyClient(api_key=self._config.tavily_api_key)
            response = client.search(
                query=f"한국 법률 {query}",
                max_results=self._config.tavily_max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
            web_results = response.get("results", [])
            if response.get("answer"):
                web_results.insert(
                    0,
                    {
                        "title": "Tavily 요약",
                        "url": "tavily-summary",
                        "content": response["answer"],
                        "score": 1.0,
                    },
                )
            logger.info(f"[CorrectiveRAG] Web search returned {len(web_results)} results")
        except ImportError:
            logger.error("[CorrectiveRAG] tavily not installed")
        except Exception as exc:
            logger.error(f"[CorrectiveRAG] Web search failed: {exc}")

        return {
            **state,
            "web_search_results": web_results,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_refine_documents(
        self, state: CorrectiveRAGState
    ) -> CorrectiveRAGState:
        """
        Refine and merge local documents with web search results.

        Uses an LLM to extract the most reliable, non-redundant information
        from the combined sources and create synthetic refined Documents.
        """
        node_name = "refine_documents_node"
        logger.info(f"[CorrectiveRAG] {node_name}")

        filtered_docs = state.get("filtered_documents", [])
        web_results = state.get("web_search_results", [])

        # If no web results, just use filtered docs as-is
        if not web_results:
            return {
                **state,
                "refined_documents": filtered_docs,
                "node_trace": state.get("node_trace", []) + [node_name],
            }

        # If no local docs, create synthetic docs from web results
        if not filtered_docs:
            synthetic_docs = [
                Document(
                    page_content=r.get("content", ""),
                    metadata={
                        "source_type": "web_search",
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "relevance_score": r.get("score", 0.5),
                    },
                )
                for r in web_results
                if r.get("content")
            ]
            return {
                **state,
                "refined_documents": synthetic_docs,
                "node_trace": state.get("node_trace", []) + [node_name],
            }

        # LLM-based knowledge refinement
        chain = KNOWLEDGE_REFINEMENT_PROMPT | self._llm | StrOutputParser()
        try:
            refined_text = chain.invoke(
                {
                    "raw_documents": _format_docs(filtered_docs[:3]),
                    "web_results": _format_web(web_results[:3]),
                    "query": state["query"],
                }
            )
            # Create a single refined Document from the LLM output
            refined_doc = Document(
                page_content=refined_text,
                metadata={
                    "source_type": "refined",
                    "original_count": len(filtered_docs),
                    "web_count": len(web_results),
                    "relevance_score": 0.85,
                },
            )
            refined_documents = [refined_doc] + filtered_docs[:2]
        except Exception as exc:
            logger.warning(f"[CorrectiveRAG] Refinement failed: {exc}, using originals")
            refined_documents = filtered_docs

        return {
            **state,
            "refined_documents": refined_documents,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_generate(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """Generate the final answer from refined knowledge."""
        node_name = "generate_node"
        logger.info(f"[CorrectiveRAG] {node_name}")

        # Use refined docs if available, else fall back
        docs = (
            state.get("refined_documents")
            or state.get("filtered_documents")
            or state.get("retrieved_documents", [])
        )
        web_results = state.get("web_search_results", [])
        query = state["query"]

        # Choose prompt based on available context
        if not docs and web_results:
            chain = WEB_ONLY_GENERATE_PROMPT | self._llm | StrOutputParser()
            answer = chain.invoke(
                {"query": query, "web_context": _format_web(web_results)}
            )
        else:
            chain = CORRECTIVE_GENERATE_PROMPT | self._llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "query": query,
                    "doc_context": _format_docs(docs),
                    "web_context": _format_web(web_results),
                }
            )

        citations = _extract_citations(docs, web_results)
        evidence = _docs_to_evidence(docs, web_results)

        return {
            **state,
            "generated_answer": answer,
            "citations": citations,
            "evidence": evidence,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_validate_answer(
        self, state: CorrectiveRAGState
    ) -> CorrectiveRAGState:
        """Validate the generated answer for hallucination and quality."""
        node_name = "validate_answer_node"
        logger.info(f"[CorrectiveRAG] {node_name}")

        answer = state.get("generated_answer", "")
        docs = (
            state.get("refined_documents")
            or state.get("filtered_documents")
            or state.get("retrieved_documents", [])
        )
        query = state["query"]

        hallucination_result = self._graders.hallucination.grade(
            answer=answer, documents=docs
        )
        answer_result = self._graders.answer.grade(answer=answer, query=query)

        relevance_scores = state.get("relevance_scores", [])
        confidence = self._graders.compute_overall_confidence(
            relevance_scores=relevance_scores,
            hallucination_result=hallucination_result,
            answer_result=answer_result,
        )

        logger.info(
            f"[CorrectiveRAG] hallucination={hallucination_result.score} "
            f"quality={answer_result.score} "
            f"confidence={confidence:.2f}"
        )

        return {
            **state,
            "hallucination_grade": hallucination_result.score,
            "answer_grade": answer_result.score,
            "overall_confidence": confidence,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def _route_after_grade_documents(
        self, state: CorrectiveRAGState
    ) -> Literal["web_search_node", "generate_node"]:
        """Route based on CRAG decision."""
        if state.get("needs_web_search", False):
            return "web_search_node"
        return "generate_node"

    def _route_after_web_search(
        self, state: CorrectiveRAGState
    ) -> Literal["refine_documents_node", "generate_node"]:
        """After web search: refine if we have local docs too, else go directly."""
        filtered = state.get("filtered_documents", [])
        if filtered:
            return "refine_documents_node"
        return "generate_node"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """
        Assemble the Corrective RAG StateGraph.

        Graph topology:
        START → retrieve_node → grade_documents_node
              → [web_search_node | generate_node]   (based on CRAG decision)
        web_search_node → [refine_documents_node | generate_node]
        refine_documents_node → generate_node
        generate_node → validate_answer_node → END
        """
        graph = StateGraph(CorrectiveRAGState)

        graph.add_node("retrieve_node", self._node_retrieve)
        graph.add_node("grade_documents_node", self._node_grade_documents)
        graph.add_node("web_search_node", self._node_web_search)
        graph.add_node("refine_documents_node", self._node_refine_documents)
        graph.add_node("generate_node", self._node_generate)
        graph.add_node("validate_answer_node", self._node_validate_answer)

        # Entry
        graph.add_edge(START, "retrieve_node")
        graph.add_edge("retrieve_node", "grade_documents_node")

        # CRAG decision
        graph.add_conditional_edges(
            "grade_documents_node",
            self._route_after_grade_documents,
            {
                "web_search_node": "web_search_node",
                "generate_node": "generate_node",
            },
        )

        # After web search
        graph.add_conditional_edges(
            "web_search_node",
            self._route_after_web_search,
            {
                "refine_documents_node": "refine_documents_node",
                "generate_node": "generate_node",
            },
        )

        # Refinement → generation
        graph.add_edge("refine_documents_node", "generate_node")

        # Generation → validation → end
        graph.add_edge("generate_node", "validate_answer_node")
        graph.add_edge("validate_answer_node", END)

        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        thread_id: str = "default",
        config: Optional[dict] = None,
    ) -> CorrectiveRAGState:
        """
        Execute the Corrective RAG graph for a query.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier for checkpointing.
            config: Optional additional LangGraph config.

        Returns:
            Final CorrectiveRAGState after graph execution.
        """
        initial_state = make_initial_corrective_rag_state(query)
        graph_config = {"configurable": {"thread_id": thread_id}}
        if config:
            graph_config.update(config)

        logger.info(f"[CorrectiveRAG] Starting run for thread={thread_id}")
        start = time.time()

        final_state = self._graph.invoke(initial_state, config=graph_config)

        elapsed = time.time() - start
        logger.info(
            f"[CorrectiveRAG] Completed in {elapsed:.2f}s | "
            f"web_search={final_state.get('needs_web_search', False)} | "
            f"confidence={final_state.get('overall_confidence', 0):.2f}"
        )
        return final_state

    def stream(
        self,
        query: str,
        thread_id: str = "default",
    ):
        """
        Stream the Corrective RAG graph execution.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier.

        Yields:
            Per-node state update dicts.
        """
        initial_state = make_initial_corrective_rag_state(query)
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self._graph.stream(initial_state, config=config):
            yield chunk

    @property
    def graph(self) -> Any:
        """Access the compiled LangGraph."""
        return self._graph
