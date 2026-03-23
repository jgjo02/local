"""
Self-RAG LangGraph for the Legal Advisor Agent.

Self-RAG implements the SELF-RAG framework (Asai et al., 2023) where the model
uses "reflection tokens" to determine:
  - ISREL: Is this retrieved document relevant?
  - ISSUP: Is the generated answer supported by the documents?
  - ISUSE: Is the generated answer useful to the user?

The graph implements iterative self-reflection with query rewriting until
the answer meets quality thresholds or max iterations are reached.
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
    DocumentGrade,
    GenerationGrade,
    LegalEvidence,
    SelfRAGState,
    make_initial_self_rag_state,
)
from src.config import get_config
from src.graders import GraderSuite
from src.legal_vector_store import LegalVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SELF_RAG_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
다음 검색된 법령 문서를 참조하여 사용자의 질문에 답변하세요.
답변은 반드시 제공된 문서에 근거해야 하며, 근거 없는 추측은 하지 마세요.

[검색된 법령 문서]
{context}

답변 지침:
1. 관련 법령과 조문을 명시적으로 인용하세요 (예: 민법 제750조)
2. 단계별로 논리적으로 설명하세요
3. 실제 적용 사례나 예시를 포함하세요 (문서에 있는 경우)
4. 예외 조항이나 조건이 있으면 반드시 언급하세요
5. 답변은 한국어로 작성하세요""",
        ),
        ("human", "질문: {query}"),
    ]
)

QUERY_REWRITE_FOR_SELF_RAG = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 법률 검색 최적화 전문가입니다.
현재 답변이 불충분하거나 환각이 감지되었습니다.
더 나은 법령 문서를 검색하기 위해 질문을 개선하세요.

개선 방법:
- 더 구체적인 법률 용어 사용
- 법령명이나 조문 번호 추가
- 검색 범위를 좁히거나 관련 개념 추가
- 모호한 표현을 명확히

개선된 질문만 출력하세요.""",
        ),
        (
            "human",
            """원래 질문: {original_query}
현재 검색 쿼리: {current_query}
현재 답변의 문제점: {issue}

개선된 검색 쿼리:""",
        ),
    ]
)

RETRIEVAL_NECESSITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 법률 질문 분석가입니다.
주어진 질문이 법령 데이터베이스 검색이 필요한지 판단하세요.

검색 불필요: 일반 인사, 법률과 무관한 질문, 매우 간단한 상식 질문
검색 필요: 특정 법령 조항, 법적 절차, 권리/의무, 계약/소송 관련

"yes" 또는 "no"만 응답하세요.""",
        ),
        ("human", "질문: {query}\n\n검색이 필요합니까?"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_context(documents: list[Document]) -> str:
    """Format filtered documents as a context string."""
    if not documents:
        return "관련 법령 문서를 찾지 못했습니다."
    parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        law = meta.get("law_name", "미상")
        article = meta.get("article_label", "")
        title = meta.get("article_title", "")
        header = f"[문서 {i}] {law} {article} {title}".strip()
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_citations_from_docs(documents: list[Document]) -> list[str]:
    """Extract citation strings."""
    citations: list[str] = []
    seen: set[str] = set()
    for doc in documents:
        meta = doc.metadata
        law = meta.get("law_name", "")
        article = meta.get("article_label", "")
        if law and article:
            cite = f"{law} {article}"
            if cite not in seen:
                citations.append(cite)
                seen.add(cite)
    return citations


def _docs_to_evidence(documents: list[Document]) -> list[LegalEvidence]:
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
    return evidence


# ---------------------------------------------------------------------------
# SelfRAGGraph class
# ---------------------------------------------------------------------------


class SelfRAGGraph:
    """
    Self-RAG LangGraph implementation.

    Graph topology:
    START → retrieve_node
          → grade_documents_node
          → [generate_node | transform_query_node]
    generate_node → grade_generation_node
          → [END | transform_query_node]
    transform_query_node → retrieve_node (loop)

    Reflection tokens implemented as grader outputs:
    - ISREL (document relevance): RelevanceGrader
    - ISSUP (answer support): HallucinationGrader
    - ISUSE (answer usefulness): AnswerGrader
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

    def _node_retrieve(self, state: SelfRAGState) -> SelfRAGState:
        """
        Retrieve node: Check if retrieval is needed, then retrieve documents.

        Implements the ISRET (retrieve token) decision.
        """
        node_name = "retrieve_node"
        logger.info(f"[SelfRAG] {node_name}: query='{state['query'][:60]}…'")

        # Check retrieval necessity
        necessity_chain = RETRIEVAL_NECESSITY_PROMPT | self._llm | StrOutputParser()
        try:
            necessity = necessity_chain.invoke({"query": state["query"]})
            should_retrieve = "yes" in necessity.lower()
        except Exception:
            should_retrieve = True

        if not should_retrieve:
            logger.debug("[SelfRAG] Retrieval deemed unnecessary")
            return {
                **state,
                "should_retrieve": False,
                "retrieved_documents": [],
                "node_trace": state.get("node_trace", []) + [node_name],
            }

        # Perform retrieval
        docs: list[Document] = []
        try:
            docs = self._vector_store.hybrid_search(
                query=state["query"],
                k=self._config.retriever_k,
            )
            logger.debug(f"[SelfRAG] Retrieved {len(docs)} documents")
        except Exception as exc:
            logger.error(f"[SelfRAG] Retrieval failed: {exc}")

        return {
            **state,
            "should_retrieve": True,
            "retrieved_documents": docs,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_grade_documents(self, state: SelfRAGState) -> SelfRAGState:
        """
        Grade documents node: Apply ISREL token to filter retrieved documents.

        Scores each document for relevance and keeps only relevant ones.
        """
        node_name = "grade_documents_node"
        logger.info(f"[SelfRAG] {node_name}")

        documents = state.get("retrieved_documents", [])
        query = state["query"]

        if not documents:
            return {
                **state,
                "filtered_documents": [],
                "document_grades": [],
                "relevance_scores": [],
                "should_rewrite": True,
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
                logger.debug(
                    f"[SelfRAG] Filtered out doc: {doc.metadata.get('article_label', '')}"
                )

        logger.info(
            f"[SelfRAG] Kept {len(filtered)}/{len(documents)} relevant documents"
        )

        # If too few relevant docs, rewrite query
        should_rewrite = len(filtered) < max(1, len(documents) // 3)

        return {
            **state,
            "filtered_documents": filtered,
            "document_grades": grades,
            "relevance_scores": scores,
            "should_rewrite": should_rewrite,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_generate(self, state: SelfRAGState) -> SelfRAGState:
        """
        Generate node: Produce an answer from filtered documents.

        If no filtered documents are available, generates from all retrieved docs.
        """
        node_name = "generate_node"
        logger.info(f"[SelfRAG] {node_name}")

        # Use filtered docs if available, otherwise fall back to all retrieved
        docs = state.get("filtered_documents") or state.get("retrieved_documents", [])
        context = _format_context(docs)

        chain = SELF_RAG_GENERATE_PROMPT | self._llm | StrOutputParser()
        try:
            answer = chain.invoke({"query": state["query"], "context": context})
        except Exception as exc:
            logger.error(f"[SelfRAG] Generation failed: {exc}")
            answer = f"답변 생성 중 오류가 발생했습니다: {exc}"

        citations = _extract_citations_from_docs(docs)
        evidence = _docs_to_evidence(docs)

        return {
            **state,
            "generated_answer": answer,
            "citations": citations,
            "evidence": evidence,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_grade_generation(self, state: SelfRAGState) -> SelfRAGState:
        """
        Grade generation node: Apply ISSUP and ISUSE reflection tokens.

        ISSUP = HallucinationGrader (is the answer supported by docs?)
        ISUSE = AnswerGrader (is the answer useful?)
        """
        node_name = "grade_generation_node"
        logger.info(f"[SelfRAG] {node_name}")

        answer = state.get("generated_answer", "")
        docs = state.get("filtered_documents") or state.get("retrieved_documents", [])
        query = state["query"]

        # ISSUP: hallucination check
        hallucination_result = self._graders.hallucination.grade(
            answer=answer, documents=docs
        )

        # ISUSE: answer quality check
        answer_result = self._graders.answer.grade(answer=answer, query=query)

        relevance_scores = state.get("relevance_scores", [])
        confidence = self._graders.compute_overall_confidence(
            relevance_scores=relevance_scores,
            hallucination_result=hallucination_result,
            answer_result=answer_result,
        )

        logger.info(
            f"[SelfRAG] ISSUP={hallucination_result.score} "
            f"ISUSE={answer_result.score} "
            f"confidence={confidence:.2f}"
        )

        return {
            **state,
            "hallucination_grade": hallucination_result.score,
            "answer_grade": answer_result.score,
            "overall_confidence": confidence,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_transform_query(self, state: SelfRAGState) -> SelfRAGState:
        """
        Transform query node: Rewrite the query to improve future retrieval.

        Triggered when documents are irrelevant or the answer is unsatisfactory.
        """
        node_name = "transform_query_node"
        logger.info(f"[SelfRAG] {node_name}")

        issue = "관련 문서 부족" if state.get("should_rewrite") else "답변 품질 미흡"
        if state.get("hallucination_grade") == "no":
            issue = "환각 감지됨"

        chain = QUERY_REWRITE_FOR_SELF_RAG | self._llm | StrOutputParser()
        try:
            new_query = chain.invoke(
                {
                    "original_query": state.get("original_query", state["query"]),
                    "current_query": state["query"],
                    "issue": issue,
                }
            )
            new_query = new_query.strip()
            logger.debug(f"[SelfRAG] Rewritten query: '{new_query[:80]}…'")
        except Exception as exc:
            logger.warning(f"[SelfRAG] Query transform failed: {exc}")
            new_query = state["query"]

        return {
            **state,
            "query": new_query,
            "should_rewrite": False,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def _route_after_grade_documents(
        self, state: SelfRAGState
    ) -> Literal["generate_node", "transform_query_node"]:
        """Route after document grading."""
        iterations = state.get("iteration_count", 0)
        should_rewrite = state.get("should_rewrite", False)
        filtered = state.get("filtered_documents", [])

        if iterations >= self._config.max_iterations:
            logger.info("[SelfRAG] Max iterations hit, generating anyway")
            return "generate_node"

        if not filtered and should_rewrite:
            logger.info("[SelfRAG] No relevant documents, rewriting query")
            return "transform_query_node"

        return "generate_node"

    def _route_after_grade_generation(
        self, state: SelfRAGState
    ) -> Literal["transform_query_node", "end"]:
        """Route after generation grading: accept or iterate."""
        hallucination = state.get("hallucination_grade")
        answer_qual = state.get("answer_grade")
        confidence = state.get("overall_confidence", 0.0)
        iterations = state.get("iteration_count", 0)

        max_iter = self._config.max_iterations

        if iterations >= max_iter:
            logger.info(f"[SelfRAG] Max iterations ({max_iter}) reached, accepting answer")
            return "end"

        # Accept: no hallucination AND useful answer
        if hallucination == "yes" and answer_qual == "yes":
            logger.info(f"[SelfRAG] Answer accepted (confidence={confidence:.2f})")
            return "end"

        # Hallucination detected → rewrite and retry
        if hallucination == "no":
            logger.info("[SelfRAG] Hallucination detected, rewriting query")
            return "transform_query_node"

        # Low-quality answer → rewrite
        if answer_qual == "no":
            logger.info("[SelfRAG] Answer quality insufficient, rewriting query")
            return "transform_query_node"

        return "end"

    def _route_after_retrieve(
        self, state: SelfRAGState
    ) -> Literal["grade_documents_node", "generate_node"]:
        """If retrieval was unnecessary, skip directly to generation."""
        if not state.get("should_retrieve", True):
            return "generate_node"
        return "grade_documents_node"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """
        Assemble the Self-RAG StateGraph.

        Graph topology:
        START → retrieve_node
              → [grade_documents_node | generate_node]  (skip grading if no retrieval)
        grade_documents_node → [generate_node | transform_query_node]
        generate_node → grade_generation_node
        grade_generation_node → [transform_query_node | END]
        transform_query_node → retrieve_node  (retry loop)
        """
        graph = StateGraph(SelfRAGState)

        # Register nodes
        graph.add_node("retrieve_node", self._node_retrieve)
        graph.add_node("grade_documents_node", self._node_grade_documents)
        graph.add_node("generate_node", self._node_generate)
        graph.add_node("grade_generation_node", self._node_grade_generation)
        graph.add_node("transform_query_node", self._node_transform_query)

        # Entry
        graph.add_edge(START, "retrieve_node")

        # After retrieval: grade or skip
        graph.add_conditional_edges(
            "retrieve_node",
            self._route_after_retrieve,
            {
                "grade_documents_node": "grade_documents_node",
                "generate_node": "generate_node",
            },
        )

        # After grading docs: generate or rewrite
        graph.add_conditional_edges(
            "grade_documents_node",
            self._route_after_grade_documents,
            {
                "generate_node": "generate_node",
                "transform_query_node": "transform_query_node",
            },
        )

        # Generate → grade generation
        graph.add_edge("generate_node", "grade_generation_node")

        # After grading: accept or retry
        graph.add_conditional_edges(
            "grade_generation_node",
            self._route_after_grade_generation,
            {
                "transform_query_node": "transform_query_node",
                "end": END,
            },
        )

        # Transform query → re-retrieve (closes the loop)
        graph.add_edge("transform_query_node", "retrieve_node")

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
    ) -> SelfRAGState:
        """
        Execute the Self-RAG graph for a query.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier for checkpointing.
            config: Optional additional LangGraph config.

        Returns:
            Final SelfRAGState after graph execution.
        """
        initial_state = make_initial_self_rag_state(query)
        graph_config = {"configurable": {"thread_id": thread_id}}
        if config:
            graph_config.update(config)

        logger.info(f"[SelfRAG] Starting run for thread={thread_id}")
        start = time.time()

        final_state = self._graph.invoke(initial_state, config=graph_config)

        elapsed = time.time() - start
        logger.info(
            f"[SelfRAG] Completed in {elapsed:.2f}s | "
            f"iterations={final_state.get('iteration_count', 0)} | "
            f"confidence={final_state.get('overall_confidence', 0):.2f}"
        )
        return final_state

    def stream(
        self,
        query: str,
        thread_id: str = "default",
    ):
        """
        Stream the Self-RAG graph execution, yielding per-node updates.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier.

        Yields:
            Per-node state update dicts.
        """
        initial_state = make_initial_self_rag_state(query)
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self._graph.stream(initial_state, config=config):
            yield chunk

    @property
    def graph(self) -> Any:
        """Access the compiled LangGraph."""
        return self._graph
