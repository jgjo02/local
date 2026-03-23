"""
Adaptive RAG LangGraph for the Legal Advisor Agent.

Adaptive RAG dynamically routes each query to the most appropriate retrieval
strategy based on query classification:
  - DIRECT: Simple factual queries answered without retrieval
  - VECTOR: Semantic search over indexed legal documents
  - WEB: Tavily web search for current events or missing local knowledge

The graph uses LangGraph's StateGraph with conditional edges to implement
the routing logic.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.agent_state import (
    AdaptiveRAGState,
    LegalEvidence,
    QueryType,
    make_initial_adaptive_state,
)
from src.config import get_config
from src.graders import GraderSuite, QueryClassification
from src.legal_vector_store import LegalVectorStore


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
사용자의 법률 질문에 명확하고 정확하게 답변하세요.
계산이나 단순 사실 질문은 직접 답변하되, 관련 법령 조항을 인용하세요.
답변은 한국어로 작성하세요.""",
        ),
        ("human", "{query}"),
    ]
)

VECTOR_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
아래 검색된 법령 문서를 참조하여 사용자의 질문에 답변하세요.

[검색된 법령 문서]
{context}

답변 시 준수사항:
1. 제공된 문서에 근거하여 답변하세요
2. 관련 법령 조항(예: 민법 제750조)을 명시적으로 인용하세요
3. 법적 불확실성이 있는 경우 명시하세요
4. 답변은 한국어로 작성하세요
5. 복잡한 내용은 번호 목록으로 정리하세요""",
        ),
        ("human", "질문: {query}"),
    ]
)

WEB_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
아래 웹 검색 결과와 법령 문서를 참조하여 사용자의 질문에 답변하세요.

[웹 검색 결과]
{web_context}

[법령 문서 (있는 경우)]
{doc_context}

답변 시 준수사항:
1. 출처를 명시하세요 (법령명, URL 등)
2. 최신 정보와 공식 법령을 함께 제시하세요
3. 정보의 신뢰도를 언급하세요
4. 답변은 한국어로 작성하세요""",
        ),
        ("human", "질문: {query}"),
    ]
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 법률 검색 전문가입니다.
사용자의 법률 질문을 벡터 데이터베이스 검색에 최적화된 형태로 재작성하세요.
법령명, 조문 번호, 핵심 법률 용어를 명시적으로 포함하세요.
재작성된 질문만 출력하세요.""",
        ),
        ("human", "원래 질문: {query}"),
    ]
)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def _build_context_from_docs(documents: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    if not documents:
        return "관련 문서를 찾을 수 없습니다."
    parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        header = f"[{i}] {meta.get('law_name', '미상')} {meta.get('article_label', '')} {meta.get('article_title', '')}".strip()
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_web_context(web_results: list[dict[str, Any]]) -> str:
    """Format web search results into a context string."""
    if not web_results:
        return "웹 검색 결과 없음"
    parts = []
    for i, r in enumerate(web_results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")[:400]
        parts.append(f"[웹 {i}] {title}\n출처: {url}\n{content}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(documents: list[Document], web_results: list[dict]) -> list[str]:
    """Extract citation strings from documents and web results."""
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
    for r in web_results:
        url = r.get("url", "")
        title = r.get("title", "")
        if url and url not in seen:
            citations.append(f"{title} ({url})")
            seen.add(url)
    return citations


def _docs_to_evidence(documents: list[Document]) -> list[LegalEvidence]:
    """Convert Documents to LegalEvidence records."""
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
                source_type=meta.get("source_type", "vector_store"),
                relevance_score=meta.get("relevance_score", 0.0),
                url=meta.get("url"),
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# AdaptiveRAGGraph class
# ---------------------------------------------------------------------------


class AdaptiveRAGGraph:
    """
    Adaptive RAG LangGraph implementation.

    Builds a stateful LangGraph that:
    1. Classifies the query (query_classifier node)
    2. Routes to: direct answer | vector search | web search
    3. Generates an answer
    4. Validates the answer (hallucination + quality graders)
    5. Optionally retries with rewritten query
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

    def _node_query_classifier(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Classify the query and decide the retrieval strategy."""
        node_name = "query_classifier"
        logger.info(f"[AdaptiveRAG] {node_name}: '{state['query'][:60]}…'")

        classification: QueryClassification = self._graders.query.classify(state["query"])

        return {
            **state,
            "query_type": classification.query_type,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_vector_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Retrieve documents from the vector store."""
        node_name = "vector_search"
        logger.info(f"[AdaptiveRAG] {node_name}")

        query = state["query"]
        docs: list[Document] = []

        try:
            docs = self._vector_store.hybrid_search(
                query=query,
                k=self._config.retriever_k,
            )
            logger.debug(f"Retrieved {len(docs)} documents")
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")

        return {
            **state,
            "retrieved_documents": docs,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_web_search(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Perform a Tavily web search."""
        node_name = "web_search"
        logger.info(f"[AdaptiveRAG] {node_name}")

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
            )
            web_results = response.get("results", [])
            if response.get("answer"):
                web_results.insert(
                    0, {"title": "요약", "url": "", "content": response["answer"]}
                )
        except Exception as exc:
            logger.warning(f"Web search failed: {exc}")

        return {
            **state,
            "web_search_results": web_results,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_direct_answer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate an answer directly without retrieval."""
        node_name = "direct_answer"
        logger.info(f"[AdaptiveRAG] {node_name}")

        chain = DIRECT_ANSWER_PROMPT | self._llm | StrOutputParser()
        answer = chain.invoke({"query": state["query"]})

        return {
            **state,
            "generated_answer": answer,
            "overall_confidence": 0.90,  # High confidence for direct answers
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_generate_answer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Generate an answer from retrieved documents and/or web results."""
        node_name = "generate_answer"
        logger.info(f"[AdaptiveRAG] {node_name}")

        documents = state.get("retrieved_documents", [])
        web_results = state.get("web_search_results", [])

        if web_results:
            chain = WEB_ANSWER_PROMPT | self._llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "query": state["query"],
                    "web_context": _build_web_context(web_results),
                    "doc_context": _build_context_from_docs(documents),
                }
            )
        else:
            chain = VECTOR_ANSWER_PROMPT | self._llm | StrOutputParser()
            answer = chain.invoke(
                {
                    "query": state["query"],
                    "context": _build_context_from_docs(documents),
                }
            )

        citations = _extract_citations(documents, web_results)
        evidence = _docs_to_evidence(documents)

        return {
            **state,
            "generated_answer": answer,
            "citations": citations,
            "evidence": evidence,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_grade_answer(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Grade the generated answer for hallucination and quality."""
        node_name = "grade_answer"
        logger.info(f"[AdaptiveRAG] {node_name}")

        answer = state.get("generated_answer", "")
        documents = state.get("retrieved_documents", [])
        query = state["query"]

        hallucination_result = self._graders.hallucination.grade(
            answer=answer, documents=documents
        )
        answer_result = self._graders.answer.grade(answer=answer, query=query)

        # Compute relevance scores from document metadata
        relevance_scores = [
            float(doc.metadata.get("relevance_score", 0.5)) for doc in documents
        ]

        confidence = self._graders.compute_overall_confidence(
            relevance_scores=relevance_scores,
            hallucination_result=hallucination_result,
            answer_result=answer_result,
        )

        return {
            **state,
            "hallucination_grade": hallucination_result.score,
            "answer_grade": answer_result.score,
            "relevance_scores": relevance_scores,
            "overall_confidence": confidence,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    def _node_transform_query(self, state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Rewrite the query for better retrieval."""
        node_name = "transform_query"
        logger.info(f"[AdaptiveRAG] {node_name}")

        chain = QUERY_REWRITE_PROMPT | self._llm | StrOutputParser()
        new_query = chain.invoke({"query": state["query"]})

        return {
            **state,
            "query": new_query,
            "node_trace": state.get("node_trace", []) + [node_name],
        }

    # ------------------------------------------------------------------
    # Routing functions (conditional edges)
    # ------------------------------------------------------------------

    def _route_after_classifier(
        self, state: AdaptiveRAGState
    ) -> Literal["direct_answer", "vector_search", "web_search"]:
        """Route the query based on its classification."""
        query_type = state.get("query_type", "general")
        logger.debug(f"[AdaptiveRAG] Routing: query_type={query_type}")

        if query_type in ("calculation", "simple_fact"):
            # Try vector first, fall back to direct
            return "vector_search"
        elif query_type in ("current_event", "court_decision"):
            return "web_search"
        else:
            return "vector_search"

    def _route_after_grading(
        self, state: AdaptiveRAGState
    ) -> Literal["transform_query", "end"]:
        """Route after grading: retry if quality is poor, else end."""
        hallucination = state.get("hallucination_grade")
        answer_qual = state.get("answer_grade")
        iterations = state.get("iteration_count", 0)
        confidence = state.get("overall_confidence", 0.0)

        max_iter = self._config.max_iterations

        # Accept answer if quality is good or we've hit max iterations
        if iterations >= max_iter:
            logger.info(f"[AdaptiveRAG] Max iterations ({max_iter}) reached")
            return "end"
        if hallucination == "yes" and answer_qual == "yes" and confidence >= 0.6:
            return "end"
        if confidence < 0.4 and iterations < max_iter:
            logger.info(f"[AdaptiveRAG] Low confidence ({confidence:.2f}), retrying")
            return "transform_query"
        return "end"

    def _route_after_transform(
        self, state: AdaptiveRAGState
    ) -> Literal["vector_search", "web_search"]:
        """After query rewrite, decide retrieval strategy."""
        query_type = state.get("query_type", "general")
        if query_type in ("current_event", "court_decision"):
            return "web_search"
        return "vector_search"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """
        Assemble the Adaptive RAG StateGraph.

        Graph topology:
        START → query_classifier
              → [direct_answer | vector_search | web_search]
              → generate_answer
              → grade_answer
              → [transform_query → (vector_search|web_search) → generate_answer → grade_answer]
              → END
        """
        graph = StateGraph(AdaptiveRAGState)

        # Add nodes
        graph.add_node("query_classifier", self._node_query_classifier)
        graph.add_node("vector_search", self._node_vector_search)
        graph.add_node("web_search", self._node_web_search)
        graph.add_node("direct_answer", self._node_direct_answer)
        graph.add_node("generate_answer", self._node_generate_answer)
        graph.add_node("grade_answer", self._node_grade_answer)
        graph.add_node("transform_query", self._node_transform_query)

        # Entry point
        graph.add_edge(START, "query_classifier")

        # After classifier: route to appropriate retrieval
        graph.add_conditional_edges(
            "query_classifier",
            self._route_after_classifier,
            {
                "direct_answer": "direct_answer",
                "vector_search": "vector_search",
                "web_search": "web_search",
            },
        )

        # Retrieval → generation
        graph.add_edge("vector_search", "generate_answer")
        graph.add_edge("web_search", "generate_answer")

        # Direct answer → grade (skip retrieval but still grade)
        graph.add_edge("direct_answer", "grade_answer")

        # Generation → grading
        graph.add_edge("generate_answer", "grade_answer")

        # After grading: accept or retry
        graph.add_conditional_edges(
            "grade_answer",
            self._route_after_grading,
            {"transform_query": "transform_query", "end": END},
        )

        # After query transform: re-retrieve
        graph.add_conditional_edges(
            "transform_query",
            self._route_after_transform,
            {"vector_search": "vector_search", "web_search": "web_search"},
        )

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
    ) -> AdaptiveRAGState:
        """
        Execute the Adaptive RAG graph for a query.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier for checkpointing.
            config: Optional additional LangGraph config.

        Returns:
            Final AdaptiveRAGState after graph execution.
        """
        initial_state = make_initial_adaptive_state(query)
        graph_config = {"configurable": {"thread_id": thread_id}}
        if config:
            graph_config.update(config)

        logger.info(f"[AdaptiveRAG] Starting run for thread={thread_id}")
        start = time.time()

        final_state = self._graph.invoke(initial_state, config=graph_config)

        elapsed = time.time() - start
        logger.info(
            f"[AdaptiveRAG] Completed in {elapsed:.2f}s | "
            f"confidence={final_state.get('overall_confidence', 0):.2f} | "
            f"nodes={final_state.get('node_trace', [])}"
        )
        return final_state

    def stream(
        self,
        query: str,
        thread_id: str = "default",
    ):
        """
        Stream the Adaptive RAG graph execution.

        Yields state updates from each node as they complete.

        Args:
            query: The user's legal question.
            thread_id: Unique thread identifier.

        Yields:
            Tuples of (node_name, state_update).
        """
        initial_state = make_initial_adaptive_state(query)
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self._graph.stream(initial_state, config=config):
            yield chunk

    @property
    def graph(self) -> Any:
        """Access the compiled LangGraph for inspection or visualization."""
        return self._graph
