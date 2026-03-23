"""
LangGraph state definitions for the Legal Advisor Agent.

This module defines all TypedDict state classes used across the three RAG graphs
and the main orchestrator. Each graph manipulates a subset of these fields.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Sequence
from typing_extensions import TypedDict, Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Classification of the incoming legal query."""

    SIMPLE_FACT = "simple_fact"          # Single law article lookup
    LEGAL_ANALYSIS = "legal_analysis"    # Multi-article / case analysis
    COURT_DECISION = "court_decision"    # Precedent / case law search
    CALCULATION = "calculation"          # Deadline / date calculation
    CURRENT_EVENT = "current_event"      # Recent legal developments
    GENERAL = "general"                  # Catch-all


class RAGStrategy(str, Enum):
    """Which RAG pipeline to use for this query."""

    ADAPTIVE = "adaptive"
    SELF = "self"
    CORRECTIVE = "corrective"
    DIRECT = "direct"          # Answer without retrieval (very simple facts)


class DocumentGrade(str, Enum):
    """Outcome of the relevance grader for a single document."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"


class GenerationGrade(str, Enum):
    """Outcome of the hallucination / answer-quality grader."""

    SUPPORTED = "supported"           # No hallucination
    NOT_SUPPORTED = "not_supported"   # Hallucination detected
    USEFUL = "useful"                 # Answers the question
    NOT_USEFUL = "not_useful"         # Does not answer the question


class HumanDecision(str, Enum):
    """Human reviewer's decision at a HITL checkpoint."""

    APPROVED = "approved"
    REJECTED = "rejected"
    REQUEST_WEB_SEARCH = "request_web_search"
    REQUEST_REWRITE = "request_rewrite"


# ---------------------------------------------------------------------------
# Evidence / citation tracking
# ---------------------------------------------------------------------------


class LegalEvidence(TypedDict, total=False):
    """A single piece of evidence supporting an answer."""

    doc_id: str
    law_name: str                # e.g. "민법", "근로기준법"
    article_number: str          # e.g. "제750조"
    article_title: str           # e.g. "불법행위의 내용"
    content_snippet: str         # Relevant excerpt (≤ 300 chars)
    source_type: str             # "vector_store" | "web_search" | "direct"
    relevance_score: float       # 0.0 – 1.0
    url: Optional[str]           # For web-sourced evidence


# ---------------------------------------------------------------------------
# Core AgentState — shared by all graphs
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Master state object threaded through every LangGraph node.

    Fields with ``Annotated[..., add_messages]`` accumulate messages;
    all other fields are overwritten on each update.
    """

    # ---- Conversation ----
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Full conversation history including system, human, and AI messages."""

    thread_id: str
    """Unique identifier for this conversation thread (enables persistence)."""

    turn_number: int
    """Monotonically increasing turn counter within a thread."""

    # ---- Query ----
    query: str
    """The current user query (may be rewritten during RAG iterations)."""

    original_query: str
    """The original, unmodified user query."""

    query_type: QueryType
    """Classified type of the query."""

    rag_strategy: RAGStrategy
    """Chosen RAG strategy for this query."""

    query_complexity: str          # "simple" | "moderate" | "complex"
    language: str                  # "ko" | "en" | "mixed"

    # ---- Retrieval ----
    retrieved_documents: list[Document]
    """Raw documents returned by the retriever."""

    filtered_documents: list[Document]
    """Documents that passed relevance grading."""

    web_search_results: list[dict[str, Any]]
    """Raw results from Tavily web search."""

    evidence: list[LegalEvidence]
    """Curated evidence list used for answer generation."""

    # ---- Grading ----
    document_grades: list[DocumentGrade]
    """Per-document relevance grades (parallel to retrieved_documents)."""

    hallucination_grade: Optional[GenerationGrade]
    """Whether the latest generated answer contains hallucinations."""

    answer_grade: Optional[GenerationGrade]
    """Whether the latest generated answer addresses the query."""

    relevance_scores: list[float]
    """Numeric relevance scores (0.0 – 1.0) per document."""

    overall_confidence: float
    """Aggregate confidence score for the current answer (0.0 – 1.0)."""

    # ---- Generation ----
    generated_answer: str
    """Latest LLM-generated answer."""

    final_answer: str
    """The approved, disclaimer-appended answer delivered to the user."""

    citations: list[str]
    """Human-readable citation strings included in the answer."""

    # ---- Iteration control (Self-RAG) ----
    iteration_count: int
    """Number of generation / reflection cycles completed."""

    should_retrieve: bool
    """Self-RAG token: should additional retrieval happen?"""

    should_rewrite: bool
    """Self-RAG token: should the query be rewritten?"""

    # ---- Corrective RAG ----
    needs_web_search: bool
    """True when retrieved docs are insufficient and web fallback is needed."""

    refined_documents: list[Document]
    """Documents after web search augmentation and refinement."""

    # ---- Human-in-the-Loop ----
    requires_human_review: bool
    """True when the answer must be reviewed before delivery."""

    human_decision: Optional[HumanDecision]
    """Decision made by the human reviewer."""

    human_feedback: str
    """Textual feedback from the human reviewer."""

    awaiting_human: bool
    """True when the graph is paused at a HITL interrupt node."""

    # ---- Metadata ----
    error: Optional[str]
    """Any error message captured during graph execution."""

    start_time: Optional[datetime]
    """Timestamp when the current query processing began."""

    node_trace: list[str]
    """Ordered list of nodes visited during graph execution (for debugging)."""


# ---------------------------------------------------------------------------
# Graph-specific lightweight states
# ---------------------------------------------------------------------------


class AdaptiveRAGState(TypedDict, total=False):
    """State used exclusively within the Adaptive RAG graph."""

    query: str
    original_query: str
    query_type: QueryType
    retrieved_documents: list[Document]
    web_search_results: list[dict[str, Any]]
    generated_answer: str
    hallucination_grade: Optional[GenerationGrade]
    answer_grade: Optional[GenerationGrade]
    overall_confidence: float
    citations: list[str]
    evidence: list[LegalEvidence]
    iteration_count: int
    error: Optional[str]
    node_trace: list[str]


class SelfRAGState(TypedDict, total=False):
    """State used exclusively within the Self-RAG graph."""

    query: str
    original_query: str
    retrieved_documents: list[Document]
    filtered_documents: list[Document]
    document_grades: list[DocumentGrade]
    relevance_scores: list[float]
    generated_answer: str
    hallucination_grade: Optional[GenerationGrade]
    answer_grade: Optional[GenerationGrade]
    overall_confidence: float
    should_retrieve: bool
    should_rewrite: bool
    iteration_count: int
    citations: list[str]
    evidence: list[LegalEvidence]
    error: Optional[str]
    node_trace: list[str]


class CorrectiveRAGState(TypedDict, total=False):
    """State used exclusively within the Corrective RAG graph."""

    query: str
    original_query: str
    retrieved_documents: list[Document]
    filtered_documents: list[Document]
    document_grades: list[DocumentGrade]
    relevance_scores: list[float]
    needs_web_search: bool
    web_search_results: list[dict[str, Any]]
    refined_documents: list[Document]
    generated_answer: str
    hallucination_grade: Optional[GenerationGrade]
    answer_grade: Optional[GenerationGrade]
    overall_confidence: float
    citations: list[str]
    evidence: list[LegalEvidence]
    error: Optional[str]
    node_trace: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_initial_agent_state(
    query: str,
    thread_id: str,
    rag_strategy: RAGStrategy = RAGStrategy.ADAPTIVE,
    require_human_approval: bool = False,
) -> AgentState:
    """
    Build the initial AgentState for a new query.

    Args:
        query: The user's legal question.
        thread_id: Unique conversation thread identifier.
        rag_strategy: Pre-selected RAG strategy (may be overridden by classifier).
        require_human_approval: Whether to trigger HITL before final delivery.

    Returns:
        A fully initialised AgentState dict.
    """
    return AgentState(
        messages=[],
        thread_id=thread_id,
        turn_number=1,
        query=query,
        original_query=query,
        query_type=QueryType.GENERAL,
        rag_strategy=rag_strategy,
        query_complexity="moderate",
        language="ko",
        retrieved_documents=[],
        filtered_documents=[],
        web_search_results=[],
        evidence=[],
        document_grades=[],
        hallucination_grade=None,
        answer_grade=None,
        relevance_scores=[],
        overall_confidence=0.0,
        generated_answer="",
        final_answer="",
        citations=[],
        iteration_count=0,
        should_retrieve=True,
        should_rewrite=False,
        needs_web_search=False,
        refined_documents=[],
        requires_human_review=require_human_approval,
        human_decision=None,
        human_feedback="",
        awaiting_human=False,
        error=None,
        start_time=datetime.utcnow(),
        node_trace=[],
    )


def make_initial_adaptive_state(query: str) -> AdaptiveRAGState:
    """Build the initial state for the Adaptive RAG graph."""
    return AdaptiveRAGState(
        query=query,
        original_query=query,
        query_type=QueryType.GENERAL,
        retrieved_documents=[],
        web_search_results=[],
        generated_answer="",
        hallucination_grade=None,
        answer_grade=None,
        overall_confidence=0.0,
        citations=[],
        evidence=[],
        iteration_count=0,
        error=None,
        node_trace=[],
    )


def make_initial_self_rag_state(query: str) -> SelfRAGState:
    """Build the initial state for the Self-RAG graph."""
    return SelfRAGState(
        query=query,
        original_query=query,
        retrieved_documents=[],
        filtered_documents=[],
        document_grades=[],
        relevance_scores=[],
        generated_answer="",
        hallucination_grade=None,
        answer_grade=None,
        overall_confidence=0.0,
        should_retrieve=True,
        should_rewrite=False,
        iteration_count=0,
        citations=[],
        evidence=[],
        error=None,
        node_trace=[],
    )


def make_initial_corrective_rag_state(query: str) -> CorrectiveRAGState:
    """Build the initial state for the Corrective RAG graph."""
    return CorrectiveRAGState(
        query=query,
        original_query=query,
        retrieved_documents=[],
        filtered_documents=[],
        document_grades=[],
        relevance_scores=[],
        needs_web_search=False,
        web_search_results=[],
        refined_documents=[],
        generated_answer="",
        hallucination_grade=None,
        answer_grade=None,
        overall_confidence=0.0,
        citations=[],
        evidence=[],
        error=None,
        node_trace=[],
    )
