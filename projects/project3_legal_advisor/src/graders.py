"""
LLM-based RAG quality graders for the Legal Advisor Agent.

Each grader is a Pydantic model paired with an LCEL chain that calls an LLM
to evaluate a specific quality dimension. Graders are used by RAG graph nodes
to decide whether to re-retrieve, re-generate, or accept an answer.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_config


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------


class RelevanceScore(BaseModel):
    """Output of the RelevanceGrader."""

    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the query, 'no' otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of why the document is or is not relevant"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this relevance assessment (0.0 – 1.0)",
    )


class HallucinationScore(BaseModel):
    """Output of the HallucinationGrader."""

    score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer is grounded in the provided documents (no hallucination), "
            "'no' if it contains unsupported claims"
        )
    )
    reasoning: str = Field(
        description="Specific unsupported claims found, or confirmation that all claims are supported"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the hallucination assessment (0.0 – 1.0)",
    )


class AnswerQualityScore(BaseModel):
    """Output of the AnswerGrader."""

    score: Literal["yes", "no"] = Field(
        description="'yes' if the answer sufficiently addresses the user's question"
    )
    reasoning: str = Field(
        description="Explanation of whether and how well the answer addresses the question"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the quality assessment (0.0 – 1.0)",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="List of aspects of the question that were not adequately addressed",
    )


class QueryClassification(BaseModel):
    """Output of the QueryGrader."""

    query_type: Literal[
        "simple_fact",
        "legal_analysis",
        "court_decision",
        "calculation",
        "current_event",
        "general",
    ] = Field(description="The classified type of the legal query")
    complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Estimated complexity of answering this query"
    )
    recommended_rag: Literal["adaptive", "self", "corrective", "direct"] = Field(
        description="Recommended RAG strategy for this query"
    )
    requires_web_search: bool = Field(
        description="True if web search is likely needed to answer this query"
    )
    reasoning: str = Field(
        description="Brief reasoning for the classification decisions"
    )
    extracted_law_name: Optional[str] = Field(
        default=None,
        description="If a specific law name is mentioned, extract it (e.g. '민법', '근로기준법')",
    )
    extracted_article: Optional[str] = Field(
        default=None,
        description="If a specific article number is mentioned, extract it (e.g. '750')",
    )


# ---------------------------------------------------------------------------
# Grader classes
# ---------------------------------------------------------------------------


class RelevanceGrader:
    """
    Grades whether a retrieved document is relevant to a user query.

    Used in Self-RAG and Corrective RAG to filter retrieved documents.

    Example::

        grader = RelevanceGrader()
        result = grader.grade(document=doc, query="임대차 해지 통보 기간")
        if result.score == "yes":
            use_document(doc)
    """

    SYSTEM_PROMPT = """당신은 법률 문서의 관련성을 평가하는 전문 평가자입니다.
법률 질문에 대해 검색된 문서가 해당 질문에 답하는 데 관련이 있는지 판단합니다.

평가 기준:
- 문서에 질문과 직접 관련된 법 조항, 정의, 또는 절차가 포함되어 있으면 관련성 있음(yes)
- 문서가 같은 법령이지만 다른 주제를 다루면 관련성 없음(no)
- 부분적으로 관련된 경우: 주요 질문을 답하는 데 도움이 되면 yes로 판단

반드시 JSON 형식으로만 응답하세요."""

    USER_PROMPT = """법률 질문: {query}

검색된 문서:
{document}

위 문서가 질문에 답하는 데 관련이 있습니까?

다음 JSON 형식으로 응답하세요:
{{
  "score": "yes" 또는 "no",
  "reasoning": "판단 근거 (한 문장)",
  "confidence": 0.0에서 1.0 사이의 숫자
}}"""

    def __init__(self) -> None:
        self._config = get_config()
        self._llm = ChatOpenAI(
            model=self._config.openai_model,
            temperature=0.0,
            api_key=self._config.openai_api_key,
        ).bind(response_format={"type": "json_object"})
        self._parser = JsonOutputParser(pydantic_object=RelevanceScore)
        self._chain = (
            ChatPromptTemplate.from_messages(
                [("system", self.SYSTEM_PROMPT), ("human", self.USER_PROMPT)]
            )
            | self._llm
            | self._parser
        )

    def grade(self, document: Document, query: str) -> RelevanceScore:
        """
        Grade the relevance of a single document to a query.

        Args:
            document: The retrieved Document.
            query: The user's legal query.

        Returns:
            RelevanceScore with score, reasoning, and confidence.
        """
        try:
            raw = self._chain.invoke(
                {
                    "query": query,
                    "document": document.page_content[:1000],
                }
            )
            if isinstance(raw, dict):
                return RelevanceScore(**raw)
            return raw  # type: ignore
        except Exception as exc:
            logger.warning(f"RelevanceGrader failed: {exc} – defaulting to relevant")
            return RelevanceScore(
                score="yes",
                reasoning=f"Grader error: {exc}",
                confidence=0.5,
            )

    def grade_batch(
        self, documents: list[Document], query: str
    ) -> list[RelevanceScore]:
        """Grade a list of documents in sequence."""
        return [self.grade(doc, query) for doc in documents]


class HallucinationGrader:
    """
    Detects whether a generated answer halluccinates facts not present in source documents.

    Used in Self-RAG and Corrective RAG to validate generated answers.

    Example::

        grader = HallucinationGrader()
        result = grader.grade(answer=answer, documents=retrieved_docs)
        if result.score == "no":   # hallucination detected
            trigger_regeneration()
    """

    SYSTEM_PROMPT = """당신은 AI가 생성한 법률 답변에서 환각(hallucination)을 탐지하는 전문 검수자입니다.
제공된 소스 문서를 기반으로, 생성된 답변에 소스에서 지지되지 않는 주장이 있는지 확인합니다.

평가 기준:
- 답변의 모든 법적 주장이 제공된 문서에 의해 뒷받침되면: score = "yes" (지지됨, 환각 없음)
- 답변이 소스에 없는 법 조항, 판례, 또는 사실을 주장하면: score = "no" (환각 있음)
- 일반적인 법률 상식에 해당하는 내용은 허용됩니다.

반드시 JSON 형식으로만 응답하세요."""

    USER_PROMPT = """소스 문서들:
{documents}

생성된 법률 답변:
{answer}

이 답변이 소스 문서에 근거한 내용만 포함하고 있습니까?

다음 JSON 형식으로 응답하세요:
{{
  "score": "yes" (환각 없음) 또는 "no" (환각 있음),
  "reasoning": "근거 없는 주장 목록 또는 모든 주장이 뒷받침됨을 확인",
  "confidence": 0.0에서 1.0 사이의 숫자
}}"""

    def __init__(self) -> None:
        self._config = get_config()
        self._llm = ChatOpenAI(
            model=self._config.openai_model,
            temperature=0.0,
            api_key=self._config.openai_api_key,
        ).bind(response_format={"type": "json_object"})
        self._parser = JsonOutputParser(pydantic_object=HallucinationScore)
        self._chain = (
            ChatPromptTemplate.from_messages(
                [("system", self.SYSTEM_PROMPT), ("human", self.USER_PROMPT)]
            )
            | self._llm
            | self._parser
        )

    def grade(self, answer: str, documents: list[Document]) -> HallucinationScore:
        """
        Grade whether the answer is grounded in the provided documents.

        Args:
            answer: The LLM-generated answer text.
            documents: Source documents used for generation.

        Returns:
            HallucinationScore with score (yes=no hallucination), reasoning, confidence.
        """
        try:
            # Concatenate document snippets
            doc_text = "\n\n---\n\n".join(
                [
                    f"[문서 {i+1}] {doc.metadata.get('law_name', '')} "
                    f"{doc.metadata.get('article_label', '')}\n{doc.page_content[:600]}"
                    for i, doc in enumerate(documents[:5])
                ]
            )

            raw = self._chain.invoke({"documents": doc_text, "answer": answer})
            if isinstance(raw, dict):
                return HallucinationScore(**raw)
            return raw  # type: ignore
        except Exception as exc:
            logger.warning(f"HallucinationGrader failed: {exc}")
            return HallucinationScore(
                score="yes",
                reasoning=f"Grader error: {exc}",
                confidence=0.5,
            )


class AnswerGrader:
    """
    Grades whether a generated answer adequately addresses the user's question.

    Used to decide whether to accept the answer or re-generate.

    Example::

        grader = AnswerGrader()
        result = grader.grade(answer=answer, query=query)
        if result.score == "no":
            retrigger_generation()
    """

    SYSTEM_PROMPT = """당신은 법률 답변의 품질을 평가하는 전문 검수자입니다.
생성된 답변이 사용자의 법률 질문을 충분히, 정확하게 답변하는지 평가합니다.

평가 기준:
- 질문의 핵심 법적 쟁점을 다루었는가?
- 관련 법령 조항이나 근거가 제시되었는가?
- 답변이 명확하고 실용적인가?
- 중요한 예외나 조건이 빠지지 않았는가?

반드시 JSON 형식으로만 응답하세요."""

    USER_PROMPT = """사용자 질문: {query}

생성된 답변:
{answer}

이 답변이 질문을 충분히 답하고 있습니까?

다음 JSON 형식으로 응답하세요:
{{
  "score": "yes" (충분함) 또는 "no" (불충분함),
  "reasoning": "평가 근거",
  "confidence": 0.0에서 1.0 사이의 숫자,
  "missing_aspects": ["빠진 측면1", "빠진 측면2"] (없으면 빈 배열)
}}"""

    def __init__(self) -> None:
        self._config = get_config()
        self._llm = ChatOpenAI(
            model=self._config.openai_model,
            temperature=0.0,
            api_key=self._config.openai_api_key,
        ).bind(response_format={"type": "json_object"})
        self._parser = JsonOutputParser(pydantic_object=AnswerQualityScore)
        self._chain = (
            ChatPromptTemplate.from_messages(
                [("system", self.SYSTEM_PROMPT), ("human", self.USER_PROMPT)]
            )
            | self._llm
            | self._parser
        )

    def grade(self, answer: str, query: str) -> AnswerQualityScore:
        """
        Grade whether the answer addresses the query.

        Args:
            answer: The generated answer.
            query: The original user query.

        Returns:
            AnswerQualityScore with score, reasoning, confidence, and missing_aspects.
        """
        try:
            raw = self._chain.invoke({"query": query, "answer": answer})
            if isinstance(raw, dict):
                raw.setdefault("missing_aspects", [])
                return AnswerQualityScore(**raw)
            return raw  # type: ignore
        except Exception as exc:
            logger.warning(f"AnswerGrader failed: {exc}")
            return AnswerQualityScore(
                score="yes",
                reasoning=f"Grader error: {exc}",
                confidence=0.5,
                missing_aspects=[],
            )


class QueryGrader:
    """
    Classifies an incoming legal query to determine the best RAG strategy.

    This grader is invoked at the entry point of the main agent to decide
    which RAG pipeline should handle the query.

    Example::

        grader = QueryGrader()
        classification = grader.classify("임대차 계약 해지 통보 기간은?")
        print(classification.recommended_rag)   # "adaptive"
        print(classification.query_type)        # "simple_fact"
    """

    SYSTEM_PROMPT = """당신은 법률 질문을 분류하는 전문가입니다.
사용자의 법률 질문을 분석하여 유형, 복잡도, 그리고 최적의 답변 전략을 결정합니다.

질문 유형:
- simple_fact: 단순 법 조항 조회 (예: "민법 제750조가 뭐예요?")
- legal_analysis: 복잡한 법률 분석 필요 (예: "이 상황에서 계약 해제가 가능한가요?")
- court_decision: 판례/판결 검색 필요 (예: "이런 경우 대법원 판결은?")
- calculation: 날짜/기간 계산 (예: "소멸시효가 언제 만료되나요?")
- current_event: 최근 법 개정이나 뉴스 (예: "최근 임대차법 개정 내용은?")
- general: 일반적인 법률 상식

RAG 전략:
- direct: 간단한 사실, 계산 → 즉시 답변
- adaptive: 일반 법령 조회 → 적응형 검색
- self: 복잡한 분석 → 자기반성 검색
- corrective: 판례/최신 정보 → 교정형 검색 + 웹

반드시 JSON 형식으로만 응답하세요."""

    USER_PROMPT = """법률 질문: {query}

이 질문을 분석하여 분류하세요.

다음 JSON 형식으로 응답하세요:
{{
  "query_type": "simple_fact|legal_analysis|court_decision|calculation|current_event|general",
  "complexity": "simple|moderate|complex",
  "recommended_rag": "direct|adaptive|self|corrective",
  "requires_web_search": true 또는 false,
  "reasoning": "분류 근거 (한두 문장)",
  "extracted_law_name": "언급된 법령명 또는 null",
  "extracted_article": "언급된 조문 번호(숫자만) 또는 null"
}}"""

    def __init__(self) -> None:
        self._config = get_config()
        self._llm = ChatOpenAI(
            model=self._config.openai_model,
            temperature=0.0,
            api_key=self._config.openai_api_key,
        ).bind(response_format={"type": "json_object"})
        self._parser = JsonOutputParser(pydantic_object=QueryClassification)
        self._chain = (
            ChatPromptTemplate.from_messages(
                [("system", self.SYSTEM_PROMPT), ("human", self.USER_PROMPT)]
            )
            | self._llm
            | self._parser
        )

    def classify(self, query: str) -> QueryClassification:
        """
        Classify an incoming legal query.

        Args:
            query: The user's legal question.

        Returns:
            QueryClassification with type, complexity, recommended RAG, and extracted entities.
        """
        try:
            raw = self._chain.invoke({"query": query})
            if isinstance(raw, dict):
                return QueryClassification(**raw)
            return raw  # type: ignore
        except Exception as exc:
            logger.warning(f"QueryGrader classification failed: {exc}")
            return QueryClassification(
                query_type="general",
                complexity="moderate",
                recommended_rag="adaptive",
                requires_web_search=False,
                reasoning=f"Classification failed: {exc}",
                extracted_law_name=None,
                extracted_article=None,
            )


# ---------------------------------------------------------------------------
# Composite grader helper
# ---------------------------------------------------------------------------


class GraderSuite:
    """
    Convenience wrapper holding all four graders.

    Instantiated once and passed to graph nodes to avoid re-creating LLM clients.
    """

    def __init__(self) -> None:
        self.relevance = RelevanceGrader()
        self.hallucination = HallucinationGrader()
        self.answer = AnswerGrader()
        self.query = QueryGrader()

    def compute_overall_confidence(
        self,
        relevance_scores: list[float],
        hallucination_result: Optional[HallucinationScore],
        answer_result: Optional[AnswerQualityScore],
    ) -> float:
        """
        Compute an aggregate confidence score from individual grader outputs.

        Weights:
        - Relevance (avg): 30%
        - Hallucination (1.0 if yes, 0.0 if no): 40%
        - Answer quality (confidence if yes, 0.0 if no): 30%

        Returns:
            Overall confidence in range [0.0, 1.0].
        """
        relevance_avg = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        )

        hallucination_factor = 1.0
        if hallucination_result:
            hallucination_factor = (
                hallucination_result.confidence
                if hallucination_result.score == "yes"
                else 0.0
            )

        answer_factor = 0.5
        if answer_result:
            answer_factor = (
                answer_result.confidence if answer_result.score == "yes" else 0.2
            )

        overall = (
            0.30 * relevance_avg
            + 0.40 * hallucination_factor
            + 0.30 * answer_factor
        )
        return round(min(max(overall, 0.0), 1.0), 4)
