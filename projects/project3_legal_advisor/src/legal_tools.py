"""
LangChain tools for the Legal Advisor Agent.

Each tool is a self-contained callable that can be bound to an LLM or invoked
directly from LangGraph nodes. Tools are schema-validated via Pydantic.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta
from typing import Any, Optional, Type

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from loguru import logger

from src.config import get_config


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class LegalDocSearchInput(BaseModel):
    query: str = Field(description="The legal search query in Korean or English")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    law_name: Optional[str] = Field(
        default=None,
        description="Optional: restrict search to a specific law (e.g. '민법')",
    )


class LawArticleLookupInput(BaseModel):
    law_name: str = Field(
        description="Korean law name, e.g. '민법', '근로기준법', '주택임대차보호법'"
    )
    article_number: str = Field(
        description="Article number (숫자만), e.g. '750', '23', '6의2'"
    )


class LegalWebSearchInput(BaseModel):
    query: str = Field(description="Legal question to search on the web")
    max_results: int = Field(default=5, ge=1, le=10)


class CourtDecisionSearchInput(BaseModel):
    query: str = Field(description="Legal issue or case description to search")
    court: Optional[str] = Field(
        default=None,
        description="Optional court filter: '대법원', '헌법재판소', '고등법원'",
    )
    max_results: int = Field(default=5, ge=1, le=10)


class LegalCalculatorInput(BaseModel):
    calculation_type: str = Field(
        description=(
            "Type of calculation: "
            "'statute_of_limitations' | 'notice_period' | 'employment_duration' | "
            "'lease_expiry' | 'custom_deadline'"
        )
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format",
    )
    duration_days: Optional[int] = Field(
        default=None,
        description="Duration in days to add to start_date",
    )
    law_reference: Optional[str] = Field(
        default=None,
        description="Law reference providing the rule, e.g. '민법 제766조'",
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context, e.g. 'tort claim' or 'wrongful termination'",
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class LegalDocSearchTool(BaseTool):
    """
    Searches the local legal document vector store using semantic similarity.

    Returns a formatted string of the top-k relevant document excerpts,
    each with its law name, article number, and source.
    """

    name: str = "legal_doc_search"
    description: str = (
        "Search the local legal document database for relevant law articles and provisions. "
        "Use this tool when you need to find specific legal rules, definitions, or procedures "
        "from Korean laws such as 민법, 근로기준법, or 주택임대차보호법."
    )
    args_schema: Type[BaseModel] = LegalDocSearchInput

    # The vector store is injected at construction time
    _vector_store: Any = None

    def __init__(self, vector_store: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass pydantic field protection
        object.__setattr__(self, "_vector_store", vector_store)

    def _run(
        self,
        query: str,
        k: int = 5,
        law_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        try:
            filter_meta = {"law_name": law_name} if law_name else None
            docs: list[Document] = self._vector_store.hybrid_search(
                query=query, k=k, filter_metadata=filter_meta
            )
            if not docs:
                return "관련 법령 조항을 찾을 수 없습니다."

            results = []
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                law = meta.get("law_name", "미상")
                article = meta.get("article_label", "")
                title = meta.get("article_title", "")
                score = meta.get("relevance_score", 0.0)
                header = f"[{i}] {law} {article}"
                if title:
                    header += f" ({title})"
                header += f" [관련도: {score:.2f}]"
                results.append(f"{header}\n{doc.page_content[:500]}")

            return "\n\n---\n\n".join(results)

        except Exception as exc:
            logger.error(f"LegalDocSearchTool error: {exc}")
            return f"검색 중 오류가 발생했습니다: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class LawArticleLookupTool(BaseTool):
    """
    Looks up a specific article (조) within a named law.

    Use when the user asks about a specific article by number, e.g.
    "민법 제750조 내용이 뭔가요?"
    """

    name: str = "law_article_lookup"
    description: str = (
        "Look up the full text of a specific article in a Korean law. "
        "Use this when you know the exact law name and article number. "
        "Example: law_name='민법', article_number='750'."
    )
    args_schema: Type[BaseModel] = LawArticleLookupInput

    _vector_store: Any = None

    def __init__(self, vector_store: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_vector_store", vector_store)

    def _run(self, law_name: str, article_number: str, **kwargs: Any) -> str:
        try:
            docs: list[Document] = self._vector_store.search_by_article(
                law_name=law_name, article_number=article_number
            )
            if not docs:
                return (
                    f"{law_name} 제{article_number}조를 찾을 수 없습니다. "
                    "법령 이름과 조문 번호를 확인하세요."
                )

            parts = []
            for doc in docs:
                meta = doc.metadata
                label = meta.get("article_label", f"제{article_number}조")
                title = meta.get("article_title", "")
                header = f"■ {law_name} {label}"
                if title:
                    header += f" [{title}]"
                parts.append(f"{header}\n\n{doc.page_content}")

            return "\n\n".join(parts)

        except Exception as exc:
            logger.error(f"LawArticleLookupTool error: {exc}")
            return f"조문 조회 중 오류: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class LegalWebSearchTool(BaseTool):
    """
    Web search specifically configured for Korean legal information.

    Uses the Tavily API with a legal-domain system prompt to return
    relevant, trustworthy results (official government sites, court sites, etc.).
    """

    name: str = "legal_web_search"
    description: str = (
        "Search the web for recent Korean legal information, recent court decisions, "
        "law amendments, or legal news. Use this when local documents may be outdated "
        "or when the question involves recent events."
    )
    args_schema: Type[BaseModel] = LegalWebSearchInput

    _config: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_config", get_config())

    def _run(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        try:
            from tavily import TavilyClient  # type: ignore

            client = TavilyClient(api_key=self._config.tavily_api_key)

            # Enhance query for legal domain
            legal_query = f"한국 법률 {query} site:law.go.kr OR site:scourt.go.kr OR site:ccourt.go.kr"

            response = client.search(
                query=legal_query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )

            results_parts: list[str] = []

            if response.get("answer"):
                results_parts.append(f"[요약 답변]\n{response['answer']}\n")

            for i, result in enumerate(response.get("results", []), 1):
                title = result.get("title", "제목 없음")
                url = result.get("url", "")
                content = result.get("content", "")[:400]
                score = result.get("score", 0.0)
                results_parts.append(
                    f"[{i}] {title}\n출처: {url}\n관련도: {score:.2f}\n{content}"
                )

            if not results_parts:
                return "관련 웹 검색 결과를 찾을 수 없습니다."

            return "\n\n---\n\n".join(results_parts)

        except ImportError:
            return "Tavily 패키지가 설치되지 않았습니다. pip install tavily-python"
        except Exception as exc:
            logger.error(f"LegalWebSearchTool error: {exc}")
            return f"웹 검색 중 오류: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class CourtDecisionSearchTool(BaseTool):
    """
    Searches for Korean court decisions and precedents.

    Uses Tavily to search the Supreme Court (대법원) and Constitutional Court
    (헌법재판소) websites for relevant case law.
    """

    name: str = "court_decision_search"
    description: str = (
        "Search for Korean court decisions, precedents, and case law. "
        "Use this for questions about how courts have ruled on specific legal issues, "
        "or when the user asks about '판례' (court precedents)."
    )
    args_schema: Type[BaseModel] = CourtDecisionSearchInput

    _config: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_config", get_config())

    def _run(
        self,
        query: str,
        court: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ) -> str:
        try:
            from tavily import TavilyClient  # type: ignore

            client = TavilyClient(api_key=self._config.tavily_api_key)

            court_filter = ""
            if court == "대법원":
                court_filter = " site:scourt.go.kr"
            elif court == "헌법재판소":
                court_filter = " site:ccourt.go.kr"
            elif court:
                court_filter = f" {court}"

            search_query = f"판례 {query}{court_filter}"

            response = client.search(
                query=search_query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
            )

            parts: list[str] = []

            if response.get("answer"):
                parts.append(f"[판례 요약]\n{response['answer']}\n")

            for i, result in enumerate(response.get("results", []), 1):
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")[:500]
                parts.append(f"[판례 {i}] {title}\n출처: {url}\n{content}")

            return "\n\n---\n\n".join(parts) if parts else "관련 판례를 찾을 수 없습니다."

        except ImportError:
            return "Tavily 패키지가 설치되지 않았습니다."
        except Exception as exc:
            logger.error(f"CourtDecisionSearchTool error: {exc}")
            return f"판례 검색 중 오류: {exc}"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class LegalCalculatorTool(BaseTool):
    """
    Performs legal date and deadline calculations.

    Handles:
    - 소멸시효 (statute of limitations) calculations
    - 해지 통고 기간 (notice periods)
    - 근로 기간 (employment duration)
    - 임대차 만료일 (lease expiry)
    - Custom deadline calculations
    """

    name: str = "legal_calculator"
    description: str = (
        "Calculate legal dates, deadlines, statutes of limitations, and notice periods. "
        "Use this for questions about '언제까지', '몇 일 이내', '시효 기간', '해지 기간' etc."
    )
    args_schema: Type[BaseModel] = LegalCalculatorInput

    # Statute of limitations lookup table (years)
    STATUTE_OF_LIMITATIONS: dict[str, int] = {
        "일반채권": 10,
        "상사채권": 5,
        "불법행위": 3,
        "임금": 3,
        "손해배상": 3,
        "계약위반": 10,
        "부당이득": 10,
        "임대차보증금": 10,
    }

    # Notice periods (days)
    NOTICE_PERIODS: dict[str, int] = {
        "임대차_해지_통고": 180,       # 임대인/임차인 6개월 전
        "임대차_묵시갱신_거절": 60,    # 계약만료 2개월 전
        "근로자_해고예고": 30,          # 30일 전
        "사용자_퇴직통보": 30,
    }

    def _run(
        self,
        calculation_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        duration_days: Optional[int] = None,
        law_reference: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        try:
            calc_type = calculation_type.lower().replace(" ", "_")

            if calc_type == "statute_of_limitations":
                return self._calc_statute_of_limitations(start_date, context)

            elif calc_type == "notice_period":
                return self._calc_notice_period(start_date, context)

            elif calc_type == "employment_duration":
                return self._calc_employment_duration(start_date, end_date)

            elif calc_type == "lease_expiry":
                return self._calc_lease_expiry(start_date, context)

            elif calc_type == "custom_deadline":
                return self._calc_custom_deadline(start_date, duration_days, law_reference)

            else:
                return (
                    f"지원하지 않는 계산 유형입니다: {calculation_type}\n"
                    "지원 유형: statute_of_limitations, notice_period, "
                    "employment_duration, lease_expiry, custom_deadline"
                )

        except Exception as exc:
            logger.error(f"LegalCalculatorTool error: {exc}")
            return f"계산 중 오류: {exc}"

    def _parse_date(self, date_str: str) -> date:
        """Parse a date string into a date object."""
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"날짜 형식을 인식할 수 없습니다: {date_str}")

    def _calc_statute_of_limitations(
        self, start_date_str: Optional[str], context: Optional[str]
    ) -> str:
        """Calculate statute of limitations deadline."""
        # Determine claim type from context
        years = 10  # default
        claim_type = "일반채권"
        legal_basis = "민법 제162조"

        if context:
            ctx_lower = context.lower()
            if "불법행위" in ctx_lower or "tort" in ctx_lower:
                years = 3
                claim_type = "불법행위"
                legal_basis = "민법 제766조"
            elif "임금" in ctx_lower or "wage" in ctx_lower:
                years = 3
                claim_type = "임금채권"
                legal_basis = "근로기준법 제49조"
            elif "상사" in ctx_lower or "commercial" in ctx_lower:
                years = 5
                claim_type = "상사채권"
                legal_basis = "상법 제64조"

        result_lines = [
            f"■ 소멸시효 계산 ({claim_type})",
            f"  근거 법령: {legal_basis}",
            f"  소멸시효 기간: {years}년",
        ]

        if start_date_str:
            start = self._parse_date(start_date_str)
            # Civil law: same date N years later
            try:
                expiry = start.replace(year=start.year + years)
            except ValueError:
                # Feb 29 edge case
                expiry = start.replace(year=start.year + years, day=28)
            today = date.today()
            remaining = (expiry - today).days

            result_lines += [
                f"  기산일: {start.strftime('%Y년 %m월 %d일')}",
                f"  소멸시효 만료일: {expiry.strftime('%Y년 %m월 %d일')}",
            ]
            if remaining > 0:
                result_lines.append(f"  오늘(기준)로부터 {remaining}일 남음")
            else:
                result_lines.append(
                    f"  ⚠️ 소멸시효가 {abs(remaining)}일 전에 만료되었습니다"
                )
        else:
            result_lines.append(f"  (기산일을 알면 만료일을 계산할 수 있습니다)")

        return "\n".join(result_lines)

    def _calc_notice_period(
        self, start_date_str: Optional[str], context: Optional[str]
    ) -> str:
        """Calculate notice period deadline."""
        notice_type = "임대차_해지_통고"
        days = 180

        if context:
            ctx_lower = context.lower()
            if "묵시갱신" in ctx_lower or "자동갱신" in ctx_lower:
                notice_type = "임대차_묵시갱신_거절"
                days = 60
            elif "해고" in ctx_lower or "해임" in ctx_lower:
                notice_type = "근로자_해고예고"
                days = 30

        days_for_type = self.NOTICE_PERIODS.get(notice_type, days)

        result_lines = [
            f"■ 통지 기한 계산",
            f"  유형: {notice_type.replace('_', ' ')}",
            f"  통지 기한: {days_for_type}일 전",
        ]

        if start_date_str:
            # start_date here is the event date (e.g. lease expiry)
            event_date = self._parse_date(start_date_str)
            notice_deadline = event_date - timedelta(days=days_for_type)
            today = date.today()
            days_until = (notice_deadline - today).days

            result_lines += [
                f"  기준일 (만료/이벤트 일): {event_date.strftime('%Y년 %m월 %d일')}",
                f"  통지 마감일: {notice_deadline.strftime('%Y년 %m월 %d일')}",
            ]
            if days_until > 0:
                result_lines.append(f"  통지 마감까지 {days_until}일 남았습니다")
            else:
                result_lines.append(
                    f"  ⚠️ 통지 마감일이 {abs(days_until)}일 전에 지났습니다"
                )

        return "\n".join(result_lines)

    def _calc_employment_duration(
        self, start_date_str: Optional[str], end_date_str: Optional[str]
    ) -> str:
        """Calculate employment duration and related rights."""
        if not start_date_str:
            return "고용 시작일이 필요합니다 (start_date 파라미터)"

        start = self._parse_date(start_date_str)
        end = self._parse_date(end_date_str) if end_date_str else date.today()
        delta = end - start
        years = delta.days // 365
        months = (delta.days % 365) // 30
        remaining_days = delta.days % 30

        # Determine entitlements
        entitlements = []
        if delta.days >= 365:
            # Annual leave (근로기준법 제60조)
            annual_leave_days = min(15 + (years - 1), 25)
            entitlements.append(f"연차 유급휴가: {annual_leave_days}일 (근로기준법 제60조)")
        if years >= 1:
            # Severance pay (근로자퇴직급여 보장법)
            entitlements.append(f"퇴직금 청구권 발생 (근로자퇴직급여 보장법 제8조)")
        if delta.days < 30:
            entitlements.append("⚠️ 수습기간 (30일 미만, 해고예고 불필요)")

        result_lines = [
            f"■ 근로 기간 계산",
            f"  시작일: {start.strftime('%Y년 %m월 %d일')}",
            f"  기준일: {end.strftime('%Y년 %m월 %d일')}",
            f"  근속 기간: {years}년 {months}개월 {remaining_days}일 (총 {delta.days}일)",
            "",
            "  [관련 권리]",
        ]
        if entitlements:
            for e in entitlements:
                result_lines.append(f"  • {e}")
        else:
            result_lines.append("  • 아직 발생한 특별 권리 없음")

        return "\n".join(result_lines)

    def _calc_lease_expiry(
        self, start_date_str: Optional[str], context: Optional[str]
    ) -> str:
        """Calculate lease expiry and related notice deadlines."""
        if not start_date_str:
            return "임대차 시작일이 필요합니다 (start_date 파라미터)"

        start = self._parse_date(start_date_str)
        duration_years = 2  # Standard residential lease (주택임대차보호법 제4조)

        if context and "1년" in context:
            duration_years = 1

        try:
            expiry = start.replace(year=start.year + duration_years)
        except ValueError:
            expiry = start.replace(year=start.year + duration_years, day=28)

        # Notice period for refusing automatic renewal: 2 months before expiry
        notice_deadline = expiry - timedelta(days=60)
        today = date.today()

        result_lines = [
            f"■ 임대차 기간 계산 (주택임대차보호법 제4조)",
            f"  임대차 시작일: {start.strftime('%Y년 %m월 %d일')}",
            f"  계약 기간: {duration_years}년",
            f"  임대차 만료일: {expiry.strftime('%Y년 %m월 %d일')}",
            "",
            f"  묵시적 갱신 거절 통지 마감: {notice_deadline.strftime('%Y년 %m월 %d일')}",
            f"  (만료 2개월 전까지 통지 필요: 주택임대차보호법 제6조의3)",
        ]

        days_to_expiry = (expiry - today).days
        if days_to_expiry > 0:
            result_lines.append(f"\n  만료까지 {days_to_expiry}일 남았습니다")
        else:
            result_lines.append(
                f"\n  ⚠️ 임대차 계약이 {abs(days_to_expiry)}일 전에 만료되었습니다"
            )

        return "\n".join(result_lines)

    def _calc_custom_deadline(
        self,
        start_date_str: Optional[str],
        duration_days: Optional[int],
        law_reference: Optional[str],
    ) -> str:
        """Calculate a custom deadline by adding days to a start date."""
        if not start_date_str:
            return "시작일이 필요합니다 (start_date 파라미터)"
        if not duration_days:
            return "기간(일 수)이 필요합니다 (duration_days 파라미터)"

        start = self._parse_date(start_date_str)
        deadline = start + timedelta(days=duration_days)
        today = date.today()
        remaining = (deadline - today).days

        result_lines = [
            f"■ 법정 기한 계산",
            f"  시작일: {start.strftime('%Y년 %m월 %d일')}",
            f"  기간: {duration_days}일",
            f"  마감일: {deadline.strftime('%Y년 %m월 %d일')}",
        ]
        if law_reference:
            result_lines.append(f"  근거: {law_reference}")
        if remaining > 0:
            result_lines.append(f"  마감까지 {remaining}일 남았습니다")
        else:
            result_lines.append(
                f"  ⚠️ 마감일이 {abs(remaining)}일 전에 지났습니다"
            )

        return "\n".join(result_lines)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def build_legal_tools(vector_store: Any) -> list[BaseTool]:
    """
    Instantiate and return all legal tools with shared dependencies injected.

    Args:
        vector_store: A LegalVectorStore instance.

    Returns:
        List of fully initialised BaseTool instances.
    """
    return [
        LegalDocSearchTool(vector_store=vector_store),
        LawArticleLookupTool(vector_store=vector_store),
        LegalWebSearchTool(),
        CourtDecisionSearchTool(),
        LegalCalculatorTool(),
    ]
