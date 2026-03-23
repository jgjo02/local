"""ETF 추천 RAG 체인 - 사용자 프로파일 기반 ETF 추천"""

from typing import List, Dict, Any, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from .config import Config
from .etf_vector_store import ETFVectorStore


SYSTEM_PROMPT = """당신은 전문 ETF 투자 상담사입니다. 고객의 투자 성향과 목표에 맞는 최적의 ETF를 추천해드립니다.

## 추천 원칙
1. **고객 우선**: 고객의 위험 허용도, 투자 기간, 목표 수익률을 항상 고려합니다.
2. **분산 투자**: 단일 ETF보다 포트폴리오 관점에서 추천합니다.
3. **투명한 설명**: 추천 이유를 명확하게 설명합니다.
4. **리스크 고지**: 위험 요소를 솔직하게 안내합니다.

## 응답 형식
ETF를 추천할 때는 다음 형식을 사용합니다:
- **ETF명 (티커)**: 추천 이유
- 예상 역할: 포트폴리오에서의 역할
- 주의사항: 해당 ETF의 위험 요소

## 제공된 ETF 정보
{context}

---
위 ETF 정보를 참고하여 고객의 질문에 답변해주세요. 제공된 ETF 외의 상품은 언급하지 마세요.
"""

USER_PROMPT_TEMPLATE = """고객 질문: {question}"""


class ETFRecommender:
    """사용자 맞춤형 ETF 추천 RAG 시스템"""

    def __init__(self, config: Config, vector_store: ETFVectorStore):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key,
            streaming=True,
        )
        self._session_histories: Dict[str, ChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """세션별 대화 이력 반환"""
        if session_id not in self._session_histories:
            self._session_histories[session_id] = ChatMessageHistory()
        return self._session_histories[session_id]

    def _format_context(self, docs: List[Document]) -> str:
        """검색된 문서를 프롬프트용 컨텍스트로 포맷"""
        if not docs:
            return "관련 ETF 정보를 찾지 못했습니다."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[ETF {i}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def _apply_risk_filter(
        self, docs: List[Document], risk_preference: Optional[str]
    ) -> List[Document]:
        """위험 선호도에 따라 문서 필터링 및 정렬"""
        if not risk_preference:
            return docs

        risk_order = {
            "저위험": ["저위험", "중립"],
            "중립": ["중립", "저위험", "중고위험"],
            "고위험": ["고위험", "중고위험", "매우고위험", "중립"],
        }

        preferred_risks = risk_order.get(risk_preference, [])
        if not preferred_risks:
            return docs

        priority_docs = [d for d in docs if d.metadata.get("risk_level") in preferred_risks]
        other_docs = [d for d in docs if d.metadata.get("risk_level") not in preferred_risks]
        return priority_docs + other_docs

    def retrieve(
        self,
        query: str,
        risk_preference: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """쿼리에 관련된 ETF 검색"""
        k = top_k or self.config.rerank_top_n
        docs = self.vector_store.hybrid_search(query, k=self.config.top_k)
        docs = self._apply_risk_filter(docs, risk_preference)
        return docs[:k]

    def recommend(
        self,
        question: str,
        session_id: str = "default",
        risk_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """ETF 추천 실행 (단일 응답)"""
        docs = self.retrieve(question, risk_preference)
        context = self._format_context(docs)

        history = self._get_session_history(session_id)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        ]
        messages.extend(history.messages)
        messages.append(HumanMessage(content=question))

        response = self.llm.invoke(messages)
        answer = response.content

        history.add_user_message(question)
        history.add_ai_message(answer)

        return {
            "answer": answer,
            "source_docs": docs,
            "source_tickers": [d.metadata.get("ticker", "") for d in docs],
        }

    def stream_recommend(
        self,
        question: str,
        session_id: str = "default",
        risk_preference: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """ETF 추천 스트리밍"""
        docs = self.retrieve(question, risk_preference)
        context = self._format_context(docs)

        history = self._get_session_history(session_id)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        ]
        messages.extend(history.messages)
        messages.append(HumanMessage(content=question))

        full_response = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_response += token
            yield token

        history.add_user_message(question)
        history.add_ai_message(full_response)

    def get_portfolio_recommendation(
        self,
        investment_amount: int,
        risk_profile: str,
        investment_period: str,
        goals: List[str],
    ) -> Dict[str, Any]:
        """포트폴리오 구성 추천"""
        # 포트폴리오 쿼리 생성
        goal_str = ", ".join(goals)
        query = f"""
        투자금액 {investment_amount:,}만원, 위험성향 {risk_profile},
        투자기간 {investment_period}, 투자목표: {goal_str}에 맞는
        ETF 포트폴리오를 구성해주세요.
        """

        docs = self.retrieve(query, risk_preference=risk_profile, top_k=5)
        context = self._format_context(docs)

        portfolio_prompt = f"""다음 조건에 맞는 ETF 포트폴리오를 구성해주세요:

투자금액: {investment_amount:,}만원
위험성향: {risk_profile}
투자기간: {investment_period}
투자목표: {goal_str}

아래 ETF 중에서 포트폴리오를 구성하고, 각 ETF의 투자 비중(%)과 금액을 포함하여 추천해주세요.

{context}

응답 형식:
## 추천 포트폴리오

| ETF명 | 비중 | 투자금액 | 선택 이유 |
|-------|------|----------|-----------|
| ... | ...% | ...만원 | ... |

## 포트폴리오 특성
- 예상 위험도:
- 예상 수익률 범위:
- 포트폴리오 전략:
"""

        response = self.llm.invoke([HumanMessage(content=portfolio_prompt)])

        return {
            "portfolio_recommendation": response.content,
            "source_docs": docs,
            "investment_amount": investment_amount,
            "risk_profile": risk_profile,
        }

    def reset_session(self, session_id: str) -> None:
        """세션 대화 이력 초기화"""
        if session_id in self._session_histories:
            del self._session_histories[session_id]

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """대화 이력 반환"""
        history = self._get_session_history(session_id)
        result = []
        for msg in history.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            result.append({"role": role, "content": msg.content})
        return result
