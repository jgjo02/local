"""
Project 3: 법률 자문 AI - 메인 에이전트
Adaptive RAG, Self-RAG, Corrective RAG를 통합한 법률 자문 시스템
"""

import os
from typing import Optional, Generator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory

from .config import Config
from .legal_vector_store import LegalVectorStore
from .adaptive_rag_graph import build_adaptive_rag_graph
from .self_rag_graph import build_self_rag_graph
from .corrective_rag_graph import build_corrective_rag_graph

load_dotenv()


class LegalAdvisorAgent:
    """법률 자문 메인 에이전트"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature
        )
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self.vector_store: Optional[LegalVectorStore] = None
        self.graphs = {}
        self.sessions: dict[str, InMemoryChatMessageHistory] = {}
        self._initialized = False

    def initialize(self, documents_path: Optional[str] = None):
        """에이전트 초기화 - 벡터 저장소 및 그래프 구성"""
        # 벡터 저장소 구축
        self.vector_store = LegalVectorStore(self.config)
        if documents_path:
            self.vector_store.build_from_files([documents_path])
        elif os.path.exists(self.config.vector_store_path):
            self.vector_store.load(self.config.vector_store_path)

        retriever = self.vector_store.get_retriever()

        # 각 RAG 그래프 구성
        self.graphs = {
            "adaptive": build_adaptive_rag_graph(retriever, self.llm, self.config),
            "self": build_self_rag_graph(retriever, self.llm, self.config),
            "corrective": build_corrective_rag_graph(retriever, self.llm, self.config),
        }

        self._initialized = True
        print("법률 자문 에이전트 초기화 완료")

    def _select_rag_strategy(self, query: str) -> str:
        """질문 유형에 따라 RAG 전략 자동 선택"""
        query_lower = query.lower()

        # 최신 정보 관련 → Corrective RAG (웹 검색 폴백)
        if any(kw in query_lower for kw in ["최근", "최신", "2024", "2025", "올해"]):
            return "corrective"

        # 복잡한 분석 필요 → Self-RAG (환각 체크)
        if any(kw in query_lower for kw in ["분석", "비교", "검토", "판단", "가능성"]):
            return "self"

        # 기본 → Adaptive RAG
        return "adaptive"

    def chat(
        self,
        query: str,
        session_id: str = "default",
        rag_type: Optional[str] = None,
    ) -> str:
        """법률 질문에 답변"""
        if not self._initialized:
            raise ValueError("먼저 initialize()를 호출하세요")

        # RAG 전략 선택
        strategy = rag_type or self._select_rag_strategy(query)
        graph = self.graphs.get(strategy, self.graphs["adaptive"])

        # 세션 이력 가져오기
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()

        history = self.sessions[session_id].messages

        # 그래프 실행
        config = {"configurable": {"thread_id": session_id}}
        result = graph.invoke(
            {
                "query": query,
                "documents": [],
                "answer": "",
                "generation_count": 0,
                "is_grounded": False,
                "is_useful": False,
            },
            config=config
        )

        answer = result.get("answer", "답변을 생성할 수 없습니다.")

        # 법적 면책 조항 추가
        disclaimer = "\n\n*이 답변은 일반적인 법률 정보를 제공하며, 구체적인 법률 자문을 대체할 수 없습니다. 실제 법적 문제는 변호사와 상담하시기 바랍니다.*"
        full_answer = answer + disclaimer

        # 세션 이력 업데이트
        from langchain_core.messages import HumanMessage, AIMessage
        self.sessions[session_id].add_message(HumanMessage(content=query))
        self.sessions[session_id].add_message(AIMessage(content=full_answer))

        return full_answer

    def stream_chat(
        self,
        query: str,
        session_id: str = "default",
        rag_type: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """스트리밍 답변 생성"""
        # 비스트리밍으로 전체 답변 생성 후 토큰 단위로 yield
        answer = self.chat(query, session_id, rag_type)
        for char in answer:
            yield char

    def get_conversation_history(self, session_id: str) -> list:
        """대화 이력 반환"""
        if session_id not in self.sessions:
            return []
        return [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            }
            for msg in self.sessions[session_id].messages
        ]

    def reset_conversation(self, session_id: str):
        """대화 이력 초기화"""
        if session_id in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()

    def get_stats(self) -> dict:
        """에이전트 통계"""
        return {
            "initialized": self._initialized,
            "active_sessions": len(self.sessions),
            "available_strategies": list(self.graphs.keys()),
            "vector_store_docs": self.vector_store.get_document_count() if self.vector_store else 0,
        }
