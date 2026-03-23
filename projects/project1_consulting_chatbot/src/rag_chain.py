"""
RAG chain module for AI Consulting Chatbot.

Implements a conversational RAG pipeline using LangChain Expression
Language (LCEL) with chat history support, streaming, and source
document retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Iterator, Optional

from langchain.schema import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (Korean)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """당신은 주택청약 전문 AI 상담사입니다.
주택청약, 청약 자격, 당첨 절차, 분양가, 금융 지원, 특별공급 등 주택청약과 관련된 모든 질문에
친절하고 정확하게 답변해 주세요.

다음 원칙을 지켜 주세요:
1. 제공된 참고 문서(context)를 최우선으로 활용하세요.
2. 참고 문서에 없는 내용은 일반적인 지식으로 보완하되, 불확실한 경우 명확히 밝혀 주세요.
3. 법령이나 정책은 변경될 수 있으므로 최신 정보를 확인하도록 안내해 주세요.
4. 전문 용어는 쉽게 풀어서 설명해 주세요.
5. 답변은 명확하고 간결하게 작성하되, 필요한 경우 항목별로 정리해 주세요.
6. 개인적인 법률/재정 자문은 전문가와 상담하도록 안내해 주세요.

참고 문서:
{context}"""


class RAGChain:
    """
    Conversational RAG chain built with LCEL.

    Combines a vector-store retriever with an OpenAI chat model,
    maintaining conversation history across turns.

    Example usage::

        from src.config import get_config
        from src.vector_store import VectorStoreManager
        from src.rag_chain import RAGChain

        cfg = get_config()
        vsm = VectorStoreManager(cfg)
        vsm.load_existing()

        rag = RAGChain(cfg, vsm)
        rag.build_chain()
        answer = rag.chat("청약 자격 조건이 무엇인가요?", history=[])
    """

    def __init__(self, config: Any, vector_store_manager: Any) -> None:
        """
        Args:
            config:               A ``Config`` instance.
            vector_store_manager: A ``VectorStoreManager`` instance with
                                  a loaded store.
        """
        self.config = config
        self.vsm = vector_store_manager
        self._chain: Any = None
        self._llm: Optional[ChatOpenAI] = None
        self._retriever: Any = None

        logger.info("RAGChain initialised.")

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        """Return (and cache) the ChatOpenAI model."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.model.chat_model,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                openai_api_key=self.config.openai.api_key,
                openai_organization=self.config.openai.org_id,
                streaming=True,  # enable token-by-token streaming
            )
            logger.debug(
                "LLM created: %s (temp=%.2f)",
                self.config.model.chat_model,
                self.config.model.temperature,
            )
        return self._llm

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def create_prompt(
        self, system_prompt: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate for conversational RAG.

        The template includes:
        - A system message with the context injected.
        - A ``MessagesPlaceholder`` for conversation history.
        - A human message for the current question.

        Args:
            system_prompt: Custom system prompt string. Falls back to
                           ``DEFAULT_SYSTEM_PROMPT`` if not provided.

        Returns:
            A ``ChatPromptTemplate`` instance.
        """
        effective_system = system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", effective_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        logger.debug("Prompt template created.")
        return prompt

    # ------------------------------------------------------------------
    # Document formatting
    # ------------------------------------------------------------------

    def format_docs(self, docs: list[Document]) -> str:
        """
        Format a list of retrieved Documents into a single context string.

        Each document is prefixed with its source and a separator line
        so the LLM can distinguish between different passages.

        Args:
            docs: Retrieved Document objects.

        Returns:
            Formatted context string.
        """
        if not docs:
            return "참고할 문서가 없습니다."

        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "알 수 없음")
            source_name = source.split("/")[-1] if "/" in source else source
            page = doc.metadata.get("page", "")
            page_info = f" (페이지 {page + 1})" if page != "" else ""
            header = f"[문서 {i}] 출처: {source_name}{page_info}"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def build_chain(self, system_prompt: Optional[str] = None) -> None:
        """
        Construct the full LCEL RAG chain and store it in ``self._chain``.

        Chain architecture::

            question + chat_history
                    │
                    ▼
            [Retriever]  ─────── context docs
                    │
                    ▼
            [Prompt Template] ← context (formatted) + question + history
                    │
                    ▼
              [ChatOpenAI]
                    │
                    ▼
            [StrOutputParser]  ──► answer string

        Args:
            system_prompt: Optional custom system prompt.
        """
        if not self.vsm.is_loaded:
            raise RuntimeError(
                "VectorStoreManager has no loaded store. "
                "Call build_from_documents() or load_existing() first."
            )

        self._retriever = self.vsm.get_retriever(
            search_type=self.config.retriever.search_type,
            k=self.config.retriever.k,
        )
        prompt = self.create_prompt(system_prompt)
        llm = self._get_llm()

        # Step 1: retrieve docs based on the question
        retrieve_docs = RunnableLambda(
            lambda x: self._retriever.invoke(x["question"])
        )

        # Step 2: format docs to context string
        format_context = RunnableLambda(
            lambda docs: self.format_docs(docs)
        )

        # Build the full chain
        self._chain = (
            RunnableParallel(
                {
                    "context": retrieve_docs | format_context,
                    "question": RunnablePassthrough()
                    | RunnableLambda(lambda x: x["question"]),
                    "chat_history": RunnablePassthrough()
                    | RunnableLambda(lambda x: x.get("chat_history", [])),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info(
            "RAG chain built (model=%s, retriever=%s, k=%d).",
            self.config.model.chat_model,
            self.config.retriever.search_type,
            self.config.retriever.k,
        )

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_history(
        history: list[tuple[str, str]] | list[dict[str, str]],
    ) -> list[HumanMessage | AIMessage]:
        """
        Convert Gradio-style history into LangChain message objects.

        Gradio passes history as a list of [user_msg, assistant_msg] pairs
        (list of 2-element lists / tuples) or, in newer versions, as dicts
        with ``role`` and ``content`` keys.

        Args:
            history: Conversation history in Gradio format.

        Returns:
            List of LangChain ``HumanMessage`` / ``AIMessage`` objects.
        """
        messages: list[HumanMessage | AIMessage] = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, ai_msg = item
                if user_msg:
                    messages.append(HumanMessage(content=str(user_msg)))
                if ai_msg:
                    messages.append(AIMessage(content=str(ai_msg)))
        return messages

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        history: list[Any] | None = None,
    ) -> str:
        """
        Process a single chat turn and return a complete response string.

        Args:
            message: The user's current question.
            history: Previous conversation turns in Gradio format.

        Returns:
            The assistant's response as a plain string.

        Raises:
            RuntimeError: If the chain has not been built yet.
        """
        if self._chain is None:
            raise RuntimeError(
                "RAG chain is not built. Call build_chain() first."
            )
        if not message or not message.strip():
            return "질문을 입력해 주세요."

        chat_history = self._convert_history(history or [])

        logger.debug("chat() question=%r, history_turns=%d", message[:80], len(chat_history))

        try:
            response: str = self._chain.invoke(
                {
                    "question": message.strip(),
                    "chat_history": chat_history,
                }
            )
        except Exception as exc:
            logger.error("RAG chain invocation failed: %s", exc)
            raise RuntimeError(f"답변 생성 중 오류가 발생했습니다: {exc}") from exc

        logger.debug("chat() response length=%d chars", len(response))
        return response

    def stream_chat(
        self,
        message: str,
        history: list[Any] | None = None,
    ) -> Iterator[str]:
        """
        Stream the response token-by-token for real-time UI updates.

        Yields incremental text chunks as the LLM generates them.

        Args:
            message: The user's current question.
            history: Previous conversation turns in Gradio format.

        Yields:
            Incremental string chunks of the assistant's response.

        Raises:
            RuntimeError: If the chain has not been built yet.
        """
        if self._chain is None:
            raise RuntimeError(
                "RAG chain is not built. Call build_chain() first."
            )
        if not message or not message.strip():
            yield "질문을 입력해 주세요."
            return

        chat_history = self._convert_history(history or [])

        logger.debug(
            "stream_chat() question=%r, history_turns=%d", message[:80], len(chat_history)
        )

        try:
            for chunk in self._chain.stream(
                {
                    "question": message.strip(),
                    "chat_history": chat_history,
                }
            ):
                if chunk:
                    yield chunk
        except Exception as exc:
            logger.error("RAG chain streaming failed: %s", exc)
            yield f"\n\n오류가 발생했습니다: {exc}"

    # ------------------------------------------------------------------
    # Source document retrieval
    # ------------------------------------------------------------------

    def get_source_documents(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[Document]:
        """
        Retrieve the documents that would be used for a given query.

        Useful for showing sources in the UI or debugging retrieval quality.

        Args:
            query: The search query string.
            k:     Number of documents to retrieve (defaults to config).

        Returns:
            List of relevant Document objects.

        Raises:
            RuntimeError: If no vector store is loaded.
        """
        if not self.vsm.is_loaded:
            raise RuntimeError(
                "VectorStoreManager has no loaded store."
            )

        docs = self.vsm.similarity_search(query, k=k)
        logger.debug(
            "get_source_documents: query=%r, found=%d", query[:60], len(docs)
        )
        return docs

    def get_source_documents_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve source documents together with their relevance scores.

        Args:
            query: The search query string.
            k:     Number of documents to retrieve.

        Returns:
            List of (Document, score) tuples.
        """
        return self.vsm.similarity_search_with_score(query, k=k)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def format_source_citations(self, docs: list[Document]) -> str:
        """
        Format source documents into a citation block for UI display.

        Args:
            docs: Retrieved source documents.

        Returns:
            Markdown-formatted citation string.
        """
        if not docs:
            return ""

        lines = ["**참고 출처:**"]
        seen: set[str] = set()
        for doc in docs:
            source = doc.metadata.get("source", "알 수 없음")
            source_name = source.split("/")[-1] if "/" in source else source
            page = doc.metadata.get("page", "")
            key = f"{source_name}:{page}"
            if key not in seen:
                seen.add(key)
                page_info = f" (p.{int(page) + 1})" if page != "" else ""
                lines.append(f"- {source_name}{page_info}")

        return "\n".join(lines)

    def is_ready(self) -> bool:
        """Return True if the chain has been built and is ready to answer."""
        return self._chain is not None

    def __repr__(self) -> str:
        status = "ready" if self.is_ready() else "not_built"
        return (
            f"RAGChain("
            f"model={self.config.model.chat_model!r}, "
            f"status={status!r})"
        )
