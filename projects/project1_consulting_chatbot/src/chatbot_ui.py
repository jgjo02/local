"""
Gradio UI module for AI Consulting Chatbot.

Builds a feature-rich Gradio Blocks interface with:
- Streaming chat
- Document upload for dynamic knowledge-base updates
- Source document display
- Conversation history management
- System-prompt customisation
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator, Iterator, Optional

import gradio as gr

logger = logging.getLogger(__name__)


class ChatbotUI:
    """
    Gradio-based chat interface for the housing subscription AI consultant.

    Wraps a ``RAGChain`` instance and exposes all interaction through a
    Gradio Blocks layout.

    Example usage::

        from src.chatbot_ui import ChatbotUI
        ui = ChatbotUI(rag_chain=rag, config=cfg, doc_processor=processor, vsm=vsm)
        ui.launch(share=False, port=7860)
    """

    # Example questions shown as quick-start buttons
    EXAMPLE_QUESTIONS = [
        "청약 1순위 자격 조건이 무엇인가요?",
        "청약통장은 어떻게 만드나요?",
        "특별공급 종류에는 어떤 것이 있나요?",
        "당첨 후 어떤 절차를 밟아야 하나요?",
        "분양가 상한제란 무엇인가요?",
        "무주택자 기준이 어떻게 되나요?",
    ]

    def __init__(
        self,
        rag_chain: Any,
        config: Any,
        doc_processor: Optional[Any] = None,
        vsm: Optional[Any] = None,
    ) -> None:
        """
        Args:
            rag_chain:     A built ``RAGChain`` instance.
            config:        A ``Config`` instance.
            doc_processor: Optional ``DocumentProcessor`` for document uploads.
            vsm:           Optional ``VectorStoreManager`` for document uploads.
        """
        self.rag = rag_chain
        self.config = config
        self.doc_processor = doc_processor
        self.vsm = vsm
        self._interface: Optional[gr.Blocks] = None

        logger.info("ChatbotUI initialised.")

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    def process_message(
        self,
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
    ) -> tuple[list[dict[str, str]], str]:
        """
        Non-streaming message handler.

        Args:
            message:       User's current input.
            history:       Chat history list (Gradio messages format).
            system_prompt: Custom system prompt text.

        Returns:
            Updated history list and an empty string to clear the input box.
        """
        if not message or not message.strip():
            return history, ""

        history = history or []
        history.append({"role": "user", "content": message.strip()})

        try:
            # Rebuild chain with custom prompt if changed
            if system_prompt and system_prompt.strip():
                self.rag.build_chain(system_prompt=system_prompt.strip())

            response = self.rag.chat(message, history=history[:-1])  # exclude current turn
            history.append({"role": "assistant", "content": response})
        except Exception as exc:
            logger.error("process_message error: %s", exc)
            history.append(
                {"role": "assistant", "content": f"오류가 발생했습니다: {exc}"}
            )

        return history, ""

    def stream_response(
        self,
        message: str,
        history: list[dict[str, str]],
        system_prompt: str,
    ) -> Iterator[tuple[list[dict[str, str]], str]]:
        """
        Streaming message handler — yields partial responses token by token.

        Each yield produces the updated history (with the partial AI reply)
        and an empty string for the input box, causing Gradio to update live.

        Args:
            message:       User's current input.
            history:       Chat history list.
            system_prompt: Custom system prompt text.

        Yields:
            (updated_history, empty_input_string) tuples.
        """
        if not message or not message.strip():
            yield history or [], ""
            return

        history = list(history or [])
        history.append({"role": "user", "content": message.strip()})

        # Immediately show the user message
        yield history, ""

        # Rebuild chain with custom prompt if provided
        if system_prompt and system_prompt.strip():
            try:
                self.rag.build_chain(system_prompt=system_prompt.strip())
            except Exception as exc:
                logger.warning("Failed to rebuild chain with custom prompt: %s", exc)

        # Start streaming
        history.append({"role": "assistant", "content": ""})
        accumulated = ""

        try:
            for chunk in self.rag.stream_chat(message, history=history[:-2]):
                accumulated += chunk
                history[-1]["content"] = accumulated
                yield history, ""
        except Exception as exc:
            logger.error("stream_response error: %s", exc)
            history[-1]["content"] = f"오류가 발생했습니다: {exc}"
            yield history, ""

    def clear_history(self) -> tuple[list, str]:
        """
        Clear the conversation history.

        Returns:
            Empty history list and empty input string.
        """
        logger.debug("Conversation history cleared.")
        return [], ""

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------

    def upload_document(self, file_obj: Any) -> str:
        """
        Handle document upload, process it, and add it to the vector store.

        Args:
            file_obj: Gradio file object (has a ``.name`` attribute pointing
                      to the temporary file path).

        Returns:
            Status message string to display to the user.
        """
        if file_obj is None:
            return "파일을 선택해 주세요."

        if self.doc_processor is None or self.vsm is None:
            return "문서 업로드 기능이 초기화되지 않았습니다."

        try:
            tmp_path = Path(file_obj.name)
            ext = tmp_path.suffix.lower()

            if ext not in {".pdf", ".txt", ".md"}:
                return f"지원하지 않는 파일 형식입니다: {ext}. PDF 또는 TXT 파일을 업로드해 주세요."

            logger.info("Processing uploaded file: %s", tmp_path)

            # Copy to data directory for persistence
            dest_path = self.config.app.data_dir / tmp_path.name
            shutil.copy2(tmp_path, dest_path)

            # Load and process
            if ext == ".pdf":
                docs = self.doc_processor.load_pdf(dest_path)
            else:
                docs = self.doc_processor.load_text(dest_path)

            chunks = self.doc_processor.split_documents(docs)

            if not chunks:
                return "문서에서 텍스트를 추출할 수 없었습니다."

            # Add to vector store
            self.vsm.add_documents(chunks)
            self.vsm.save()

            # Rebuild retriever in the RAG chain
            self.rag._retriever = self.vsm.get_retriever(
                search_type=self.config.retriever.search_type,
                k=self.config.retriever.k,
            )

            msg = (
                f"문서 업로드 완료!\n"
                f"- 파일명: {tmp_path.name}\n"
                f"- 처리된 청크: {len(chunks)}개\n"
                f"- 벡터 DB 문서 수: {self.vsm.get_document_count()}개"
            )
            logger.info("Document upload successful: %s (%d chunks)", tmp_path.name, len(chunks))
            return msg

        except Exception as exc:
            logger.error("Document upload failed: %s", exc)
            return f"문서 처리 중 오류가 발생했습니다: {exc}"

    # ------------------------------------------------------------------
    # Source display
    # ------------------------------------------------------------------

    def show_sources(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> str:
        """
        Return a formatted string of source documents for the last question.

        Args:
            message: Current user input (may be empty if using history).
            history: Chat history.

        Returns:
            Markdown string listing source documents.
        """
        query = message.strip()
        if not query and history:
            # Use the last user message if current input is empty
            for item in reversed(history):
                if item.get("role") == "user":
                    query = item.get("content", "")
                    break

        if not query:
            return "질문을 입력하면 참고 문서를 표시합니다."

        try:
            docs_with_scores = self.rag.get_source_documents_with_scores(query, k=3)
            if not docs_with_scores:
                return "관련 참고 문서를 찾지 못했습니다."

            parts = ["### 참고 문서\n"]
            for i, (doc, score) in enumerate(docs_with_scores, start=1):
                source = doc.metadata.get("source", "알 수 없음")
                source_name = source.split("/")[-1] if "/" in source else source
                page = doc.metadata.get("page", "")
                page_info = f" (p.{int(page) + 1})" if page != "" else ""
                preview = doc.page_content[:200].replace("\n", " ")
                parts.append(
                    f"**[{i}] {source_name}{page_info}**  \n"
                    f"유사도: `{score:.3f}`  \n"
                    f"{preview}…\n"
                )
            return "\n".join(parts)
        except Exception as exc:
            logger.error("show_sources error: %s", exc)
            return f"참고 문서 조회 중 오류: {exc}"

    # ------------------------------------------------------------------
    # Interface creation
    # ------------------------------------------------------------------

    def create_interface(self) -> gr.Blocks:
        """
        Build and return the full Gradio Blocks interface.

        Layout::

            ┌─────────────────────────────────────────────┐
            │  Header: app name + description              │
            ├──────────────────────┬──────────────────────┤
            │  Chatbot panel       │  Side panel           │
            │  - Chatbot component │  - System prompt      │
            │  - Input row         │  - Upload             │
            │  - Example buttons   │  - Sources accordion  │
            │  - Clear / Submit    │                       │
            └──────────────────────┴──────────────────────┘

        Returns:
            Configured ``gr.Blocks`` instance (not yet launched).
        """
        app_name = self.config.app.app_name
        app_desc = self.config.app.app_description
        theme_name = self.config.gradio.theme

        # Resolve theme
        theme_map = {
            "soft": gr.themes.Soft(),
            "base": gr.themes.Base(),
            "monochrome": gr.themes.Monochrome(),
            "glass": gr.themes.Glass(),
        }
        theme = theme_map.get(theme_name, gr.themes.Soft())

        with gr.Blocks(
            title=app_name,
            theme=theme,
            css=self._custom_css(),
        ) as interface:

            # ----------------------------------------------------------
            # Header
            # ----------------------------------------------------------
            gr.HTML(
                f"""
                <div class="header">
                    <h1>🏠 {app_name}</h1>
                    <p>{app_desc}</p>
                </div>
                """
            )

            with gr.Row():
                # ----------------------------------------------------------
                # Left panel: main chat area
                # ----------------------------------------------------------
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="대화",
                        height=500,
                        show_copy_button=True,
                        avatar_images=(
                            None,  # user avatar (None = default)
                            "https://api.dicebear.com/7.x/bottts/svg?seed=housing",  # bot
                        ),
                        type="messages",
                        bubble_full_width=False,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="주택청약에 대해 궁금한 점을 물어보세요...",
                            label="",
                            scale=5,
                            container=False,
                            lines=2,
                            max_lines=5,
                        )
                        submit_btn = gr.Button(
                            "전송",
                            variant="primary",
                            scale=1,
                            min_width=80,
                        )

                    with gr.Row():
                        clear_btn = gr.Button("대화 초기화", variant="secondary", scale=1)
                        sources_btn = gr.Button("참고 문서 보기", variant="secondary", scale=1)

                    # Example questions
                    gr.Markdown("**예시 질문:**")
                    with gr.Row():
                        example_btns = [
                            gr.Button(q, size="sm", variant="secondary")
                            for q in self.EXAMPLE_QUESTIONS[:3]
                        ]
                    with gr.Row():
                        example_btns += [
                            gr.Button(q, size="sm", variant="secondary")
                            for q in self.EXAMPLE_QUESTIONS[3:]
                        ]

                # ----------------------------------------------------------
                # Right panel: settings + upload + sources
                # ----------------------------------------------------------
                with gr.Column(scale=1):
                    with gr.Accordion("시스템 설정", open=False):
                        system_prompt_input = gr.Textbox(
                            label="시스템 프롬프트",
                            placeholder="AI의 역할을 정의하는 프롬프트를 입력하세요...",
                            lines=8,
                            value="",
                            info="비워두면 기본 프롬프트를 사용합니다.",
                        )
                        model_info = gr.Markdown(
                            f"**모델:** {self.config.model.chat_model}  \n"
                            f"**온도:** {self.config.model.temperature}  \n"
                            f"**검색 방식:** {self.config.retriever.search_type}  \n"
                            f"**검색 문서 수:** {self.config.retriever.k}"
                        )

                    with gr.Accordion("문서 업로드", open=False):
                        upload_file = gr.File(
                            label="PDF 또는 TXT 파일 업로드",
                            file_types=[".pdf", ".txt", ".md"],
                            file_count="single",
                        )
                        upload_btn = gr.Button("업로드 처리", variant="primary")
                        upload_status = gr.Textbox(
                            label="업로드 상태",
                            interactive=False,
                            lines=4,
                        )

                    with gr.Accordion("참고 문서", open=False) as sources_accordion:
                        sources_output = gr.Markdown(
                            "질문 후 '참고 문서 보기'를 클릭하세요."
                        )

            # ----------------------------------------------------------
            # Status bar
            # ----------------------------------------------------------
            with gr.Row():
                status_bar = gr.Markdown(
                    f"**상태:** 준비 완료 | "
                    f"벡터 DB 문서 수: {self.vsm.get_document_count() if self.vsm else 'N/A'}개"
                )

            # ----------------------------------------------------------
            # Event wiring
            # ----------------------------------------------------------

            # Submit on Enter key in textbox
            msg_input.submit(
                fn=self.stream_response,
                inputs=[msg_input, chatbot, system_prompt_input],
                outputs=[chatbot, msg_input],
            )

            # Submit button click
            submit_btn.click(
                fn=self.stream_response,
                inputs=[msg_input, chatbot, system_prompt_input],
                outputs=[chatbot, msg_input],
            )

            # Clear conversation
            clear_btn.click(
                fn=self.clear_history,
                inputs=[],
                outputs=[chatbot, msg_input],
            )

            # Show source documents
            sources_btn.click(
                fn=self.show_sources,
                inputs=[msg_input, chatbot],
                outputs=[sources_output],
            )

            # Example question buttons — clicking fills the input box
            for btn in example_btns:
                btn.click(
                    fn=lambda q: q,
                    inputs=[btn],
                    outputs=[msg_input],
                )

            # Document upload
            upload_btn.click(
                fn=self.upload_document,
                inputs=[upload_file],
                outputs=[upload_status],
            )

        self._interface = interface
        logger.info("Gradio interface created.")
        return interface

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(
        self,
        share: Optional[bool] = None,
        port: Optional[int] = None,
        server_name: Optional[str] = None,
    ) -> None:
        """
        Build (if needed) and launch the Gradio server.

        Args:
            share:       Override config share setting.
            port:        Override config port.
            server_name: Override config server name / bind address.
        """
        if self._interface is None:
            self.create_interface()

        effective_share = share if share is not None else self.config.gradio.share
        effective_port = port or self.config.gradio.server_port
        effective_server = server_name or self.config.gradio.server_name
        auth = self.config.gradio.auth

        logger.info(
            "Launching Gradio on %s:%d (share=%s)",
            effective_server,
            effective_port,
            effective_share,
        )

        launch_kwargs: dict[str, Any] = {
            "server_name": effective_server,
            "server_port": effective_port,
            "share": effective_share,
            "show_error": True,
        }
        if auth:
            launch_kwargs["auth"] = auth

        self._interface.launch(**launch_kwargs)

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------

    @staticmethod
    def _custom_css() -> str:
        """Return a CSS string for custom styling."""
        return """
        .header {
            text-align: center;
            padding: 20px 0 10px 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 15px;
        }
        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1d4ed8;
            margin-bottom: 4px;
        }
        .header p {
            color: #6b7280;
            font-size: 0.95rem;
        }
        /* Make the chatbot bubbles slightly wider */
        .message-bubble-border {
            max-width: 85% !important;
        }
        """
