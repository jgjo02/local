"""
5차시: Gradio 챗봇 구현
- gr.ChatInterface 기본 챗봇
- 스트리밍 응답 챗봇
- gr.Blocks 커스텀 레이아웃
- LCEL + Gradio 통합
"""

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ============================================================
# 1. 기본 ChatInterface 챗봇
# ============================================================

def create_basic_chatbot():
    """가장 간단한 Gradio 챗봇"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def chat(message: str, history: list) -> str:
        messages = [SystemMessage(content="당신은 친절한 AI 어시스턴트입니다.")]

        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                messages.append(AIMessage(content=ai_msg))

        messages.append(HumanMessage(content=message))
        response = llm.invoke(messages)
        return response.content

    return gr.ChatInterface(
        fn=chat,
        title="🤖 AI 어시스턴트",
        description="무엇이든 물어보세요!",
        examples=[
            "파이썬의 장점은?",
            "RAG가 무엇인지 설명해줘",
            "LangChain과 LlamaIndex의 차이점은?"
        ],
        theme=gr.themes.Soft()
    )


# ============================================================
# 2. 스트리밍 챗봇
# ============================================================

def create_streaming_chatbot():
    """스트리밍 응답 챗봇"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def chat_stream(message: str, history: list):
        messages = [SystemMessage(content="당신은 친절한 AI 어시스턴트입니다.")]

        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                messages.append(AIMessage(content=ai_msg))

        messages.append(HumanMessage(content=message))

        partial_response = ""
        for chunk in llm.stream(messages):
            partial_response += chunk.content
            yield partial_response  # 실시간 부분 응답 전송

    return gr.ChatInterface(
        fn=chat_stream,
        title="⚡ 스트리밍 AI 챗봇",
        description="실시간으로 응답을 받아보세요!",
    )


# ============================================================
# 3. gr.Blocks - 고급 커스텀 챗봇
# ============================================================

def create_advanced_chatbot():
    """gr.Blocks로 구성한 고급 챗봇"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    def chat(message: str, history: list, system_prompt: str,
             temperature: float, max_tokens: int):
        """시스템 프롬프트와 설정이 반영된 챗"""
        if not message.strip():
            return history, ""

        custom_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens
        )

        messages = []
        if system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt))
        else:
            messages.append(SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."))

        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                messages.append(AIMessage(content=ai_msg))

        messages.append(HumanMessage(content=message))

        partial = ""
        history.append([message, ""])

        for chunk in custom_llm.stream(messages):
            partial += chunk.content
            history[-1][1] = partial
            yield history, ""

    with gr.Blocks(title="🎛️ 고급 AI 챗봇", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎛️ 고급 AI 챗봇")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="대화",
                    bubble_full_width=False
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="메시지를 입력하세요...",
                        show_label=False,
                        scale=5
                    )
                    send_btn = gr.Button("전송 ▶", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")
                system_prompt = gr.Textbox(
                    label="시스템 프롬프트",
                    placeholder="AI의 역할을 정의하세요",
                    lines=4,
                    value="당신은 친절하고 유익한 AI 어시스턴트입니다."
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (창의성)"
                )
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=1000,
                    step=100,
                    label="최대 토큰 수"
                )
                gr.Markdown("### 🎭 페르소나 예시")
                with gr.Row():
                    gr.Button("👨‍⚕️ 의사").click(
                        lambda: "당신은 친절한 내과 의사입니다. 의학적 조언을 제공하되, 전문 진료를 권장하세요.",
                        outputs=system_prompt
                    )
                    gr.Button("👩‍💼 변호사").click(
                        lambda: "당신은 법률 전문가입니다. 법적 정보를 제공하되, 공식 법률 자문을 권장하세요.",
                        outputs=system_prompt
                    )
                clear_btn = gr.Button("🗑️ 대화 초기화", variant="stop")

        # 이벤트 연결
        inputs = [msg_input, chatbot, system_prompt, temperature_slider, max_tokens_slider]
        outputs = [chatbot, msg_input]

        msg_input.submit(chat, inputs, outputs)
        send_btn.click(chat, inputs, outputs)
        clear_btn.click(lambda: [], None, chatbot)

    return demo


# ============================================================
# 4. LCEL + Gradio 통합
# ============================================================

def create_lcel_chatbot():
    """LCEL 체인 + Gradio 통합"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 {persona}입니다. 해당 역할에 맞게 답변하세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    chain = prompt | llm | StrOutputParser()

    def chat(message: str, history: list, persona: str):
        chat_history = []
        for user_msg, ai_msg in history:
            chat_history.append(HumanMessage(content=user_msg))
            if ai_msg:
                chat_history.append(AIMessage(content=ai_msg))

        response = chain.invoke({
            "persona": persona,
            "chat_history": chat_history,
            "input": message
        })
        return response

    persona_options = [
        "영어 원어민 교사 (영어로만 답변)",
        "요리 전문가 쉐프",
        "스타트업 멘토",
        "역사 교수",
        "파이썬 전문 개발자"
    ]

    with gr.Blocks(title="페르소나 챗봇") as demo:
        gr.Markdown("# 🎭 페르소나 AI 챗봇")

        persona_selector = gr.Dropdown(
            choices=persona_options,
            value=persona_options[0],
            label="AI 역할 선택"
        )

        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="메시지 입력...", show_label=False)

        def respond(message, history, persona):
            response = chat(message, history, persona)
            history.append([message, response])
            return history, ""

        msg.submit(respond, [msg, chatbot, persona_selector], [chatbot, msg])
        gr.Button("초기화").click(lambda: [], None, chatbot)

    return demo


# ============================================================
# 5. 파일 업로드 챗봇
# ============================================================

def create_file_chatbot():
    """파일 업로드 지원 챗봇"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    def chat_with_file(message: str, history: list, uploaded_file):
        context = ""
        if uploaded_file is not None:
            try:
                with open(uploaded_file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                context = f"\n\n참고 문서:\n{content[:3000]}"
            except Exception:
                context = ""

        system = "당신은 친절한 AI 어시스턴트입니다."
        if context:
            system += f"{context}\n\n위 문서를 참고하여 답변하세요."

        messages = [SystemMessage(content=system)]
        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                messages.append(AIMessage(content=ai_msg))
        messages.append(HumanMessage(content=message))

        response = llm.invoke(messages)
        return response.content

    with gr.Blocks() as demo:
        gr.Markdown("# 📄 문서 기반 챗봇")
        file_upload = gr.File(
            label="텍스트 파일 업로드 (선택)",
            file_types=[".txt", ".md", ".py"]
        )
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="질문을 입력하세요...")

        def respond(message, history, file):
            response = chat_with_file(message, history, file)
            history.append([message, response])
            return history, ""

        msg.submit(respond, [msg, chatbot, file_upload], [chatbot, msg])

    return demo


# ============================================================
# 메인 실행 - 데모 앱 선택
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        choices=["basic", "streaming", "advanced", "lcel", "file"],
        default="streaming",
        help="실행할 데모 선택"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demos = {
        "basic": create_basic_chatbot,
        "streaming": create_streaming_chatbot,
        "advanced": create_advanced_chatbot,
        "lcel": create_lcel_chatbot,
        "file": create_file_chatbot,
    }

    demo_fn = demos[args.demo]
    demo = demo_fn()

    print(f"\n🚀 {args.demo} 데모를 시작합니다...")
    demo.launch(
        server_port=args.port,
        share=False,
        show_error=True
    )
