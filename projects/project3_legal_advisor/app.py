"""
Project 3: 법률 자문 AI - Gradio 앱
"""

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# 간단한 법률 Q&A (API 없이도 동작하는 데모)
LEGAL_FAQ = {
    "소멸시효": "민사 채권의 소멸시효는 원칙적으로 10년입니다. 상행위로 인한 채권은 5년입니다.",
    "전세": "전세계약 시 전입신고와 확정일자를 반드시 받으세요. 전세보증금 반환소송의 소멸시효는 10년입니다.",
    "해고": "부당해고는 노동위원회에 구제신청(3개월 이내)하거나 법원에 소를 제기할 수 있습니다.",
    "이혼": "협의이혼은 법원 확인이 필요하며, 재산분할 청구권은 이혼 후 2년 이내 행사해야 합니다.",
    "손해배상": "불법행위로 인한 손해배상청구권의 소멸시효는 손해와 가해자를 안 날로부터 3년입니다.",
}

try:
    from src.main_agent import LegalAdvisorAgent
    from src.config import Config

    config = Config()
    agent = LegalAdvisorAgent(config)

    # 샘플 법률 문서가 있으면 초기화
    sample_docs = "data/sample_legal_docs.txt"
    if os.path.exists(sample_docs):
        agent.initialize(sample_docs)
    else:
        agent.initialize()

    USE_AGENT = True
except Exception as e:
    print(f"에이전트 초기화 실패 (데모 모드로 실행): {e}")
    USE_AGENT = False


def get_demo_answer(query: str) -> str:
    """데모용 간단 답변"""
    for keyword, answer in LEGAL_FAQ.items():
        if keyword in query:
            return f"[참고 정보]\n{answer}\n\n*정확한 법률 자문은 변호사에게 문의하세요.*"
    return "해당 질문에 대한 기본 정보가 없습니다. 변호사 상담을 권장합니다.\n\n*이 시스템은 일반적인 법률 정보만 제공합니다.*"


def chat(message: str, history: list, case_type: str, rag_strategy: str, session_id: str) -> str:
    """챗봇 답변 생성"""
    if not message.strip():
        return ""

    # 케이스 타입 컨텍스트 추가
    full_query = f"[{case_type}] {message}"

    if USE_AGENT:
        strategy_map = {"자동 선택": None, "Adaptive RAG": "adaptive",
                       "Self-RAG": "self", "Corrective RAG": "corrective"}
        rag_type = strategy_map.get(rag_strategy)
        try:
            return agent.chat(full_query, session_id=session_id, rag_type=rag_type)
        except Exception as e:
            return f"[오류] {str(e)}\n\n{get_demo_answer(message)}"
    else:
        return get_demo_answer(message)


with gr.Blocks(title="⚖️ AI 법률 자문 시스템", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# ⚖️ AI 법률 자문 시스템

> **⚠️ 법적 면책 조항**: 이 시스템은 일반적인 법률 정보를 제공합니다.
> 구체적인 법적 조언이 필요한 경우 반드시 변호사와 상담하세요.
""")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, label="법률 상담")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="법률 질문을 입력하세요...",
                    show_label=False,
                    scale=5
                )
                send = gr.Button("전송", variant="primary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 상담 설정")
            case_type = gr.Dropdown(
                choices=["민사", "형사", "행정", "노동", "가족", "부동산", "계약", "기타"],
                value="민사",
                label="사건 분류"
            )
            rag_strategy = gr.Dropdown(
                choices=["자동 선택", "Adaptive RAG", "Self-RAG", "Corrective RAG"],
                value="자동 선택",
                label="RAG 전략"
            )
            session_id = gr.Textbox(
                label="세션 ID",
                value="legal_user_001"
            )

            gr.Markdown("### 📚 자주 묻는 질문")
            examples = [
                "소멸시효 3년과 10년의 차이는?",
                "전세 계약 시 주의사항은?",
                "부당해고 대응 방법은?",
                "이혼 시 재산분할 방법은?",
                "손해배상 청구 절차는?",
            ]
            for ex in examples:
                gr.Button(ex, size="sm").click(lambda x=ex: x, outputs=msg)

            clear = gr.Button("대화 초기화", variant="stop")

    def respond(message, history, case, strategy, sid):
        if not message.strip():
            return history, ""
        answer = chat(message, history, case, strategy, sid)
        history.append([message, answer])
        return history, ""

    msg.submit(respond, [msg, chatbot, case_type, rag_strategy, session_id], [chatbot, msg])
    send.click(respond, [msg, chatbot, case_type, rag_strategy, session_id], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
