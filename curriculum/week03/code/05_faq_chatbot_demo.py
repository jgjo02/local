"""
3주차 5차시: 주택청약 FAQ 챗봇 완성 구현
- RAG + 메모리 + Gradio 통합
"""

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

SYSTEM_PROMPT = """당신은 주택청약 전문 상담사 'AI 청약도우미'입니다.

역할:
- 주택청약 관련 질문에 정확하고 친절하게 답변합니다
- 복잡한 청약 절차를 이해하기 쉽게 설명합니다

답변 원칙:
1. 제공된 참고 문서를 기반으로 답변하세요
2. 문서에 없는 내용은 "정확한 정보를 위해 청약홈(applyhome.co.kr)이나 담당 기관에 문의하세요"라고 안내
3. 법적 조언이나 투자 조언은 제공하지 않습니다
4. 정책은 변경될 수 있으므로 최신 정보를 확인하도록 안내하세요

참고 문서:
{context}"""

FAQ_DATA = """
Q: 주택청약 1순위 조건은?
A: 투기과열지구는 청약통장 가입 후 2년 이상, 수도권은 1년 이상, 기타 지역은 6개월 이상 납입해야 합니다. 세대주로서 무주택자이거나 1주택자여야 합니다.

Q: 특별공급 종류는?
A: 신혼부부(혼인 7년 이내), 생애최초(처음 주택 구입), 다자녀(미성년 자녀 3명 이상), 노부모부양(65세 이상 직계존속 3년 이상 부양), 장애인, 국가유공자, 기관추천 특별공급이 있습니다.

Q: 가점제 계산 방법은?
A: 무주택기간(최대 32점), 부양가족수(최대 35점), 청약통장 가입기간(최대 17점)으로 최대 84점입니다. 무주택기간은 세대구성원 모두 무주택인 기간, 부양가족은 동일 주민등록에 등재된 직계가족입니다.

Q: 청약통장 납입 금액과 횟수?
A: 국민주택은 납입 횟수(월 1회, 최소 2만원), 민간분양은 납입 총액(1,500만 원 이상 권장)이 중요합니다. 매달 10만원씩 납입하면 12년 후 1,440만 원이 됩니다.

Q: 청약 당첨 후 포기하면?
A: 분양가 상한제 적용 주택이나 투기과열지구 당첨 시 최대 10년간 재청약이 제한됩니다. 일반 민간분양은 별도 제한 없는 경우도 있습니다.

Q: 청약 신청은 어디서?
A: 청약홈(applyhome.co.kr) 또는 금융결제원 앱에서 신청합니다. 은행 영업점 방문도 가능합니다.

Q: 분양가 상한제란?
A: 공공택지 및 규제 지역 민간택지 아파트의 분양 가격을 제한하는 제도입니다. 기본형 건축비와 택지비를 합산하여 산정합니다.

Q: 전매 제한이란?
A: 분양권이나 입주권을 다른 사람에게 팔 수 없는 기간입니다. 투기과열지구는 소유권 이전 등기 시까지 전매 금지입니다.

Q: LH 청약은?
A: 한국토지주택공사(LH)에서 공급하는 공공임대, 공공분양 주택입니다. 소득 기준이 있으며 일반 청약과 별도 신청합니다.

Q: 청년 특별공급이 있나요?
A: 공공분양에 청년 특별공급이 있습니다. 만 19~39세 무주택 청년으로 소득 기준을 충족해야 합니다.
"""


def build_rag_system():
    """RAG 시스템 구축"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docs = []
    for i, qa in enumerate(FAQ_DATA.strip().split("\n\n")):
        if qa.strip():
            docs.append(Document(
                page_content=qa,
                metadata={"source": "faq", "id": i}
            ))

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def create_rag_chain(retriever):
    """RAG + 메모리 체인 구성"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    def get_context(inputs):
        docs = retriever.invoke(inputs["question"])
        return "\n\n".join(doc.page_content for doc in docs)

    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    chain = (
        RunnableParallel({
            "context": lambda x: get_context(x),
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", [])
        })
        | prompt | llm | StrOutputParser()
    )

    return chain


class FAQChatbot:
    def __init__(self):
        self.retriever = build_rag_system()
        self.chain = create_rag_chain(self.retriever)
        self.store = {}

    def get_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def build_chain_with_memory(self):
        return RunnableWithMessageHistory(
            self.chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    def chat(self, question: str, session_id: str = "default") -> str:
        chain = self.build_chain_with_memory()
        config = {"configurable": {"session_id": session_id}}
        return chain.invoke({"question": question}, config=config)

    def stream_chat(self, question: str, session_id: str = "default"):
        chain = self.build_chain_with_memory()
        config = {"configurable": {"session_id": session_id}}
        for chunk in chain.stream({"question": question}, config=config):
            yield chunk


def create_gradio_app(chatbot: FAQChatbot):
    """Gradio 앱 생성"""

    def respond(message, history, session_id):
        if not session_id.strip():
            session_id = "default"

        partial = ""
        for chunk in chatbot.stream_chat(message, session_id):
            partial += chunk
            yield history + [[message, partial]], ""

    with gr.Blocks(title="AI 청약도우미", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🏠 AI 청약도우미

주택청약에 관한 궁금한 점을 질문해보세요!
> ⚠️ 이 챗봇은 일반적인 청약 정보를 제공합니다. 정확한 정보는 청약홈(applyhome.co.kr)을 확인하세요.
""")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot_ui = gr.Chatbot(height=450, label="대화")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="청약 관련 질문을 입력하세요...",
                        show_label=False,
                        scale=5
                    )
                    send = gr.Button("전송", variant="primary", scale=1)

            with gr.Column(scale=1):
                session_id = gr.Textbox(
                    label="세션 ID",
                    value="user_001",
                    placeholder="사용자 ID 입력"
                )
                gr.Markdown("### 📚 자주 묻는 질문")
                examples = [
                    "청약 1순위 조건은?",
                    "특별공급 종류를 알려주세요",
                    "가점제 점수 계산 방법은?",
                    "청약 당첨 후 포기하면?",
                    "분양가 상한제란?",
                ]
                for ex in examples:
                    gr.Button(ex, size="sm").click(
                        lambda x=ex: x, outputs=msg
                    )
                clear = gr.Button("대화 초기화", variant="stop")

        msg.submit(respond, [msg, chatbot_ui, session_id], [chatbot_ui, msg])
        send.click(respond, [msg, chatbot_ui, session_id], [chatbot_ui, msg])
        clear.click(lambda: [], None, chatbot_ui)

    return demo


if __name__ == "__main__":
    print("주택청약 FAQ 챗봇 시작...")
    chatbot = FAQChatbot()

    import sys
    if "--cli" in sys.argv:
        print("CLI 모드 (종료: q)\n")
        while True:
            q = input("질문: ").strip()
            if q.lower() in ['q', 'quit']:
                break
            if q:
                answer = chatbot.chat(q)
                print(f"답변: {answer}\n")
    else:
        app = create_gradio_app(chatbot)
        app.launch(server_port=7860, share=False)
