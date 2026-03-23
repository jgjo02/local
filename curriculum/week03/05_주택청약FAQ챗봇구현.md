# 5차시: 주택청약 FAQ 챗봇 구현

## 학습 목표
- 3주간 학습한 내용을 통합하여 완전한 FAQ 챗봇을 구현한다
- 대화 이력 + RAG + 도메인 특화 프롬프트를 통합할 수 있다
- Gradio UI를 통해 사용 가능한 챗봇을 완성한다

---

## 1. 프로젝트 설계

```
주택청약 FAQ 챗봇 아키텍처:

사용자 질문
    │
    ▼
┌─────────────────────────────────────────────┐
│  Gradio ChatInterface                        │
│  - 대화 이력 표시                             │
│  - 스트리밍 응답                              │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │    RunnableWithMessageHistory     │
    │    (대화 이력 관리)                │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │    RAG Chain (LCEL)               │
    │    Retriever + Prompt + LLM       │
    └──────────────┬────────────────────┘
                   │
    ┌──────────────▼────────────────────┐
    │    FAISS VectorStore              │
    │    (청약 FAQ 문서)                 │
    └───────────────────────────────────┘
```

---

## 2. 시스템 프롬프트 설계

```python
SYSTEM_PROMPT = """당신은 주택청약 전문 상담사 'AI 청약도우미'입니다.

역할:
- 주택청약 관련 질문에 정확하고 친절하게 답변합니다
- 복잡한 청약 절차를 쉽게 설명합니다
- 개인 상황에 맞는 맞춤 안내를 제공합니다

답변 원칙:
1. 제공된 FAQ 문서를 기반으로 답변하세요
2. 문서에 없는 내용은 "정확한 정보를 위해 청약홈이나 담당 기관에 문의하세요"라고 안내
3. 법적 조언이 필요한 경우 전문가 상담을 권유하세요
4. 답변은 간결하고 실용적으로 작성하세요

주의사항:
- 청약 당첨을 보장하는 발언 금지
- 투자 조언 금지
- 최신 정책 변경 가능성을 항상 언급"""
```

---

## 3. 완전한 구현

```python
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# FAQ 데이터 로드 및 인덱스 구축
def build_faq_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # (실제 구현에서는 FAQ 파일 로드)
    docs = load_faq_documents()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 챗봇 체인 구성
def create_chatbot_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\n참고 문서:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "history": lambda x: x.get("history", [])
        }
        | prompt | llm | StrOutputParser()
    )
    return chain

# Gradio 앱
def create_gradio_app(chain):
    store = {}

    def get_history(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        chain, get_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    def chat(message, history, session_id):
        config = {"configurable": {"session_id": session_id}}
        response = ""
        for chunk in chain_with_memory.stream(
            {"question": message}, config=config
        ):
            response += chunk
            yield response

    with gr.Blocks(title="AI 청약도우미") as demo:
        gr.Markdown("# 🏠 AI 청약도우미\n주택청약에 관한 모든 것을 알려드립니다!")
        session_id = gr.State("default_session")
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="청약 관련 질문을 입력하세요...")

        def respond(message, history, sid):
            partial = ""
            for chunk in chat(message, history, sid):
                partial = chunk
                yield history + [[message, partial]], ""
            yield history + [[message, partial]], ""

        msg.submit(respond, [msg, chatbot, session_id], [chatbot, msg])
        gr.Examples(
            examples=["청약 1순위 조건은?", "특별공급 종류를 알려주세요"],
            inputs=msg
        )

    return demo
```

---

## 4. 테스트 케이스

```python
test_questions = [
    "청약 1순위 되려면 어떻게 해야 해요?",
    "신혼부부 특별공급 자격은?",
    "청약통장 얼마나 넣어야 해요?",
    "당첨됐는데 계약 포기하면 어떻게 돼요?",
    "분양가 상한제가 뭔가요?",
]
```

---

## 💡 핵심 포인트
- 도메인 특화 시스템 프롬프트가 일반 프롬프트보다 훨씬 나은 결과
- 대화 이력 + RAG 결합으로 맥락 있는 답변 제공
- 면책 조항을 프롬프트에 명시하여 법적 리스크 관리

## ❓ 차시별 질문
1. 시스템 프롬프트에서 면책 조항을 제거하면 어떻게 될까요?
2. RAG 없이 메모리만 있는 챗봇과 성능을 비교해보세요
3. k=3과 k=5 검색 결과 품질 차이는?
