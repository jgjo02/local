# 5차시: LCEL & Gradio 챗봇 구현

## 학습 목표
- LCEL의 고급 기능(RunnablePassthrough, RunnableParallel)을 사용할 수 있다
- Gradio로 인터랙티브 챗봇 UI를 구현할 수 있다
- 스트리밍 응답을 Gradio와 연동할 수 있다
- gr.State를 활용하여 대화 이력을 관리할 수 있다

---

## 1. LCEL 고급 기능

### 1.1 RunnablePassthrough

입력을 그대로 다음 단계로 전달:

```python
from langchain_core.runnables import RunnablePassthrough

# 검색 결과와 원본 질문을 모두 다음 단계로 전달
chain = (
    {
        "context": retriever,           # 검색 수행
        "question": RunnablePassthrough()  # 원본 질문 그대로 전달
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### 1.2 RunnableParallel

여러 작업을 병렬 실행:

```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
    keywords=keyword_chain
)

# 3개 작업이 동시에 실행됨
result = parallel_chain.invoke("분석할 텍스트...")
# result = {"summary": ..., "translation": ..., "keywords": ...}
```

### 1.3 RunnableLambda

파이썬 함수를 Runnable로 래핑:

```python
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

format_chain = RunnableLambda(format_docs)
```

### 1.4 .bind() 메서드

실행 시 파라미터를 미리 고정:

```python
# 특정 함수 호출을 강제하는 LLM
llm_with_json = llm.bind(response_format={"type": "json_object"})

# 특정 도구만 사용하도록 제한
llm_with_tools = llm.bind_tools([search_tool, calculator_tool])
```

---

## 2. LCEL 완전한 RAG 체인 구성

```
사용자 질문
    │
    ▼
┌─────────────────────────────────────────┐
│  RunnableParallel                        │
│  ┌─────────────────┐  ┌───────────────┐ │
│  │ context         │  │ question      │ │
│  │ (retriever)     │  │ (passthrough) │ │
│  └─────────────────┘  └───────────────┘ │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ChatPromptTemplate                      │
│  "다음 컨텍스트를 바탕으로 {question}에  │
│   답하세요: {context}"                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────┐    ┌───────────────┐
│   LLM   │ ──▶│ StrOutputParser│
└─────────┘    └───────────────┘
```

---

## 3. Gradio 기초

### 3.1 설치 및 빠른 시작

```bash
pip install gradio
```

```python
import gradio as gr

def greet(name):
    return f"안녕하세요, {name}님!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="이름"),
    outputs=gr.Textbox(label="인사")
)

demo.launch()
```

### 3.2 gr.ChatInterface - 가장 간단한 챗봇

```python
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini")

def chat(message, history):
    # history: [[user_msg, ai_msg], ...]
    messages = []
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return response.content

demo = gr.ChatInterface(
    fn=chat,
    title="AI 챗봇",
    description="무엇이든 물어보세요!",
    examples=["파이썬이란?", "LangChain 소개"]
)

demo.launch(share=False, server_port=7860)
```

### 3.3 스트리밍 응답

```python
def chat_stream(message, history):
    messages = []
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        if ai_msg:
            messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=message))

    partial_message = ""
    for chunk in llm.stream(messages):
        partial_message += chunk.content
        yield partial_message  # 부분 응답을 실시간 전송

demo = gr.ChatInterface(
    fn=chat_stream,
    title="스트리밍 AI 챗봇"
)
```

### 3.4 gr.Blocks - 커스텀 레이아웃

```python
import gradio as gr

with gr.Blocks(title="AI 어시스턴트") as demo:
    gr.Markdown("# 🤖 AI 어시스턴트")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="메시지 입력...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("전송", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### 설정")
            temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
            model_selector = gr.Dropdown(
                ["gpt-4o-mini", "gpt-4o"],
                value="gpt-4o-mini",
                label="모델 선택"
            )
            clear_btn = gr.Button("대화 초기화")

    # 이벤트 핸들러
    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
    send_btn.click(chat_fn, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

demo.launch()
```

### 3.5 gr.State - 세션 상태 관리

```python
def update_history(message, history, state):
    # state는 사용자별로 독립적으로 유지됨
    if state is None:
        state = {"count": 0, "context": []}

    state["count"] += 1
    response = f"[{state['count']}번째 답변] {process(message)}"
    history.append([message, response])

    return history, state

with gr.Blocks() as demo:
    session_state = gr.State(None)  # 사용자별 상태
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    msg.submit(
        update_history,
        [msg, chatbot, session_state],
        [chatbot, session_state]
    )
```

---

## 4. 파일 업로드 챗봇

```python
def process_file_chat(message, history, file):
    if file is not None:
        # 파일 처리 후 컨텍스트로 활용
        with open(file.name, 'r') as f:
            content = f.read()
        response = llm.invoke(
            f"다음 문서를 참고하여 답하세요:\n{content}\n\n질문: {message}"
        )
    else:
        response = llm.invoke(message)
    return response.content

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        file_upload = gr.File(label="파일 업로드 (선택)")
        msg = gr.Textbox(label="질문")
    msg.submit(process_file_chat, [msg, chatbot, file_upload], chatbot)
```

---

## 💡 핵심 포인트
- `RunnablePassthrough`는 입력을 변경 없이 전달 (RAG에서 question 전달 시 필수)
- `RunnableParallel`로 여러 체인을 동시 실행하여 성능 향상
- Gradio `ChatInterface`는 기본 챗봇에, `Blocks`는 복잡한 UI에 사용
- 스트리밍에서 `yield`를 사용하면 실시간으로 응답이 표시됨

## ⚠️ 주의사항
- Gradio 스트리밍 함수는 반드시 `yield`를 사용해야 함 (`return` 불가)
- `gr.State`는 서버 메모리에 저장되므로 대규모 서비스엔 Redis 등 외부 저장소 사용
- `demo.launch(share=True)`는 임시 공개 URL 생성 (보안 주의)

## 🔍 심화학습
- [Gradio 공식 문서](https://www.gradio.app/docs)
- [LCEL Cookbook](https://python.langchain.com/docs/expression_language/cookbook/)

## ❓ 차시별 질문
1. `RunnableParallel`로 요약과 번역을 동시에 실행해보세요
2. Gradio Blocks로 사이드바에 설정 패널이 있는 챗봇을 만들어보세요
3. 파일 업로드 후 문서 기반 Q&A를 구현해보세요
