"""
3주차 4차시: 메모리 & 대화 이력 관리
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 세션 저장소
store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def demo_basic_memory():
    """기본 메모리 챗봇"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "demo_session"}}

    print("=== 기본 메모리 챗봇 ===")
    turns = [
        "내 이름은 홍길동이야",
        "나는 30살이고 개발자야",
        "내 이름이 뭐야?",
        "내 직업은?",
    ]

    for user_msg in turns:
        response = chain_with_history.invoke({"input": user_msg}, config=config)
        print(f"User: {user_msg}")
        print(f"AI:   {response}\n")


def demo_multi_session():
    """멀티 사용자 세션"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain_with_history = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    print("\n=== 멀티 사용자 세션 ===")

    # 사용자 1
    c1 = {"configurable": {"session_id": "user_001"}}
    chain_with_history.invoke({"input": "내 이름은 김철수입니다"}, config=c1)
    r1 = chain_with_history.invoke({"input": "제 이름을 알고 있나요?"}, config=c1)
    print(f"[User 001] 제 이름을 알고 있나요?")
    print(f"AI: {r1}\n")

    # 사용자 2 (독립적 세션)
    c2 = {"configurable": {"session_id": "user_002"}}
    chain_with_history.invoke({"input": "내 이름은 이영희입니다"}, config=c2)
    r2 = chain_with_history.invoke({"input": "제 이름을 알고 있나요?"}, config=c2)
    print(f"[User 002] 제 이름을 알고 있나요?")
    print(f"AI: {r2}\n")


def demo_token_trimming():
    """토큰 수 제한 메모리"""
    trimmer = trim_messages(
        max_tokens=500,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = trimmer | prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id": "trim_session"}}

    print("=== 토큰 제한 메모리 ===")
    # 여러 번 대화
    messages = [
        "파이썬에 대해 알려줘",
        "머신러닝은?",
        "딥러닝은?",
        "자연어 처리는?",
        "RAG가 뭔지 설명해줘",
    ]

    for msg in messages:
        response = chain_with_history.invoke({"input": msg}, config=config)
        history = get_session_history("trim_session").messages
        print(f"대화 {len(history)//2}번째 - 이력 메시지 수: {len(history)}")

    print("토큰 제한으로 오래된 이력이 삭제되어 최근 대화만 유지됩니다")


def interactive_chat():
    """대화형 챗봇"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain_with_history = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    session_id = "interactive"
    config = {"configurable": {"session_id": session_id}}

    print("\n=== 대화형 챗봇 (종료: 'q') ===\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        if not user_input:
            continue

        response = chain_with_history.invoke({"input": user_input}, config=config)
        print(f"AI: {response}\n")

        # 이력 크기 표시
        history = get_session_history(session_id).messages
        print(f"[대화 이력: {len(history)}개 메시지]\n")


if __name__ == "__main__":
    import sys
    if "--interactive" in sys.argv:
        interactive_chat()
    else:
        demo_basic_memory()
        demo_multi_session()
        demo_token_trimming()
