"""
4차시: LangChain 기초 실습
- ChatModel 기본 사용
- 메시지 타입 활용
- PromptTemplate / ChatPromptTemplate
- LCEL 파이프라인 구성
- invoke / batch / stream 메서드
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
)
from langchain_core.output_parsers import (
    StrOutputParser, JsonOutputParser
)
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda
)
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


# ============================================================
# 1. ChatModel 기본 사용
# ============================================================

def demo_chat_model():
    """ChatModel 기본 사용법"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500
    )

    # 단일 메시지
    response = llm.invoke("파이썬의 특징을 3가지 말해줘")
    print(f"타입: {type(response)}")
    print(f"내용: {response.content}")
    print(f"모델: {response.response_metadata.get('model_name', 'unknown')}")

    return response


# ============================================================
# 2. 메시지 타입 활용
# ============================================================

def demo_message_types():
    """다양한 메시지 타입 사용"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # SystemMessage + HumanMessage 조합
    messages = [
        SystemMessage(content="당신은 파이썬 전문가입니다. 간결하게 답변하세요."),
        HumanMessage(content="리스트 컴프리헨션이란?"),
    ]

    response1 = llm.invoke(messages)
    print(f"전문가 답변: {response1.content[:200]}")

    # 대화 이력 포함
    conversation = [
        SystemMessage(content="당신은 친절한 AI 튜터입니다."),
        HumanMessage(content="파이썬이란?"),
        AIMessage(content="파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다."),
        HumanMessage(content="그렇다면 장점은?"),  # 이전 맥락을 알고 있음
    ]

    response2 = llm.invoke(conversation)
    print(f"\n연속 대화 답변: {response2.content[:200]}")


# ============================================================
# 3. PromptTemplate 활용
# ============================================================

def demo_prompt_templates():
    """다양한 프롬프트 템플릿 사용"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 1. ChatPromptTemplate (최신 방식)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 {domain} 전문가입니다."),
        ("human", "{question}")
    ])

    chain = chat_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "domain": "블록체인",
        "question": "스마트 컨트랙트란?"
    })
    print(f"전문가 답변: {result[:200]}")

    # 2. MessagesPlaceholder - 동적 대화 이력
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", "친절한 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 이전 대화가 있는 경우
    history_chain = history_prompt | llm | StrOutputParser()
    result2 = history_chain.invoke({
        "history": [
            HumanMessage(content="내 이름은 김철수야"),
            AIMessage(content="안녕하세요, 김철수님!")
        ],
        "input": "내 이름이 뭔지 알아?"
    })
    print(f"\n이력 포함 답변: {result2}")


# ============================================================
# 4. LCEL 파이프라인
# ============================================================

def demo_lcel_pipeline():
    """LCEL 파이프라인 구성"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 기본 파이프라인
    prompt = ChatPromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해줘")
    chain = prompt | llm | StrOutputParser()

    # invoke
    result = chain.invoke({"topic": "양자 컴퓨팅"})
    print(f"invoke 결과: {result}")

    # batch - 여러 입력 병렬 처리
    topics = [
        {"topic": "머신러닝"},
        {"topic": "딥러닝"},
        {"topic": "강화학습"},
    ]
    results = chain.batch(topics)
    print("\nbatch 결과:")
    for topic, result in zip(topics, results):
        print(f"  {topic['topic']}: {result[:80]}")

    # stream
    print("\nstream 결과: ", end="")
    for token in chain.stream({"topic": "RAG"}):
        print(token, end="", flush=True)
    print()


# ============================================================
# 5. RunnablePassthrough & RunnableParallel
# ============================================================

def demo_runnable_utilities():
    """RunnablePassthrough, RunnableParallel 활용"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RunnablePassthrough: 입력을 그대로 다음 단계로 전달
    expand_prompt = ChatPromptTemplate.from_template(
        "다음 주제를 자세히 설명해줘: {topic}"
    )

    chain = (
        {
            "topic": RunnablePassthrough(),
            "echo": RunnablePassthrough()  # 입력을 그대로 유지
        }
        | RunnableLambda(lambda x: {"topic": x["topic"]})
        | expand_prompt
        | llm
        | StrOutputParser()
    )

    # RunnableParallel: 여러 작업 동시 실행
    summarize_prompt = ChatPromptTemplate.from_template(
        "{text}를 한 문장으로 요약해줘"
    )
    translate_prompt = ChatPromptTemplate.from_template(
        "{text}를 영어로 번역해줘"
    )

    parallel_chain = RunnableParallel(
        summary=(summarize_prompt | llm | StrOutputParser()),
        translation=(translate_prompt | llm | StrOutputParser()),
    )

    results = parallel_chain.invoke({"text": "인공지능은 인간의 지능을 모방하는 기술입니다."})
    print(f"요약: {results['summary']}")
    print(f"번역: {results['translation']}")


# ============================================================
# 6. JSON 출력 파서
# ============================================================

class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자명")
    year: int = Field(description="출판 연도")
    genre: str = Field(description="장르")
    summary: str = Field(description="한 문장 요약")


def demo_json_output():
    """JSON 형식 출력"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = JsonOutputParser(pydantic_object=BookInfo)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "책 정보를 JSON 형식으로 반환하세요.\n{format_instructions}"),
        ("human", "{book_title}에 대한 정보를 제공해주세요.")
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "book_title": "해리 포터와 마법사의 돌",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"파싱된 결과: {result}")
    print(f"타입: {type(result)}")


# ============================================================
# 7. .bind() 메서드
# ============================================================

def demo_bind_method():
    """bind로 파라미터 고정"""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # temperature 고정
    creative_llm = llm.bind(temperature=1.5)
    precise_llm = llm.bind(temperature=0.0)

    prompt = "미래의 교통수단에 대해 예측해줘 (100자 이내)"

    creative_response = creative_llm.invoke(prompt)
    precise_response = precise_llm.invoke(prompt)

    print(f"창의적 답변 (temp=1.5): {creative_response.content}")
    print(f"정확한 답변 (temp=0.0): {precise_response.content}")


# ============================================================
# 8. 에러 처리 및 폴백
# ============================================================

def demo_fallback():
    """폴백(Fallback) 체인"""
    primary_llm = ChatOpenAI(model="gpt-4o")  # 비쌈
    fallback_llm = ChatOpenAI(model="gpt-4o-mini")  # 저렴한 폴백

    # 기본 모델이 실패하면 폴백 모델 사용
    llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

    try:
        response = llm_with_fallback.invoke("안녕하세요!")
        print(f"응답: {response.content}")
    except Exception as e:
        print(f"모든 모델 실패: {e}")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("1. ChatModel 기본 사용")
    print("=" * 50)
    demo_chat_model()

    print("\n" + "=" * 50)
    print("2. 메시지 타입 활용")
    print("=" * 50)
    demo_message_types()

    print("\n" + "=" * 50)
    print("3. PromptTemplate")
    print("=" * 50)
    demo_prompt_templates()

    print("\n" + "=" * 50)
    print("4. LCEL 파이프라인")
    print("=" * 50)
    demo_lcel_pipeline()

    print("\n" + "=" * 50)
    print("5. RunnableParallel")
    print("=" * 50)
    demo_runnable_utilities()

    print("\n" + "=" * 50)
    print("6. JSON 출력 파서")
    print("=" * 50)
    demo_json_output()
