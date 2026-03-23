"""
3주차 1차시: 프롬프트 엔지니어링 & 템플릿 실습
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def demo_bad_vs_good_prompts():
    """나쁜 프롬프트 vs 좋은 프롬프트 비교"""
    bad_prompt = ChatPromptTemplate.from_template("설명해줘: {topic}")
    good_prompt = ChatPromptTemplate.from_template("""
{topic}에 대해 다음 형식으로 설명해주세요:

1. 정의 (1-2문장)
2. 핵심 특징 (3가지, 불릿 포인트)
3. 실제 사용 예시 (코드 포함)
4. 언제 사용하면 좋은가

초보자도 이해할 수 있게 작성하세요.""")

    chain_bad = bad_prompt | llm | StrOutputParser()
    chain_good = good_prompt | llm | StrOutputParser()

    topic = "파이썬 데코레이터"
    print("나쁜 프롬프트:")
    print(chain_bad.invoke({"topic": topic})[:300])
    print("\n좋은 프롬프트:")
    print(chain_good.invoke({"topic": topic})[:500])


class AnalysisResult(BaseModel):
    summary: str = Field(description="2문장 요약")
    keywords: List[str] = Field(description="핵심 키워드 3개")
    difficulty: str = Field(description="초급/중급/고급")
    target_audience: str = Field(description="추천 대상")


def demo_json_output():
    """JSON 형식 출력"""
    parser = JsonOutputParser(pydantic_object=AnalysisResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "텍스트를 분석하여 JSON으로 반환하세요.\n{format_instructions}"),
        ("human", "분석할 텍스트:\n{text}")
    ])

    chain = prompt | llm | parser

    text = """RAG(Retrieval-Augmented Generation)는 외부 지식 베이스에서
    관련 문서를 검색하여 LLM의 답변 생성을 보강하는 기법입니다.
    실시간 정보 반영과 환각 감소에 효과적입니다."""

    result = chain.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions()
    })
    print(f"JSON 분석 결과: {result}")


def demo_role_assignment():
    """역할 부여 프롬프트"""
    roles = [
        ("초등학생 선생님", "초등학생도 이해할 수 있게"),
        ("기술 블로거", "전문적이고 상세하게"),
        ("마케터", "흥미롭고 설득력있게"),
    ]

    for role, style in roles:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"당신은 {role}입니다. {style} 설명하세요."),
            ("human", "머신러닝이란?")
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({})
        print(f"\n[{role}]\n{result[:200]}\n")


def demo_few_shot():
    """Few-Shot 프롬프팅"""
    examples = [
        {"input": "좋아요!", "output": "긍정"},
        {"input": "정말 실망이에요", "output": "부정"},
        {"input": "그냥 그래요", "output": "중립"},
    ]

    few_shot = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "텍스트 감성을 긍정/부정/중립으로 분류하세요."),
        few_shot,
        ("human", "{text}")
    ])

    chain = prompt | llm | StrOutputParser()

    tests = ["완전 최고야!", "돈 낭비했네", "보통이에요", "기대 이상이었어요"]
    for test in tests:
        result = chain.invoke({"text": test})
        print(f"'{test}' → {result}")


if __name__ == "__main__":
    print("3주차 1차시: 프롬프트 엔지니어링\n")
    print("=== JSON 출력 파서 ===")
    demo_json_output()
    print("\n=== 역할 부여 ===")
    demo_role_assignment()
    print("\n=== Few-Shot ===")
    demo_few_shot()
