"""Week 3 실습 2: Zero-shot 및 Few-shot 학습"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── 1. Zero-shot 프롬프트 ─────────────────────────────────────────────────────

def zero_shot_classification(text: str) -> str:
    """Zero-shot 감성 분류"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "텍스트의 감성을 분류하세요: 긍정/부정/중립"),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def zero_shot_extraction(text: str) -> str:
    """Zero-shot 정보 추출"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """텍스트에서 다음 정보를 JSON으로 추출하세요:
- 날짜
- 금액
- 장소
없으면 null로 표시하세요."""),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


# ── 2. Few-shot 프롬프트 ─────────────────────────────────────────────────────

# ETF 투자 성향 분류 예시
INVESTMENT_EXAMPLES = [
    {
        "input": "은퇴 후 안정적인 수입이 필요해요. 원금을 잃기 싫습니다.",
        "output": "투자 성향: 안정형\n추천 유형: 채권형 ETF, MMF\n위험 허용도: 낮음",
    },
    {
        "input": "30대 직장인인데 10년 후 목돈을 만들고 싶어요.",
        "output": "투자 성향: 성장형\n추천 유형: 주식형 ETF (국내/해외 혼합)\n위험 허용도: 중간",
    },
    {
        "input": "AI와 반도체 섹터가 미래라고 생각해요. 변동성은 감수할 수 있어요.",
        "output": "투자 성향: 공격형\n추천 유형: 섹터 ETF (AI/반도체)\n위험 허용도: 높음",
    },
]

def few_shot_investment_classifier(user_input: str) -> str:
    """Few-shot 투자 성향 분류기"""
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=INVESTMENT_EXAMPLES,
    )

    full_prompt = ChatPromptTemplate.from_messages([
        ("system", "투자자의 말을 듣고 투자 성향을 분류하세요."),
        few_shot_prompt,
        ("human", "{user_input}"),
    ])

    chain = full_prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": user_input})


# ── 3. 동적 예시 선택 ────────────────────────────────────────────────────────

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_dynamic_few_shot():
    """유사도 기반 동적 예시 선택"""
    examples = [
        {"query": "임대차 계약 해지 방법", "category": "민사법", "difficulty": "중간"},
        {"query": "부당해고 대응 방법", "category": "노동법", "difficulty": "중간"},
        {"query": "교통사고 합의 절차", "category": "민사법", "difficulty": "낮음"},
        {"query": "형사 고소 방법", "category": "형사법", "difficulty": "높음"},
        {"query": "임금 체불 신고", "category": "노동법", "difficulty": "낮음"},
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=FAISS,
        k=2,
    )
    return selector


# ── 4. Chain-of-Thought Few-shot ──────────────────────────────────────────────

COT_EXAMPLES = [
    {
        "question": "월 50만원씩 5년간 투자하면 7% 연수익률로 얼마가 될까요?",
        "reasoning": """단계적 계산:
1. 월 납입액: 500,000원
2. 투자 기간: 5년 = 60개월
3. 월 이자율: 7% ÷ 12 = 0.583%
4. 적립식 미래가치 공식: PMT × [(1+r)^n - 1] / r
5. = 500,000 × [(1.00583)^60 - 1] / 0.00583
6. ≈ 500,000 × 71.59
7. ≈ 35,795,000원""",
        "answer": "약 3,580만원이 됩니다. (원금 3,000만원 + 수익 580만원)",
    },
]

def few_shot_cot_calculator(question: str) -> str:
    """CoT Few-shot 계산기"""
    examples_text = "\n\n".join([
        f"질문: {ex['question']}\n추론 과정:\n{ex['reasoning']}\n답변: {ex['answer']}"
        for ex in COT_EXAMPLES
    ])

    prompt = f"""다음 예시처럼 단계적으로 추론하여 답변하세요.

{examples_text}

질문: {question}
추론 과정:"""

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        exit()

    print("=" * 60)
    print("1. Zero-shot 감성 분류")
    print("=" * 60)
    texts = [
        "이 ETF는 수익률이 정말 훌륭하네요!",
        "수수료가 너무 높아서 실망했습니다.",
        "평범한 성과를 보이고 있습니다.",
    ]
    for t in texts:
        result = zero_shot_classification(t)
        print(f"텍스트: {t}\n분류: {result}\n")

    print("\n" + "=" * 60)
    print("2. Few-shot 투자 성향 분류")
    print("=" * 60)
    test_inputs = [
        "60대 은퇴 예정자입니다. 안전하게 자산을 지키고 싶어요.",
        "20대 사회초년생, 공격적으로 투자하고 싶어요!",
    ]
    for inp in test_inputs:
        result = few_shot_investment_classifier(inp)
        print(f"입력: {inp}\n결과:\n{result}\n")

    print("\n" + "=" * 60)
    print("3. CoT Few-shot 계산")
    print("=" * 60)
    question = "월 30만원씩 3년간 5% 연수익으로 투자하면?"
    result = few_shot_cot_calculator(question)
    print(f"질문: {question}\n{result}")
