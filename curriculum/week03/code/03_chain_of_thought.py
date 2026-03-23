"""Week 3 실습 3: Chain-of-Thought 프롬프팅"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_precise = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ── 1. 기본 CoT ────────────────────────────────────────────────────────────────

def basic_cot(question: str) -> dict:
    """기본 Chain-of-Thought"""
    # 직접 답변 (CoT 없음)
    direct_response = llm.invoke([HumanMessage(content=question)])

    # CoT 프롬프트
    cot_prompt = f"{question}\n\n단계별로 생각해봅시다:"
    cot_response = llm.invoke([HumanMessage(content=cot_prompt)])

    return {
        "direct": direct_response.content,
        "cot": cot_response.content,
    }


# ── 2. ETF 투자 분석 CoT ──────────────────────────────────────────────────────

ETF_ANALYSIS_COT = """ETF 투자 분석을 단계별로 진행해주세요.

분석 대상: {etf_info}
투자자 프로파일: {investor_profile}

단계별 분석:
1. ETF 특성 파악
   - 카테고리, 위험도, 수익률 확인

2. 투자자 적합성 평가
   - 위험 허용도와 ETF 위험도 비교
   - 투자 기간과 수익률 패턴 비교

3. 장단점 분석
   - 장점 3가지
   - 단점/위험 3가지

4. 최종 추천 의견
   - 적합/부적합 판정
   - 이유 (2-3문장)
"""

def analyze_etf_with_cot(etf_info: dict, investor_profile: dict) -> str:
    """CoT로 ETF 적합성 분석"""
    etf_str = f"""
ETF명: {etf_info['name']}
카테고리: {etf_info['category']}
위험도: {etf_info['risk_level']}
1년 수익률: {etf_info['1y_return']}%
운용보수: {etf_info['expense_ratio']}%
"""

    profile_str = f"""
나이: {investor_profile['age']}세
투자 경험: {investor_profile['experience']}
위험 허용도: {investor_profile['risk_tolerance']}
투자 목표: {investor_profile['goal']}
투자 기간: {investor_profile['period']}
"""

    prompt = ChatPromptTemplate.from_template(ETF_ANALYSIS_COT)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "etf_info": etf_str,
        "investor_profile": profile_str,
    })


# ── 3. Self-Consistency ────────────────────────────────────────────────────────

def self_consistency_answer(question: str, n_samples: int = 3) -> str:
    """Self-Consistency: 여러 추론 경로의 다수결"""
    prompt = f"{question}\n단계별로 추론하세요:"

    answers = []
    for i in range(n_samples):
        temp_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response = temp_llm.invoke([HumanMessage(content=prompt)])
        answers.append(response.content)
        print(f"  추론 {i+1}: {response.content[:100]}...")

    # 다수결 또는 종합
    consensus_prompt = f"""다음 {n_samples}가지 추론 결과를 종합하여 최종 답변을 도출하세요.

원래 질문: {question}

추론 결과들:
{"".join(f'{i+1}. {a}' + chr(10) for i, a in enumerate(answers))}

종합 최종 답변:"""

    final = llm.invoke([HumanMessage(content=consensus_prompt)])
    return final.content


# ── 4. Tree-of-Thought ────────────────────────────────────────────────────────

def tree_of_thought(problem: str) -> str:
    """Tree-of-Thought: 여러 접근법 탐색"""
    tot_prompt = f"""문제: {problem}

다음 3가지 관점에서 각각 분석해주세요:

## 관점 1: 재무 분석가 시각
{{재무 지표, 수익성, 비용 효율성 중심}}

## 관점 2: 리스크 매니저 시각
{{위험 요소, 분산투자, 헤지 전략 중심}}

## 관점 3: 장기 투자자 시각
{{성장 잠재력, 배당, 복리 효과 중심}}

## 종합 결론
{{세 관점을 통합한 최적 전략}}"""

    response = llm.invoke([HumanMessage(content=tot_prompt)])
    return response.content


# ── 5. ReAct 패턴 시뮬레이션 ─────────────────────────────────────────────────

def react_simulation(task: str) -> str:
    """ReAct 패턴: Reason + Act 시뮬레이션"""
    react_prompt = f"""다음 태스크를 ReAct 방식으로 수행하세요.
각 단계에서 Thought(추론), Action(행동), Observation(관찰)을 반복합니다.

태스크: {task}

Thought 1: 어떤 정보가 필요한지 생각해봅시다.
Action 1: [필요한 행동]
Observation 1: [결과]

Thought 2: 결과를 바탕으로...
Action 2: [다음 행동]
Observation 2: [결과]

Thought 3: 최종 판단
Final Answer: [결론]"""

    response = llm.invoke([HumanMessage(content=react_prompt)])
    return response.content


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 필요합니다")
        exit()

    print("=" * 60)
    print("1. 기본 CoT vs 직접 답변 비교")
    print("=" * 60)
    question = "월 200만원 투자, 연 8% 수익으로 10년 후 얼마가 되나요?"
    result = basic_cot(question)
    print(f"직접 답변: {result['direct'][:200]}")
    print(f"\nCoT 답변: {result['cot'][:400]}")

    print("\n" + "=" * 60)
    print("2. ETF CoT 분석")
    print("=" * 60)
    etf = {
        "name": "KODEX AI반도체핵심장비",
        "category": "섹터",
        "risk_level": "고위험",
        "1y_return": 65.3,
        "expense_ratio": 0.45,
    }
    investor = {
        "age": 35,
        "experience": "중급",
        "risk_tolerance": "중간",
        "goal": "자산 성장",
        "period": "5년",
    }
    analysis = analyze_etf_with_cot(etf, investor)
    print(analysis[:600])

    print("\n" + "=" * 60)
    print("3. Tree-of-Thought 포트폴리오 설계")
    print("=" * 60)
    problem = "1억원을 ETF로 분산 투자할 최적의 포트폴리오를 설계해주세요."
    tot_result = tree_of_thought(problem)
    print(tot_result[:800])
