"""
Week 7 - Session 1: Tool Calling 기초 및 에이전트 개념 실습
=============================================================
이 파일에서는 OpenAI Function Calling API부터 LangChain 에이전트까지
단계별로 학습합니다.

Prerequisites:
    pip install langchain langchain-openai langchain-community
    export OPENAI_API_KEY="your-api-key"
"""

import json
import os
from typing import Any

# ============================================================
# Part 1: OpenAI API 수준에서의 Function Calling (직접 구현)
# ============================================================

from openai import OpenAI

client = OpenAI()


def demo_raw_function_calling():
    """OpenAI API를 직접 사용하는 Function Calling 데모"""

    # Step 1: 도구(함수) 정의
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_korean_law",
                "description": (
                    "한국 법령 데이터베이스에서 법령을 검색합니다. "
                    "특정 법률의 조문이나 법령명으로 검색이 가능합니다. "
                    "법률 내용 확인, 관련 조문 탐색 시 사용하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색할 법령명 또는 키워드 (예: '근로기준법 제50조')"
                        },
                        "law_type": {
                            "type": "string",
                            "enum": ["법률", "대통령령", "부령", "조례", "전체"],
                            "description": "검색할 법령의 종류",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_legal_deadline",
                "description": (
                    "법률상 기간(시효, 제척기간, 신청기한 등)을 계산합니다. "
                    "기준일로부터 특정 기간이 경과한 날짜를 반환합니다."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "기준일 (YYYY-MM-DD 형식)"
                        },
                        "period_type": {
                            "type": "string",
                            "enum": ["소멸시효", "제척기간", "항소기간", "신청기한"],
                            "description": "기간 종류"
                        },
                        "years": {
                            "type": "integer",
                            "description": "연 단위 기간"
                        },
                        "months": {
                            "type": "integer",
                            "description": "월 단위 기간"
                        },
                        "days": {
                            "type": "integer",
                            "description": "일 단위 기간"
                        },
                    },
                    "required": ["start_date", "period_type"],
                },
            },
        },
    ]

    # 실제 함수 구현 (Mock)
    def search_korean_law(query: str, law_type: str = "전체") -> str:
        """법령 검색 Mock 함수"""
        mock_results = {
            "근로기준법": (
                "근로기준법 제50조(근로시간) "
                "① 1주 간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. "
                "② 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다."
            ),
            "최저임금": (
                "최저임금법 제5조(최저임금액) "
                "최저임금은 시간·일(日)·주(週) 또는 월(月)을 단위로 하여 정한다. "
                "2024년 최저임금: 시간당 9,860원"
            ),
        }
        for key, value in mock_results.items():
            if key in query:
                return value
        return f"'{query}'에 관한 법령 검색 결과: 관련 법령 3건 발견"

    def calculate_legal_deadline(
        start_date: str,
        period_type: str,
        years: int = 0,
        months: int = 0,
        days: int = 0
    ) -> str:
        """법률 기간 계산 Mock 함수"""
        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = start + relativedelta(years=years, months=months, days=days)
            return (
                f"{period_type} 기간 계산 결과:\n"
                f"기준일: {start.strftime('%Y년 %m월 %d일')}\n"
                f"만료일: {end.strftime('%Y년 %m월 %d일')}\n"
                f"주의: 만료일이 공휴일인 경우 다음 영업일로 연장될 수 있습니다."
            )
        except Exception as e:
            return f"계산 오류: {str(e)}"

    # 함수 맵 (이름 → 실제 함수)
    function_map = {
        "search_korean_law": search_korean_law,
        "calculate_legal_deadline": calculate_legal_deadline,
    }

    # Step 2: 첫 번째 API 호출
    messages = [
        {
            "role": "user",
            "content": "근로기준법상 근로시간 규정을 알려주고, 2024년 1월 1일부터 소멸시효 3년을 계산해줘."
        }
    ]

    print("=" * 60)
    print("Step 1: 첫 번째 API 호출")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"tool_calls: {assistant_message.tool_calls}")

    # Step 3: 도구 호출 처리
    if assistant_message.tool_calls:
        messages.append(assistant_message)  # assistant 메시지 추가

        print("\n" + "=" * 60)
        print("Step 2: 도구 실행")
        print("=" * 60)

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n도구 호출: {function_name}")
            print(f"인수: {function_args}")

            # 실제 함수 실행
            result = function_map[function_name](**function_args)
            print(f"결과: {result[:100]}...")

            # tool 결과 메시지 추가
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Step 4: 최종 답변 생성
    print("\n" + "=" * 60)
    print("Step 3: 최종 답변 생성")
    print("=" * 60)

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    final_answer = final_response.choices[0].message.content
    print(f"\n최종 답변:\n{final_answer}")
    return final_answer


# ============================================================
# Part 2: LangChain Tool 인터페이스 활용
# ============================================================

from langchain_core.tools import tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Type


# 방법 1: @tool 데코레이터 (가장 간단한 방법)
@tool
def search_law_simple(query: str) -> str:
    """한국 법령 데이터베이스에서 법령을 검색합니다.
    법률 조문, 법령명, 키워드로 검색하세요.

    Args:
        query: 검색할 법령명 또는 키워드

    Returns:
        검색된 법령 내용
    """
    # Mock 구현
    mock_db = {
        "형법": "형법 제250조(살인) 사람을 살해한 자는 사형, 무기 또는 5년 이상의 징역에 처한다.",
        "민법": "민법 제750조(불법행위의 내용) 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
        "근로기준법": "근로기준법 제23조(해고 등의 제한) ① 사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉, 그 밖의 징벌을 하지 못한다.",
    }

    for key, value in mock_db.items():
        if key in query:
            return value

    return f"'{query}' 관련 법령: 검색 결과 없음"


# 확인: @tool 데코레이터로 생성된 도구의 속성
print(f"Tool 이름: {search_law_simple.name}")
print(f"Tool 설명: {search_law_simple.description[:50]}...")
print(f"Tool 스키마: {search_law_simple.args_schema.model_json_schema()}")


# 방법 2: StructuredTool (복잡한 입력 스키마)
class LawSearchInput(BaseModel):
    """법령 검색 도구의 입력 스키마"""
    query: str = Field(description="검색할 법령명 또는 키워드")
    law_type: Optional[str] = Field(
        default="전체",
        description="법령 종류: 법률, 대통령령, 부령, 조례, 전체"
    )
    max_results: int = Field(
        default=5,
        description="반환할 최대 결과 수",
        ge=1,
        le=20
    )


def search_law_impl(query: str, law_type: str = "전체", max_results: int = 5) -> str:
    """실제 법령 검색 구현 (Mock)"""
    return f"[{law_type}] '{query}' 검색 결과 (최대 {max_results}건):\n법령 내용..."


search_law_structured = StructuredTool.from_function(
    func=search_law_impl,
    name="search_korean_law",
    description=(
        "한국 법령 데이터베이스에서 법령을 검색합니다. "
        "법률 종류별 필터링과 결과 수 제한이 가능합니다."
    ),
    args_schema=LawSearchInput,
    return_direct=False,  # 에이전트가 결과를 계속 처리할 수 있도록
)


# 방법 3: BaseTool 상속 (최대 제어권)
class LegalCalculatorTool(BaseTool):
    """법률 기간 계산 도구 - BaseTool 상속 방식"""

    name: str = "legal_calculator"
    description: str = (
        "법률상 기간을 계산합니다. 소멸시효, 제척기간, 항소기간 등을 "
        "기준일로부터 계산하여 만료일을 반환합니다. "
        "날짜는 반드시 YYYY-MM-DD 형식으로 입력하세요."
    )

    def _run(
        self,
        start_date: str,
        period_days: int = 0,
        period_months: int = 0,
        period_years: int = 0,
    ) -> str:
        """동기 실행"""
        from datetime import datetime
        try:
            from dateutil.relativedelta import relativedelta
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = start + relativedelta(
                years=period_years,
                months=period_months,
                days=period_days
            )
            return f"만료일: {end.strftime('%Y년 %m월 %d일')}"
        except ImportError:
            # dateutil 없을 경우 간단한 계산
            from datetime import timedelta
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = start + timedelta(days=period_days + period_months * 30 + period_years * 365)
            return f"만료일 (근사): {end.strftime('%Y년 %m월 %d일')}"
        except ValueError as e:
            return f"날짜 형식 오류: {str(e)}. YYYY-MM-DD 형식을 사용하세요."

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """비동기 실행 - 동기 메서드를 호출"""
        return self._run(*args, **kwargs)


# ============================================================
# Part 3: 에이전트 구축 및 실행
# ============================================================

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_legal_agent():
    """법률 자문 에이전트 생성"""

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # 결정론적 출력을 위해 0으로 설정
    )

    # 도구 목록
    tools = [
        search_law_simple,
        search_law_structured,
        LegalCalculatorTool(),
    ]

    # 프롬프트 템플릿
    # 주의: {agent_scratchpad} placeholder는 필수입니다
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 한국 법률 전문가 AI 어시스턴트입니다.
사용자의 법률 질문에 대해 정확하고 신뢰할 수 있는 정보를 제공합니다.

중요 지침:
1. 항상 관련 법령을 검색하여 근거를 제시하세요
2. 불확실한 경우 "전문 법률가와 상담을 권장합니다"라고 안내하세요
3. 법률 기간 계산이 필요한 경우 계산 도구를 사용하세요
4. 답변은 명확하고 이해하기 쉽게 작성하세요

면책 조항: 이 AI의 법률 정보는 참고용이며, 실제 법률 조언을 대체하지 않습니다."""
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor로 래핑
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # 중간 단계도 반환
    )

    return executor


def demo_agent_execution():
    """에이전트 실행 데모"""

    agent = create_legal_agent()

    test_questions = [
        "근로기준법에서 해고 관련 규정을 알려주세요.",
        "2024년 3월 15일에 교통사고가 발생했을 때 손해배상 소멸시효는 언제까지인가요?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print(f"질문: {question}")
        print("=" * 70)

        result = agent.invoke({"input": question})

        print("\n최종 답변:")
        print(result["output"])

        if result.get("intermediate_steps"):
            print(f"\n[사용된 도구: {len(result['intermediate_steps'])}회 호출]")
            for step in result["intermediate_steps"]:
                action, observation = step
                print(f"  - {action.tool}: {str(observation)[:80]}...")


# ============================================================
# Part 4: 에이전트 스트리밍 (실시간 출력)
# ============================================================

def demo_agent_streaming():
    """에이전트 스트리밍 실행 데모"""

    agent = create_legal_agent()

    print("\n에이전트 스트리밍 실행:")
    print("=" * 70)

    # 스트리밍으로 실행
    for chunk in agent.stream(
        {"input": "형법상 사기죄의 구성요건은 무엇인가요?"}
    ):
        # chunk 타입에 따라 다른 처리
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(f"\n[도구 호출] {action.tool}: {action.tool_input}")
        elif "steps" in chunk:
            for step in chunk["steps"]:
                print(f"[관찰] {str(step.observation)[:100]}...")
        elif "output" in chunk:
            print(f"\n[최종 답변]\n{chunk['output']}")


# ============================================================
# Part 5: 도구 직접 테스트
# ============================================================

def demo_tool_testing():
    """도구 단위 테스트 데모"""

    print("\n도구 단위 테스트:")
    print("=" * 70)

    # @tool 데코레이터로 만든 도구 테스트
    result1 = search_law_simple.invoke({"query": "근로기준법"})
    print(f"법령 검색 결과: {result1}")

    # StructuredTool 테스트
    result2 = search_law_structured.invoke({
        "query": "민법",
        "law_type": "법률",
        "max_results": 3
    })
    print(f"구조화 검색 결과: {result2}")

    # BaseTool 인스턴스 테스트
    calc_tool = LegalCalculatorTool()
    result3 = calc_tool.invoke({
        "start_date": "2024-01-01",
        "period_years": 3
    })
    print(f"기간 계산 결과: {result3}")

    # 도구 스키마 확인
    print("\n도구 스키마 확인:")
    for t in [search_law_simple, search_law_structured, calc_tool]:
        print(f"\n{t.name}:")
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        print(f"  스키마: {json.dumps(schema, ensure_ascii=False, indent=2)[:200]}")


# ============================================================
# Part 6: 에이전트 유형 비교
# ============================================================

def compare_agent_types():
    """다양한 에이전트 유형 비교"""

    from langchain.agents import create_react_agent
    from langchain_core.prompts import PromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_law_simple]

    print("\n에이전트 유형 비교:")
    print("=" * 70)

    # 1. Tool Calling Agent (권장 - OpenAI native tool calling 사용)
    tool_calling_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 법률 전문가입니다."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    tool_calling_agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)
    tool_calling_executor = AgentExecutor(
        agent=tool_calling_agent, tools=tools, verbose=False
    )
    print("1. Tool Calling Agent: OpenAI의 native tool calling 활용")
    print("   - 장점: 정확한 JSON 인수 생성, 병렬 호출 지원")
    print("   - 단점: OpenAI 모델에 의존적")

    # 2. ReAct Agent (범용성 높음)
    react_template = """당신은 법률 전문가입니다. 다음 도구를 사용할 수 있습니다:

{tools}

사용 형식:
Thought: 무엇을 해야 할지 생각
Action: 사용할 도구 이름 [{tool_names}]
Action Input: 도구 입력값
Observation: 도구 실행 결과
... (이 과정 반복)
Thought: 이제 최종 답변을 알겠습니다
Final Answer: 최종 답변

질문: {input}
{agent_scratchpad}"""

    react_prompt = PromptTemplate.from_template(react_template)

    try:
        react_agent = create_react_agent(llm, tools, react_prompt)
        react_executor = AgentExecutor(
            agent=react_agent, tools=tools, verbose=False
        )
        print("\n2. ReAct Agent: 범용 추론+행동 프레임워크")
        print("   - 장점: 모든 LLM과 호환, 추론 과정 명시적")
        print("   - 단점: 프롬프트 설계 복잡, 파싱 오류 가능")
    except Exception as e:
        print(f"\n2. ReAct Agent 생성 오류 (정상): {e}")

    print("\n3. 권장 선택:")
    print("   - OpenAI 모델 사용 시: Tool Calling Agent")
    print("   - 오픈소스 모델 사용 시: ReAct Agent")
    print("   - 복잡한 제어 필요 시: LangGraph (8주차 학습)")


# ============================================================
# Main 실행
# ============================================================

if __name__ == "__main__":
    import sys

    # 환경 변수 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("일부 데모는 Mock 데이터를 사용합니다.\n")

    # 데모 선택
    demos = {
        "1": ("Raw Function Calling API", demo_raw_function_calling),
        "2": ("도구 단위 테스트", demo_tool_testing),
        "3": ("에이전트 실행", demo_agent_execution),
        "4": ("에이전트 스트리밍", demo_agent_streaming),
        "5": ("에이전트 유형 비교", compare_agent_types),
    }

    if len(sys.argv) > 1:
        demo_key = sys.argv[1]
        if demo_key in demos:
            print(f"\n실행: {demos[demo_key][0]}")
            demos[demo_key][1]()
        else:
            print(f"사용법: python {sys.argv[0]} [1-5]")
    else:
        # 기본: 도구 테스트만 실행 (API 키 불필요)
        print("도구 단위 테스트 실행 (API 키 불필요):")
        demo_tool_testing()
        print("\n비교 정보:")
        compare_agent_types()

        print("\n\n전체 데모를 실행하려면:")
        print("  python 01_tool_calling_basics.py 3  # 에이전트 실행")
        print("  python 01_tool_calling_basics.py 4  # 스트리밍")
