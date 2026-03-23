"""
3차시: OpenAI API 실습
- ChatCompletion API 기본 사용
- 스트리밍 응답 처리
- 토큰 카운팅 및 비용 계산
- 에러 처리 및 재시도 로직
"""

import os
import time
import tiktoken
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
from typing import Generator

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 1. 기본 Chat Completion
# ============================================================

def basic_chat_completion(user_message: str, system_message: str = None) -> str:
    """기본 챗 완성 요청"""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content


def chat_with_history(conversation_history: list, new_message: str) -> tuple[str, list]:
    """대화 이력을 유지하면서 대화"""
    conversation_history.append({
        "role": "user",
        "content": new_message
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        temperature=0.7
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message, conversation_history


# ============================================================
# 2. 스트리밍 응답
# ============================================================

def stream_chat_completion(user_message: str) -> Generator[str, None, None]:
    """스트리밍으로 응답 생성"""
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


def print_streaming_response(user_message: str):
    """스트리밍 응답을 실시간 출력"""
    print("AI: ", end="", flush=True)
    full_response = ""

    for token in stream_chat_completion(user_message):
        print(token, end="", flush=True)
        full_response += token

    print()  # 줄바꿈
    return full_response


# ============================================================
# 3. 토큰 카운팅 및 비용 계산
# ============================================================

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """텍스트의 토큰 수 계산"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_message_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    """메시지 리스트의 총 토큰 수 계산"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0

    for message in messages:
        total_tokens += 4  # 메시지 포맷 오버헤드
        for key, value in message.items():
            total_tokens += len(encoding.encode(str(value)))

    total_tokens += 2  # 응답 시작 토큰
    return total_tokens


# 모델별 가격 (1M 토큰당 USD)
MODEL_PRICES = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini"
) -> dict:
    """API 호출 비용 추정"""
    if model not in MODEL_PRICES:
        return {"error": f"Unknown model: {model}"}

    prices = MODEL_PRICES[model]
    input_cost = (input_tokens * prices["input"]) / 1_000_000
    output_cost = (output_tokens * prices["output"]) / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "total_cost_krw": round(total_cost * 1350, 2)
    }


def chat_with_cost_tracking(user_message: str, model: str = "gpt-4o-mini") -> dict:
    """비용 추적 포함 챗 완성"""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    usage = response.usage
    cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens, model)

    return {
        "response": response.choices[0].message.content,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        },
        "cost": cost
    }


# ============================================================
# 4. Temperature 비교 실험
# ============================================================

def compare_temperatures(prompt: str, temperatures: list = [0.0, 0.5, 1.0, 1.5]) -> dict:
    """다양한 temperature로 같은 프롬프트 실행 비교"""
    results = {}

    for temp in temperatures:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=200
        )
        results[f"temp_{temp}"] = response.choices[0].message.content

    return results


# ============================================================
# 5. 에러 처리 및 재시도 로직
# ============================================================

def call_with_exponential_backoff(
    messages: list,
    model: str = "gpt-4o-mini",
    max_retries: int = 5,
    initial_delay: float = 1.0
) -> str:
    """지수 백오프 재시도 로직"""
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Rate limit 도달 (시도 {attempt + 1}/{max_retries}). "
                  f"{delay:.1f}초 대기...")
            time.sleep(delay)
            delay *= 2  # 지수 증가

        except APIConnectionError as e:
            if attempt == max_retries - 1:
                raise
            print(f"연결 오류 (시도 {attempt + 1}/{max_retries}). "
                  f"{delay:.1f}초 대기...")
            time.sleep(delay)
            delay *= 2

        except APIError as e:
            if e.status_code in [500, 502, 503, 529]:  # 서버 오류
                if attempt == max_retries - 1:
                    raise
                print(f"서버 오류 {e.status_code} (시도 {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise  # 4xx 오류는 재시도 불필요


# ============================================================
# 6. JSON 모드
# ============================================================

def extract_json_info(text: str) -> dict:
    """텍스트에서 JSON 형식으로 정보 추출"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "정보를 JSON 형식으로 추출하세요."
            },
            {
                "role": "user",
                "content": f"다음 텍스트에서 이름, 나이, 직업을 추출하세요:\n{text}"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    import json
    return json.loads(response.choices[0].message.content)


# ============================================================
# 7. 한국어 vs 영어 토큰 비교
# ============================================================

def compare_token_counts():
    """한국어와 영어의 토큰 수 비교"""
    examples = [
        ("Hello, how are you?", "안녕하세요, 어떻게 지내세요?"),
        ("What is machine learning?", "머신러닝이란 무엇인가요?"),
        ("Please explain RAG in detail.", "RAG에 대해 자세히 설명해주세요."),
    ]

    print("=" * 60)
    print(f"{'영어':30s} {'토큰':5s} | {'한국어':30s} {'토큰':5s}")
    print("-" * 60)

    for en, ko in examples:
        en_tokens = count_tokens(en)
        ko_tokens = count_tokens(ko)
        print(f"{en[:28]:30s} {en_tokens:5d} | {ko[:14]:30s} {ko_tokens:5d}")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("1. 기본 Chat Completion")
    print("=" * 50)
    response = basic_chat_completion(
        "파이썬의 주요 특징 3가지를 간단히 설명해주세요.",
        "당신은 프로그래밍 교육 전문가입니다. 간결하게 답변하세요."
    )
    print(f"응답: {response}\n")

    print("=" * 50)
    print("2. 스트리밍 응답")
    print("=" * 50)
    print_streaming_response("LangChain이란 무엇인지 2-3문장으로 설명해주세요.")
    print()

    print("=" * 50)
    print("3. 비용 추적")
    print("=" * 50)
    result = chat_with_cost_tracking("AI 기술의 미래는?")
    print(f"응답: {result['response'][:100]}...")
    print(f"사용 토큰: {result['usage']}")
    print(f"비용: ${result['cost']['total_cost_usd']:.6f} "
          f"(₩{result['cost']['total_cost_krw']:.4f})\n")

    print("=" * 50)
    print("4. 한국어 vs 영어 토큰 비교")
    print("=" * 50)
    compare_token_counts()
