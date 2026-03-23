# 3차시: 환경설정 & LLM 생성 원리 & OpenAI API

## 학습 목표
- Transformer 아키텍처의 핵심 원리를 이해한다
- LLM이 텍스트를 생성하는 메커니즘을 설명할 수 있다
- OpenAI API를 활용하여 다양한 요청을 처리할 수 있다
- 토큰 기반 비용 계산 및 최적화 방법을 익힌다

---

## 1. Transformer 아키텍처 이해

### 1.1 딥러닝 NLP 발전사

```
RNN/LSTM (2015)  →  Attention (2016)  →  Transformer (2017)  →  GPT/BERT (2018)  →  GPT-3/4 (2020~)
순차 처리          어텐션 도입          병렬 처리              사전학습              초거대 LLM
기울기 소실        핵심 혁신            학습 속도 향상          파인튜닝 패러다임      In-context Learning
```

### 1.2 Transformer 전체 구조

```
                    ┌─────────────────────────────┐
                    │         Output               │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Linear + Softmax         │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴───────────────────┐
              │              Decoder Stack              │
              │  ┌─────────────────────────────────┐   │
              │  │  Masked Self-Attention           │   │
              │  │  Cross-Attention (Encoder)       │   │
              │  │  Feed-Forward Network            │   │
              │  └─────────────────────────────────┘   │
              │           × N layers                    │
              └────────────────────────────────────────┘
                                   │
              ┌────────────────────┴───────────────────┐
              │              Encoder Stack              │
              │  ┌─────────────────────────────────┐   │
              │  │  Multi-Head Self-Attention       │   │
              │  │  Feed-Forward Network            │   │
              │  │  Add & Norm                      │   │
              │  └─────────────────────────────────┘   │
              │           × N layers                    │
              └────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Input Embedding + Positional│
                    │  Encoding                    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │         Input Tokens         │
                    └─────────────────────────────┘
```

### 1.3 Self-Attention 메커니즘

Self-Attention의 핵심: **각 토큰이 다른 모든 토큰과의 관계를 학습**

```
수식: Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Q (Query): 현재 토큰이 "무엇을 찾는가?"
K (Key):   각 토큰이 "무엇을 제공하는가?"
V (Value): 각 토큰의 실제 정보 내용
d_k:       키 벡터의 차원 (스케일링 목적)
```

**예시: "나는 사과를 먹었다"**
```
토큰:  나는   사과를   먹었다
        ↕       ↕       ↕
나는  [1.0]  [0.2]  [0.8]  → "나는"은 "먹었다"와 강하게 연결
사과를[0.1]  [1.0]  [0.7]  → "사과를"은 "먹었다"와 연결
먹었다[0.8]  [0.6]  [1.0]  → "먹었다"는 주어와 목적어 모두 참조
```

### 1.4 Multi-Head Attention

여러 "헤드"가 서로 다른 관점에서 관계를 학습:
- Head 1: 문법적 관계 (주어-동사)
- Head 2: 의미적 관계 (동의어, 유의어)
- Head 3: 위치적 관계 (인접 단어)
- Head 4~N: 다양한 추상적 패턴

---

## 2. GPT의 텍스트 생성 원리

### 2.1 Autoregressive 생성

```
GPT는 다음 토큰을 예측하는 방식으로 텍스트 생성

입력: "한국의 수도는"
        ↓
   토큰화: ["한국의", "수도는"]
        ↓
   확률 분포 계산:
   "서울" → 0.89
   "부산" → 0.04
   "대전" → 0.02
   ...
        ↓
   "서울" 선택 → "한국의 수도는 서울"
        ↓
   계속 생성: "이다" → "." → [EOS]
```

### 2.2 토큰 (Token)

```
토큰 ≠ 단어

예시:
"ChatGPT"    → ["Chat", "G", "PT"]          (3 tokens)
"안녕하세요"  → ["안녕", "하세요"]            (2 tokens)
"Hello"      → ["Hello"]                    (1 token)
"unbelievable" → ["un", "believ", "able"]   (3 tokens)

규칙:
- 영어: 단어의 ~75%가 1 token
- 한국어: 단어당 평균 2-3 tokens (영어보다 비쌈)
- 1 token ≈ 4 영문자 ≈ 0.75 영단어
```

### 2.3 Temperature, Top-p, Top-k 파라미터

```
Temperature (창의성 조절):
  0.0  → 항상 가장 높은 확률 토큰 선택 (결정론적)
  0.7  → 적당한 다양성 (일반 대화 추천)
  1.0  → 학습 분포 그대로 샘플링
  2.0  → 매우 무작위 (창의적이지만 횡설수설 가능)

Top-k (후보 수 제한):
  k=50  → 상위 50개 토큰 중에서만 선택
  낮을수록 더 일관된 출력

Top-p (확률 누적 임계값):
  p=0.9  → 누적 확률 90%에 해당하는 토큰들에서만 선택
  nucleus sampling이라고도 함

권장 설정:
  정확한 답변: temperature=0.0~0.3
  일반 대화:   temperature=0.7, top_p=0.9
  창의적 글쓰기: temperature=1.0, top_p=0.95
```

---

## 3. OpenAI API 활용

### 3.1 API 키 설정 및 보안

```bash
# .env 파일 (절대 git에 커밋하지 말것!)
OPENAI_API_KEY=sk-...

# .gitignore에 반드시 추가
echo ".env" >> .gitignore
```

```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")
```

### 3.2 Chat Completion API 구조

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
        {"role": "user", "content": "파이썬의 장점은?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### 3.3 모델 버전 비교

| 모델 | 컨텍스트 | 입력 비용 | 출력 비용 | 특징 |
|------|---------|---------|---------|------|
| gpt-4o | 128K | $5/1M | $15/1M | 최고 성능, 멀티모달 |
| gpt-4o-mini | 128K | $0.15/1M | $0.6/1M | 경제적, 빠름 |
| gpt-3.5-turbo | 16K | $0.5/1M | $1.5/1M | 레거시, 저렴 |

### 3.4 스트리밍 응답

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "긴 이야기를 해줘"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3.5 토큰 비용 계산

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# 비용 계산
def estimate_cost(prompt: str, response: str, model: str = "gpt-4o-mini") -> float:
    prices = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6}
    }
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    price = prices[model]
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
    return cost
```

---

## 4. 에러 처리 및 재시도 로직

```python
import time
from openai import RateLimitError, APIError

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except RateLimitError:
            wait = 2 ** attempt  # 지수 백오프
            print(f"Rate limit 도달. {wait}초 대기...")
            time.sleep(wait)
        except APIError as e:
            print(f"API 오류: {e}")
            raise
    raise Exception("최대 재시도 초과")
```

---

## 💡 핵심 포인트
- Transformer의 Self-Attention은 모든 토큰 간 관계를 병렬로 계산
- GPT는 자동회귀(Autoregressive) 방식으로 한 번에 하나의 토큰을 생성
- Temperature가 낮을수록 결정론적, 높을수록 창의적
- 한국어는 영어보다 토큰 소모가 많아 비용이 더 높음

## ⚠️ 주의사항
- API 키는 절대 코드에 하드코딩 금지 (.env 파일 사용)
- max_tokens 설정으로 비용 제한 필수
- 프로덕션 환경에서는 반드시 재시도 로직 구현

## 🔍 심화학습
- [Attention Is All You Need 논문](https://arxiv.org/abs/1706.03762)
- [OpenAI Tokenizer 도구](https://platform.openai.com/tokenizer)
- [OpenAI Pricing](https://openai.com/pricing)

## ❓ 차시별 질문
1. Temperature=0과 Temperature=1의 차이를 실험해보세요
2. 같은 내용을 한국어/영어로 요청했을 때 토큰 수 차이는?
3. 스트리밍이 UX에 왜 중요한가요?
