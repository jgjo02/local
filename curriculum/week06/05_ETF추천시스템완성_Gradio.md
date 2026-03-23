# Week 6-5: ETF 추천 시스템 완성 및 Gradio 배포

## 학습 목표
- ETF 추천 시스템 전체 아키텍처 통합
- Gradio 멀티탭 인터페이스 구현
- 프로덕션 배포 고려사항 이해

---

## 1. 시스템 아키텍처 전체 흐름

```
사용자 입력 (Gradio UI)
    ↓
ETFRecommender.recommend()
    ↓
하이브리드 검색 (ETFVectorStore)
├── BM25Retriever (키워드)
└── FAISS Retriever (벡터)
    ↓ EnsembleRetriever
위험도 필터링
    ↓
컨텍스트 포맷팅
    ↓
ChatOpenAI (GPT-4o-mini) 스트리밍
    ↓
Gradio 챗봇 UI 출력
```

---

## 2. 시스템 초기화 흐름

```python
# app.py 초기화 패턴
from src.config import Config
from src.etf_data_collector import ETFDataCollector
from src.etf_vector_store import ETFVectorStore
from src.etf_recommender import ETFRecommender

def initialize_system():
    config = Config()

    # 데이터 수집
    collector = ETFDataCollector(config.etf_data_path)

    # 벡터 스토어 초기화 (캐시 활용)
    vector_store = ETFVectorStore(config)
    if not vector_store.load():
        collector.load_data()
        vector_store.build_from_collector(collector)
        vector_store.save()

    # 추천 시스템
    recommender = ETFRecommender(config, vector_store)
    return recommender, collector
```

---

## 3. Gradio 멀티탭 UI

### 3.1 탭 구성

```python
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    session_id = gr.State("session_001")

    with gr.Tabs():
        # 탭 1: 챗봇
        with gr.TabItem("💬 상담 챗봇"):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="질문을 입력하세요...")
            with gr.Row():
                send_btn = gr.Button("전송", variant="primary")
                clear_btn = gr.Button("초기화")

        # 탭 2: 포트폴리오 설계
        with gr.TabItem("📊 포트폴리오"):
            amount = gr.Slider(100, 10000, 1000, label="투자금액(만원)")
            risk = gr.Radio(["저위험", "중립", "고위험"], value="중립")
            period = gr.Radio(["1년", "3년", "5년+"], value="3년")
            goals = gr.CheckboxGroup(
                ["노후준비", "자산성장", "배당수익"],
                value=["자산성장"]
            )
            portfolio_btn = gr.Button("포트폴리오 추천", variant="primary")
            portfolio_out = gr.Markdown()

        # 탭 3: ETF 현황
        with gr.TabItem("📋 ETF 목록"):
            stats_btn = gr.Button("현황 불러오기")
            stats_out = gr.Markdown()
```

### 3.2 스트리밍 이벤트 연결

```python
def stream_chat(message, history, risk_pref, sess_id):
    if not message.strip():
        return history, ""

    history.append((message, ""))
    full_response = ""

    for token in recommender.stream_recommend(
        question=message,
        session_id=sess_id,
        risk_preference=risk_pref if risk_pref != "선택 안함" else None,
    ):
        full_response += token
        history[-1] = (message, full_response)
        yield history, ""

send_btn.click(
    fn=stream_chat,
    inputs=[msg, chatbot, risk_selector, session_id],
    outputs=[chatbot, msg],
)
msg.submit(
    fn=stream_chat,
    inputs=[msg, chatbot, risk_selector, session_id],
    outputs=[chatbot, msg],
)
```

---

## 4. 예제 질문 버튼

```python
EXAMPLE_QUESTIONS = [
    "노후 대비 안전한 ETF 추천",
    "AI/반도체 성장주 투자",
    "월 100만원 장기 적립식 투자",
    "미국 주식 저비용 ETF 비교",
]

with gr.Column():
    gr.Markdown("### 💡 예제 질문")
    for q in EXAMPLE_QUESTIONS:
        btn = gr.Button(q, size="sm", variant="secondary")
        btn.click(fn=lambda x=q: x, outputs=msg)
```

---

## 5. 환경 설정 및 배포

### 5.1 .env 설정

```bash
OPENAI_API_KEY=sk-your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
SERVER_PORT=7862
```

### 5.2 실행

```bash
cd project2_etf_recommendation
pip install -r requirements.txt
python app.py
```

### 5.3 API 키 없이 테스트

```python
if not config.openai_api_key:
    gr.Warning("OpenAI API 키가 필요합니다. .env 파일을 설정해주세요.")
```

---

## 6. 전체 프로젝트 체크리스트

**데이터 레이어**:
- [x] `data/sample_etf_data.json` - 20개 ETF 데이터
- [x] `src/etf_data_collector.py` - 데이터 로드 및 Document 생성

**검색 레이어**:
- [x] `src/etf_vector_store.py` - FAISS + BM25 하이브리드

**추천 레이어**:
- [x] `src/etf_recommender.py` - RAG 체인 + 대화 이력

**평가 레이어**:
- [x] `src/etf_evaluator.py` - LLM-as-Judge 평가

**UI 레이어**:
- [x] `app.py` - Gradio 멀티탭 인터페이스

---

## 핵심 정리

**Project 2 핵심 학습 내용**:
1. 정형 데이터(ETF 수치)의 텍스트 표현 전략
2. 하이브리드 검색(BM25+FAISS)의 실전 적용
3. 금융 도메인 프롬프트 설계 원칙
4. LLM-as-Judge를 통한 추천 품질 평가
5. 멀티탭 Gradio UI 구현
