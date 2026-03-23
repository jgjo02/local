# Week 9-5: 고급 RAG 실습 및 Project 3 완성

## 학습 목표
- Project 3 법률 상담 시스템 전체 통합
- 고급 RAG 패턴 비교 평가
- 프로덕션 배포 최적화

---

## 1. Project 3 전체 아키텍처

```
project3_legal_advisor/
├── app.py                      # Gradio 웹 UI
├── src/
│   ├── main_agent.py          # LegalAdvisorAgent (통합)
│   ├── agent_state.py         # LangGraph 상태 정의
│   ├── adaptive_rag_graph.py  # Adaptive RAG 그래프
│   ├── self_rag_graph.py      # Self-RAG 그래프
│   ├── corrective_rag_graph.py # Corrective RAG 그래프
│   ├── graders.py             # 관련성/지지도 평가기
│   ├── legal_tools.py         # 법률 검색 도구
│   ├── legal_vector_store.py  # 벡터 스토어
│   └── legal_document_processor.py
├── data/
│   └── sample_legal_docs.txt  # 법률 문서 샘플
└── requirements.txt
```

---

## 2. LegalAdvisorAgent 통합

```python
# src/main_agent.py 핵심 구조

class LegalAdvisorAgent:
    def __init__(self, config):
        self.config = config
        self.vector_store = LegalVectorStore(config)
        self.adaptive_graph = None
        self.self_rag_graph = None
        self.corrective_graph = None
        self._initialized = False

    def initialize(self):
        if not self.vector_store.load():
            processor = LegalDocumentProcessor(self.config)
            docs = processor.load_legal_documents()
            self.vector_store.build(docs)
            self.vector_store.save()

        retriever = self.vector_store.get_hybrid_retriever()

        self.adaptive_graph = build_adaptive_rag_graph(retriever, self.config)
        self.self_rag_graph = build_self_rag_graph(retriever, self.config)
        self.corrective_graph = build_corrective_rag_graph(retriever, self.config)

        self._initialized = True

    def _select_strategy(self, query: str) -> str:
        """쿼리 특성에 따라 RAG 전략 선택"""
        keywords_corrective = ["최근", "2024", "최신", "판례", "개정"]
        keywords_self = ["가능한가", "적법한가", "요건", "여부"]

        for kw in keywords_corrective:
            if kw in query:
                return "corrective"
        for kw in keywords_self:
            if kw in query:
                return "self_rag"
        return "adaptive"

    def chat(self, message: str, session_id: str = "default") -> dict:
        if not self._initialized:
            self.initialize()

        strategy = self._select_strategy(message)

        if strategy == "adaptive":
            result = self.adaptive_graph.invoke(
                {"query": message, ...},
                config={"configurable": {"thread_id": session_id}}
            )
        elif strategy == "self_rag":
            result = self.self_rag_graph.invoke(...)
        else:
            result = self.corrective_graph.invoke(...)

        return {
            "answer": result["answer"],
            "strategy": strategy,
            "sources": result.get("retrieved_docs", [])
        }
```

---

## 3. Gradio 법률 상담 UI

```python
# app.py 핵심 구조

import gradio as gr
from src.main_agent import LegalAdvisorAgent

agent = LegalAdvisorAgent(config)
agent.initialize()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ AI 법률 상담 시스템")
    gr.Markdown("> ⚠️ 법적 조언이 아닌 정보 제공 목적입니다.")

    with gr.Row():
        case_type = gr.Radio(
            choices=["민사", "노동", "형사", "부동산"],
            value="민사",
            label="사건 유형",
        )
        strategy_display = gr.Textbox(
            label="선택된 RAG 전략",
            interactive=False,
        )

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="법률 질문을 입력하세요...")
    send_btn = gr.Button("상담하기", variant="primary")

    def respond(message, history, case):
        result = agent.chat(f"[{case}] {message}")
        history.append((message, result["answer"]))
        return history, "", result["strategy"]

    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot, case_type],
        outputs=[chatbot, msg, strategy_display],
    )

demo.launch(server_port=7863)
```

---

## 4. 고급 RAG 성능 비교 평가

```python
test_scenarios = [
    {
        "query": "묵시적 임대차 갱신이란 무엇인가요?",
        "expected_strategy": "adaptive",
        "difficulty": "easy",
    },
    {
        "query": "세입자가 퇴거를 거부할 경우 집주인이 취할 수 있는 법적 조치는?",
        "expected_strategy": "self_rag",
        "difficulty": "medium",
    },
    {
        "query": "2024년 개정된 임대차보호법의 주요 변경사항은?",
        "expected_strategy": "corrective",
        "difficulty": "hard",
    },
]

def evaluate_rag_strategies(agent, test_cases):
    results = []
    for case in test_cases:
        result = agent.chat(case["query"])
        strategy_correct = result["strategy"] == case["expected_strategy"]

        results.append({
            "query": case["query"][:40],
            "difficulty": case["difficulty"],
            "expected_strategy": case["expected_strategy"],
            "actual_strategy": result["strategy"],
            "strategy_correct": strategy_correct,
        })

    accuracy = sum(1 for r in results if r["strategy_correct"]) / len(results)
    print(f"\n전략 선택 정확도: {accuracy:.0%}")
    return results
```

---

## 5. 프로덕션 최적화

### 5.1 캐싱

```python
import hashlib
from functools import lru_cache

class CachedLegalAgent:
    def __init__(self, agent, cache_ttl: int = 3600):
        self.agent = agent
        self._cache = {}
        self._cache_ttl = cache_ttl

    def chat(self, message: str, session_id: str = "default") -> dict:
        cache_key = hashlib.md5(message.encode()).hexdigest()
        if cache_key in self._cache:
            print("📋 캐시 히트")
            return self._cache[cache_key]

        result = self.agent.chat(message, session_id)
        self._cache[cache_key] = result
        return result
```

### 5.2 비동기 처리

```python
import asyncio

async def async_chat(agent, message: str) -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, agent.chat, message)
    return result
```

---

## 핵심 정리

**고급 RAG 전략 선택 가이드**:

```
쿼리 복잡도 낮음 + 정보 충분  →  Adaptive RAG
쿼리 정확성 중요 + 검증 필요   →  Self-RAG
최신 정보 필요 + 로컬 DB 부족  →  Corrective RAG
```

**Project 3 완성 체크리스트**:
- [x] 법률 문서 벡터 스토어 구축
- [x] 세 가지 RAG 전략 구현
- [x] 메타 라우터로 전략 자동 선택
- [x] Gradio 법률 상담 UI
- [x] 법적 면책조항 포함
