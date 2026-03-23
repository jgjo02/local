# 12주 AI 서비스 개발 과정

> LangChain v0.3 + RAG + LangGraph + Gradio를 활용한 실전 AI 서비스 개발

## 📚 커리큘럼 개요

| 주차 | 주제 | 핵심 기술 |
|------|------|-----------|
| Week 01 | LLM 기초, LangChain 소개 | OpenAI API, LCEL, Gradio |
| Week 02 | RAG 기초, 문서 처리 | FAISS, 임베딩, NaiveRAG |
| Week 03 | 프롬프트 엔지니어링 | Few-shot, CoT, 메모리 |
| Week 04 | 고급 검색 | 하이브리드 검색, 쿼리 확장, 재순위화 |
| Week 05 | LLM 평가 | LLM-as-Judge, RAGAS, 임베딩 모델 |
| Week 06 | **Project 2**: ETF 추천 | 하이브리드 RAG, 포트폴리오 설계 |
| Week 07 | Tool Calling, 에이전트 | ReAct, LangChain Tools |
| Week 08 | LangGraph | StateGraph, 조건부 엣지, 멀티에이전트 |
| Week 09 | 고급 RAG | Adaptive/Self/Corrective RAG |
| Week 10 | 비정형 문서 | unstructured, PDF/표 파싱 |
| Week 11 | 고급 기술 | GraphRAG, 멀티모달, 배포 최적화 |
| Week 12 | 마무리 | 포트폴리오, 취업, 트렌드 |

## 🚀 4개 프로젝트

### Project 1: 주택청약 FAQ 챗봇
```
기술: NaiveRAG + LCEL + RunnableWithMessageHistory + Gradio
경로: projects/project1_consulting_chatbot/
```

### Project 2: ETF 추천 시스템
```
기술: 하이브리드 검색(BM25+FAISS) + LLM-as-Judge + Gradio
경로: projects/project2_etf_recommendation/
```

### Project 3: 법률 상담 AI
```
기술: LangGraph + Adaptive/Self/Corrective RAG + 멀티전략
경로: projects/project3_legal_advisor/
```

### Project 4: 기업 분석 시스템
```
기술: GraphRAG(Neo4j) + 멀티모달(CLIP+GPT-4o) + unstructured
경로: projects/project4_company_analysis/
```

## 🛠 기술 스택

```
LLM: OpenAI GPT-4o-mini, GPT-4o
Framework: LangChain v0.3 (LCEL, Agent, LangGraph)
Vector DB: FAISS, Chroma
Keyword Search: BM25 (rank-bm25)
Embeddings: OpenAI text-embedding-3-small, BAAI/bge-m3
Graph DB: Neo4j
Multimodal: CLIP, GPT-4o Vision
Document: unstructured (PDF, HTML, 표)
UI: Gradio 4.x
Evaluation: LLM-as-Judge, RAGAS
Deployment: HuggingFace Spaces, Docker
```

## 📁 디렉토리 구조

```
local/
├── README.md                   # 이 파일
├── curriculum/
│   ├── week01/                 # Week 1: LLM 기초
│   │   ├── README.md
│   │   ├── 01_OT_아이스브레이킹.md
│   │   ├── 02_파이썬_클린코드.md
│   │   ├── 03_환경설정_LLM생성원리_OpenAI.md
│   │   ├── 04_LangChain_소개_아키텍처.md
│   │   ├── 05_LCEL_Gradio_챗봇.md
│   │   └── code/
│   ├── week02/ ... week12/     # Week 2-12
└── projects/
    ├── project1_consulting_chatbot/
    ├── project2_etf_recommendation/
    ├── project3_legal_advisor/
    └── project4_company_analysis/
```

## ⚡ 빠른 시작

```bash
# 프로젝트별 의존성 설치 및 실행
cd projects/project2_etf_recommendation
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 입력
python app.py
```

## 💡 핵심 학습 포인트

1. **RAG 파이프라인**: 문서 수집 → 전처리 → 임베딩 → 검색 → 생성
2. **하이브리드 검색**: BM25(키워드) + Dense(의미) = 더 나은 검색
3. **LangGraph**: 조건부 분기, 반복, Human-in-the-Loop 가능한 에이전트
4. **LLM 평가**: 정량/정성 지표, LLM-as-Judge 패턴
5. **비용 최적화**: gpt-4o-mini 활용, 캐싱, 배치 처리

---

*12주 AI 서비스 개발 과정 | LangChain v0.3 기반*