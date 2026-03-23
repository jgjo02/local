# Week 3: 프롬프트 엔지니어링 및 메모리 관리

## 학습 목표

이번 주는 LLM 애플리케이션의 핵심 역량인 **프롬프트 엔지니어링**과 **대화 이력 관리**를 심층적으로 다룬다.
단순한 API 호출을 넘어, 모델이 원하는 방식으로 작동하도록 유도하는 설계 원칙과,
멀티턴 대화에서 맥락을 유지하는 메모리 아키텍처를 학습한다.
최종적으로 실제 서비스 수준의 주택청약 FAQ 챗봇을 완성한다.

---

## 주차 개요

```
Week 3 커리큘럼 구조
==============================================
세션 01 │ 프롬프트 엔지니어링 & 템플릿 설계
세션 02 │ Zero-shot / Few-shot 프롬프팅
세션 03 │ Chain-of-Thought 고급 추론
세션 04 │ 메모리 & 대화 이력 관리
세션 05 │ 주택청약 FAQ 챗봇 구현 (종합 프로젝트)
==============================================
```

---

## 세션별 상세 내용

### 세션 01: 프롬프트 엔지니어링 & 템플릿 설계
**파일:** `01_프롬프트엔지니어링_템플릿설계.md` / `code/01_prompt_templates.py`

LLM은 입력 프롬프트에 따라 전혀 다른 출력을 생성한다.
이 세션에서는 프롬프트의 구조적 원칙부터 LangChain의 `PromptTemplate`,
`ChatPromptTemplate`, `FewShotPromptTemplate`까지 체계적으로 학습한다.

**핵심 학습 항목:**
- 프롬프트 엔지니어링의 정의와 중요성
- System / Human / AI 역할 분리 원칙
- 명확한 지시문 작성 원칙 (구체성, 단계성, 맥락 제공)
- 출력 형식 지정 (JSON, Markdown, List, Table)
- PromptTemplate vs ChatPromptTemplate 구조 비교
- FewShotPromptTemplate 설계 방법
- 프롬프트 버전 관리 (LangChain Hub)
- 나쁜 프롬프트 vs 좋은 프롬프트 대조 사례 10개

**실습 목표:** 다양한 업무 도메인에 적용 가능한 재사용 템플릿 라이브러리 구축

---

### 세션 02: Zero-shot / Few-shot 프롬프팅
**파일:** `02_ZeroShot_FewShot.md` / `code/02_zero_few_shot.py`

모델에게 예시를 얼마나, 어떻게 제공하느냐에 따라 성능이 크게 달라진다.
이 세션에서는 In-context Learning의 원리를 이해하고,
동적 예제 선택기(SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector)를
활용하여 최적의 예시를 런타임에 자동 선택하는 시스템을 구현한다.

**핵심 학습 항목:**
- Zero-shot 프롬프팅 이론
- In-context Learning 원리와 한계
- Few-shot 예제 선택 전략 (다양성, 대표성, 난이도)
- 예제 수와 성능의 관계 (실험 데이터 포함)
- SemanticSimilarityExampleSelector 구현
- MaxMarginalRelevanceExampleSelector 활용
- 한국어 Few-shot 설계 팁

**실습 목표:** 동적 예제 선택 시스템을 갖춘 한국어 분류기 구현

---

### 세션 03: Chain-of-Thought 고급 추론
**파일:** `03_ChainOfThought.md` / `code/03_chain_of_thought.py`

단순한 질답을 넘어, 복잡한 다단계 추론이 필요한 작업에서
CoT(Chain-of-Thought) 기법이 왜 효과적인지 이론과 실험으로 검증한다.
Tree of Thoughts, ReAct, Self-Consistency 등 최신 추론 기법도 함께 소개한다.

**핵심 학습 항목:**
- Wei et al. (2022) CoT 논문 핵심 내용
- Zero-shot CoT ("Let's think step by step") 원리
- Few-shot CoT 예제 설계 방법론
- Tree of Thoughts (ToT) 탐색 알고리즘
- ReAct 패턴 (Reasoning + Acting)
- Self-Consistency (다수결 투표)
- Least-to-Most Prompting
- Program-aided Language Model (PAL)

**실습 목표:** 수학 문제 / 논리 퍼즐 자동 풀이기 구현 (CoT + Self-Consistency)

---

### 세션 04: 메모리 & 대화 이력 관리
**파일:** `04_메모리_대화이력관리.md` / `code/04_memory_management.py`

실제 챗봇 서비스에서 가장 중요한 것 중 하나는 이전 대화 맥락을 얼마나
효과적으로 유지하느냐이다. LangChain v0.3의 최신 메모리 API인
`RunnableWithMessageHistory`를 중심으로 다양한 메모리 전략을 학습한다.

**핵심 학습 항목:**
- 대화 이력 관리의 필요성 및 토큰 비용 문제
- ConversationBufferMemory / WindowMemory / SummaryMemory 비교
- ConversationSummaryBufferMemory (하이브리드 전략)
- RunnableWithMessageHistory (v0.3 최신 방식)
- ChatMessageHistory (외부 Redis/DB 연동)
- 멀티 사용자 session_id 관리
- 토큰 제한 전략

**실습 목표:** session_id 기반 멀티 사용자 대화 서비스 구현

---

### 세션 05: 주택청약 FAQ 챗봇 구현 (종합 프로젝트)
**파일:** `05_주택청약FAQ챗봇구현.md` / `code/05_faq_chatbot_demo.py`

지금까지 배운 모든 기술을 통합하여 실제 서비스 수준의 주택청약 FAQ 챗봇을 구현한다.
RAG 파이프라인, 메모리 관리, 특화 프롬프트, Gradio UI까지 포함한 엔드-투-엔드 프로젝트이다.

**핵심 학습 항목:**
- 프로젝트 아키텍처 설계
- 주택청약 FAQ 데이터 구조화
- PDF 로딩 및 청약 특화 청크 전략
- 임베딩 및 Chroma 벡터저장소 구축
- 메모리 포함 RAG 체인 구성
- 청약 상담사 시스템 프롬프트 설계
- Gradio UI 구현
- 성능 테스트 및 개선

**실습 목표:** 배포 가능한 주택청약 FAQ Gradio 챗봇 완성

---

## 주차별 커리큘럼 전체 맵

```
Week 1: LangChain 기초 & LLM 연동
Week 2: 문서 처리 & 임베딩 & 벡터 저장소
Week 3: 프롬프트 엔지니어링 & 메모리 관리  ◀ 현재
Week 4: 고급 검색 기술 (BM25, 하이브리드, Re-ranking)
Week 5: LangChain Agents & Tools
Week 6: 실전 RAG 시스템 구축
Week 7: 평가 & 모니터링 & 배포
Week 8: 종합 프로젝트 발표
```

---

## 선행 학습 요건

| 항목 | 필요 수준 |
|------|-----------|
| Python | 중급 (함수, 클래스, 데코레이터) |
| LangChain | Week 1-2 완료 |
| OpenAI API | 기본 사용법 숙지 |
| 벡터 데이터베이스 | Chroma 기본 이해 |

---

## 이번 주 설치 패키지

```bash
pip install langchain langchain-openai langchain-community
pip install chromadb sentence-transformers
pip install gradio
pip install rank_bm25
pip install tiktoken
```

---

## 학습 성과 지표 (KPI)

이번 주를 완료하면 다음을 독립적으로 구현할 수 있어야 한다:

1. 도메인 특화 ChatPromptTemplate 설계 및 버전 관리
2. SemanticSimilarityExampleSelector를 활용한 동적 Few-shot 시스템
3. CoT 기법을 적용한 다단계 추론 파이프라인
4. RunnableWithMessageHistory 기반 멀티 세션 챗봇
5. RAG + 메모리 + Gradio가 통합된 FAQ 챗봇

---

## 💡 핵심 포인트

- 프롬프트 엔지니어링은 "모델을 바꾸는 것"이 아니라 "모델과 소통하는 언어를 최적화하는 것"이다.
- Zero-shot은 빠르지만, Few-shot은 일관성을 높인다. 상황에 맞게 선택하라.
- CoT는 복잡한 추론 작업에서 정확도를 획기적으로 높이지만, 토큰 비용이 증가한다.
- 메모리 관리는 UX의 핵심이다. 사용자는 "챗봇이 기억하는가"로 품질을 판단한다.
- 실전 프로젝트는 "완벽한 기술"보다 "적절한 기술의 조합"이 더 중요하다.

---

## ⚠️ 주의사항

- OpenAI API 키는 절대 코드에 하드코딩하지 말 것 (`.env` 파일 사용)
- Few-shot 예제는 편향되지 않도록 다양한 케이스를 포함할 것
- 메모리 크기를 제한하지 않으면 토큰 비용이 폭발적으로 증가할 수 있음
- LangChain v0.3에서 일부 메모리 API가 변경됨 — 공식 문서와 버전 확인 필수
- Gradio 배포 시 API 키 노출에 주의할 것

---

## 🔍 심화학습 자료

- [Prompt Engineering Guide](https://www.promptingguide.ai/kr)
- [Wei et al. 2022 - Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [LangChain Memory Docs (v0.3)](https://python.langchain.com/docs/concepts/memory/)
- [Yao et al. 2023 - Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [LangChain Hub](https://smith.langchain.com/hub)

---

## ❓ 차시별 질문

1. 프롬프트의 길이와 품질은 항상 비례 관계인가? 그렇지 않다면 왜인가?
2. Few-shot 예제가 많을수록 성능이 좋아지는가? 언제 역효과가 나는가?
3. CoT가 작은 모델(7B 이하)에서도 효과적인가?
4. 대화 이력을 전부 유지하는 것이 왜 문제인가?
5. 실제 서비스에서 메모리 전략을 선택할 때 가장 중요한 고려사항은 무엇인가?
