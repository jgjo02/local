# 📚 Week 1: LangChain 기초

## 🎯 주차 개요

1주차는 AI 서비스 개발의 기반을 다지는 주차입니다. Python 클린코드 작성법부터 시작하여 LLM의 생성 원리를 이해하고, LangChain의 핵심 아키텍처를 학습한 후 최종적으로 LCEL 문법을 활용한 Gradio 챗봇을 직접 구현합니다.

---

## 📅 세션 구성

| 차시 | 제목 | 핵심 내용 | 실습 |
|------|------|-----------|------|
| 1차시 | OT + 아이스브레이킹 + 프로젝트1 안내 | 커리큘럼 소개, AI 트렌드, 학습 로드맵 | 아이스브레이킹 활동 |
| 2차시 | 파이썬 클린코드 | PEP8, 타입힌팅, 데이터클래스, 예외처리 | 클린코드 리팩토링 |
| 3차시 | 환경설정 / LLM 생성 원리 + OpenAI API | Transformer, GPT, ChatCompletion API | API 호출 실습 |
| 4차시 | LangChain 소개 - 아키텍처 및 컴포넌트 | LangChain 구조, Runnable, Message | 컴포넌트 탐색 |
| 5차시 | LCEL 문법 + Gradio 챗봇 구현 | LCEL 파이프라인, Gradio UI | 챗봇 구현 |

---

## 🎓 학습 목표

### 지식 (Knowledge)
- LLM의 동작 원리와 토큰 생성 메커니즘 이해
- LangChain의 주요 컴포넌트 역할과 관계 파악
- LCEL(LangChain Expression Language) 문법 습득

### 기술 (Skills)
- Python 클린코드 원칙에 따른 코드 작성
- OpenAI ChatCompletion API 활용
- LangChain을 이용한 기본 체인 구성
- Gradio를 활용한 웹 UI 챗봇 구현

### 태도 (Attitude)
- AI 서비스 개발의 전체 흐름 이해
- 협업을 위한 코드 품질 의식 함양

---

## 🛠️ 사전 준비사항

### 필수 설치
```bash
# Python 3.11+ 설치 확인
python --version

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# 핵심 패키지 설치
pip install langchain langchain-openai langchain-community
pip install openai python-dotenv gradio
pip install faiss-cpu chromadb tiktoken
pip install black flake8 mypy
```

### 환경변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...  # LangSmith (선택)
LANGCHAIN_TRACING_V2=true  # LangSmith 추적 활성화
```

---

## 📁 파일 구조

```
week01/
├── README.md                          # 이 파일
├── 01_OT_아이스브레이킹.md             # 1차시 강의 자료
├── 02_파이썬_클린코드.md               # 2차시 강의 자료
├── 03_환경설정_LLM생성원리_OpenAI.md   # 3차시 강의 자료
├── 04_LangChain_소개_아키텍처.md       # 4차시 강의 자료
├── 05_LCEL_Gradio_챗봇.md             # 5차시 강의 자료
└── code/
    ├── 02_clean_code_examples.py      # 2차시 실습 코드
    ├── 03_openai_api.py               # 3차시 실습 코드
    ├── 04_langchain_basics.py         # 4차시 실습 코드
    └── 05_gradio_chatbot.py           # 5차시 실습 코드
```

---

## 🗺️ 주차별 학습 흐름

```
[1차시]          [2차시]          [3차시]          [4차시]          [5차시]
  OT &      →  Python      →  LLM 원리 &   →  LangChain    →  LCEL &
아이스브레이킹    클린코드       OpenAI API      아키텍처        Gradio 챗봇
                                                              ↓
                                                         🤖 첫 번째 챗봇 완성!
```

---

## 💡 핵심 포인트

1. **클린코드는 협업의 기본**: AI 서비스는 팀 프로젝트이므로 읽기 좋은 코드가 필수입니다
2. **LLM은 확률 기계**: 토큰을 하나씩 예측하는 원리를 이해하면 프롬프트 설계가 쉬워집니다
3. **LangChain = 추상화 레이어**: 복잡한 AI 파이프라인을 단순하게 연결하는 도구입니다
4. **LCEL = 함수 합성**: 파이프(|) 연산자로 컴포넌트를 연결하는 직관적인 방식입니다

---

## ⚠️ 주의사항

- API 키는 절대 코드에 직접 입력하지 마세요 (`.env` 파일 활용)
- OpenAI API는 유료입니다. 실습 중 불필요한 API 호출을 최소화하세요
- LangChain은 빠르게 업데이트됩니다. 항상 버전을 명시하여 설치하세요

---

## 📚 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/docs/)
- [OpenAI API 문서](https://platform.openai.com/docs/)
- [Gradio 공식 문서](https://www.gradio.app/docs/)
- [PEP 8 스타일 가이드](https://peps.python.org/pep-0008/)

---

## 🏆 1주차 미션

1주차를 마치면 다음을 완료해야 합니다:
- [ ] 로컬 개발환경 설정 완료
- [ ] OpenAI API 키 발급 및 테스트
- [ ] 기본 LangChain 체인 구성 및 실행
- [ ] Gradio 챗봇 로컬 실행 성공
