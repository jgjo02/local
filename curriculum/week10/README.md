# 📚 Week 10: 비정형 문서 처리 (Unstructured Document Processing)

## 🎯 학습 목표

이번 주차에서는 실제 기업 문서(사업보고서, IR 보고서 등)에서 발생하는 **비정형 데이터 처리** 문제를 심층적으로 다룹니다. `unstructured` 라이브러리를 중심으로 PDF, 표, 이미지 등 다양한 형태의 문서를 파싱하고, 이를 RAG 시스템에 통합하는 방법을 학습합니다.

---

## 📋 차시 구성

| 차시 | 주제 | 핵심 내용 |
|------|------|-----------|
| 1차시 | [프로젝트 4] 안내 + 비정형 문서 처리의 이해 | 비정형 데이터 개념, 처리 파이프라인 개요 |
| 2차시 | 비정형 문서 파싱 - unstructured 라이브러리 활용 1 | 기본 파싱, Element 타입, PDF 전략 |
| 3차시 | 비정형 문서 파싱 - unstructured 라이브러리 활용 2 | 고급 파싱, OCR, 청킹 전략 |
| 4차시 | 비정형 문서 파싱 - unstructured 라이브러리 활용 3 | 사업보고서 특화, 배치 처리, 품질 검증 |
| 5차시 | 상장기업 사업보고서를 활용한 비정형 문서 처리 및 기본 RAG 구현 | DART API, 사업보고서 RAG 완성 |

---

## 🗂️ 주요 학습 내용

### 비정형 문서 처리 전체 흐름

```
원본 문서 (PDF/DOCX/HTML/...)
        │
        ▼
┌───────────────────┐
│  Document Parsing  │  ← unstructured, pymupdf, pdfplumber
│  (레이아웃 분석)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Element 추출     │  ← Title, Text, Table, Image
│  (구조 인식)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Chunking         │  ← 섹션 기반, 의미 기반
│  (분할 전략)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Embedding        │  ← 텍스트/표/이미지 임베딩
│  (벡터화)         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Vector Store     │  ← Qdrant, Chroma, Pinecone
│  (저장 및 검색)   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  RAG Chain        │  ← LangChain, LlamaIndex
│  (질의응답)       │
└───────────────────┘
```

---

## 🛠️ 사용 기술 스택

| 분류 | 도구 | 용도 |
|------|------|------|
| 문서 파싱 | `unstructured` | 멀티포맷 문서 파싱 |
| PDF 처리 | `pymupdf`, `pdfplumber` | PDF 직접 조작 |
| OCR | `tesseract`, `paddleocr` | 이미지 내 텍스트 인식 |
| 임베딩 | `OpenAI`, `HuggingFace` | 벡터 생성 |
| 벡터 저장소 | `Qdrant`, `Chroma` | 벡터 저장 및 검색 |
| RAG 프레임워크 | `LangChain` | 체인 구성 |
| 공시 데이터 | `DART API` | 사업보고서 다운로드 |

---

## 💡 핵심 포인트

1. **비정형 데이터의 다양성**: 현실 기업 문서는 텍스트, 표, 이미지, 차트가 혼재하는 복잡한 구조를 가짐
2. **파싱 전략 선택**: 문서 특성에 따라 fast/hi_res/ocr_only 전략을 선택적으로 적용
3. **청킹의 중요성**: 단순 길이 기반 청킹이 아닌 문서 구조를 반영한 의미론적 청킹 필요
4. **메타데이터 설계**: 효율적인 필터링과 검색을 위한 메타데이터 스키마 사전 설계
5. **품질 검증**: 파싱 결과의 정확도와 완전성을 체계적으로 검증하는 절차 수립

---

## ⚠️ 주의사항

- `unstructured` 라이브러리의 `hi_res` 모드는 처리 시간이 길므로 대용량 문서에는 배치 처리 적용
- OCR 품질은 원본 이미지 해상도에 크게 의존하므로 전처리(해상도 향상, 노이즈 제거) 중요
- 표(Table) 파싱 시 복잡한 병합 셀 처리에 주의
- 사업보고서의 경우 연도별 양식 변경에 대응하는 유연한 파서 설계 필요

---

## 📦 프로젝트 4: 기업 정보 분석 에이전트

이번 주 강의는 **프로젝트 4: 기업 정보 분석 에이전트** 구현의 기반이 됩니다.

- **목표**: 상장기업 사업보고서/IR 보고서를 자동으로 파싱하여 질의응답이 가능한 지능형 에이전트 구축
- **데이터**: DART(전자공시시스템) API를 통한 실제 사업보고서
- **기술**: unstructured + DART API + LangChain + Qdrant
- **결과물**: 기업 재무 현황, 사업 전략, 리스크 요인 등을 분석하는 RAG 에이전트

---

## 🔍 사전 준비

```bash
# 필수 패키지 설치
pip install unstructured[all-docs]
pip install pymupdf pdfplumber
pip install langchain langchain-openai langchain-community
pip install qdrant-client
pip install pytesseract paddlepaddle paddleocr

# 시스템 의존성 (Ubuntu)
apt-get install tesseract-ocr
apt-get install poppler-utils
apt-get install libmagic1
```

---

## 📖 참고 자료

- [unstructured 공식 문서](https://docs.unstructured.io/)
- [DART OpenAPI 가이드](https://opendart.fss.or.kr/guide/main.do)
- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
