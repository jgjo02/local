# 📄 2차시: 비정형 문서 파싱 - unstructured 라이브러리 활용 1

## 🎯 학습 목표

- `unstructured` 라이브러리의 아키텍처와 핵심 개념을 이해한다
- Document Elements의 각 타입과 용도를 구별할 수 있다
- `partition()` 함수를 활용하여 다양한 문서를 파싱할 수 있다
- PDF 파싱 전략(fast/hi_res/ocr_only)을 상황에 맞게 선택할 수 있다
- 텍스트, 표, 이미지 요소를 추출하고 메타데이터를 보존할 수 있다

---

## 1. 📦 unstructured 라이브러리 소개

### 1.1 라이브러리 개요

`unstructured`는 Unstructured.io가 개발한 오픈소스 문서 전처리 라이브러리입니다. LLM 파이프라인에 공급할 비정형 데이터를 표준화된 형식으로 변환하는 것을 목표로 합니다.

```
unstructured 라이브러리 위치

  원본 문서                         LLM/RAG 시스템
  ┌──────────┐                    ┌──────────────┐
  │ PDF      │                    │              │
  │ DOCX     │──► unstructured ──►│ LangChain    │
  │ HTML     │    (전처리 허브)    │ LlamaIndex   │
  │ PPTX     │                    │ Custom RAG   │
  │ XLSX     │                    │              │
  └──────────┘                    └──────────────┘

  역할: 다양한 형식 → 일관된 Document Elements
```

### 1.2 설치 방법

```bash
# 기본 설치 (PDF, DOCX, HTML 지원)
pip install unstructured

# 전체 설치 (모든 파일 형식 지원)
pip install "unstructured[all-docs]"

# 파일 형식별 선택 설치
pip install "unstructured[pdf]"       # PDF만
pip install "unstructured[docx]"      # Word 문서
pip install "unstructured[pptx]"      # PowerPoint
pip install "unstructured[xlsx]"      # Excel

# 시스템 의존성 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-kor \
    poppler-utils \
    libmagic1 \
    libreoffice

# 고해상도 파싱을 위한 추가 패키지
pip install "unstructured-inference"
pip install "unstructured[local-inference]"
```

### 1.3 지원 파일 형식

```
지원 파일 형식 완전 목록
╔══════════════════╦══════════════════╦══════════════════╗
║  문서 형식       ║  웹/마크업       ║  데이터 형식     ║
╠══════════════════╬══════════════════╬══════════════════╣
║  PDF (.pdf)      ║  HTML (.html)    ║  CSV (.csv)      ║
║  Word (.docx)    ║  Markdown (.md)  ║  Excel (.xlsx)   ║
║  PowerPoint      ║  RST (.rst)      ║  TSV (.tsv)      ║
║  (.pptx)         ║  XML (.xml)      ║                  ║
║  Word 97 (.doc)  ╠══════════════════╣                  ║
║  RTF (.rtf)      ║  이메일 형식     ║  코드 파일       ║
║  ODT (.odt)      ║  EML (.eml)      ║  .py, .js, .java ║
║  ODP (.odp)      ║  MSG (.msg)      ║  .cpp, .go 등    ║
╚══════════════════╩══════════════════╩══════════════════╝
```

---

## 2. 🧩 Document Elements 타입 이해

### 2.1 Element 타입 계층 구조

```
Document Element 계층 구조
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Element (기본 클래스)
│
├── Text (텍스트 기반)
│   ├── Title            ← 제목/소제목
│   ├── NarrativeText    ← 서술형 본문
│   ├── ListItem         ← 목록 항목
│   ├── Header           ← 페이지 헤더
│   ├── Footer           ← 페이지 푸터
│   ├── Address          ← 주소 정보
│   └── EmailAddress     ← 이메일 주소
│
├── Table                ← 표 데이터
│   └── (HTML + 텍스트 표현)
│
├── Image                ← 이미지 데이터
│   └── (base64 또는 파일 경로)
│
├── Formula              ← 수식
│
├── FigureCaption        ← 그림 설명
│
├── PageBreak            ← 페이지 구분자
│
└── CodeSnippet          ← 코드 블록
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2.2 주요 Element 타입 상세

**Title**
- 문서의 제목, 섹션 제목, 소제목
- 문서 계층 구조 파악의 핵심 요소
- `metadata.category_depth`로 제목 레벨 확인

**NarrativeText**
- 연속적인 서술형 텍스트 (보통 2문장 이상)
- 단일 짧은 문장은 `Text`로 분류될 수도 있음
- 가장 많은 내용 정보를 담는 타입

**Table**
- HTML 형식으로 표 구조 보존
- `metadata.text_as_html`에 HTML 표현 저장
- 병합 셀 정보 포함

**ListItem**
- 불릿 포인트, 번호 목록 항목
- `metadata.parent_id`로 목록 계층 파악

**Image**
- `metadata.image_base64`에 이미지 데이터
- `metadata.image_mime_type`에 MIME 타입

### 2.3 Element 메타데이터 구조

```python
# Element 메타데이터 구조 예시
{
    "element_id": "a8d3f1b2...",    # 고유 ID
    "type": "Table",                # Element 타입
    "text": "구분 2023 2022...",    # 텍스트 내용
    "metadata": {
        "filename": "report.pdf",   # 파일명
        "file_directory": "/data",  # 디렉토리
        "last_modified": "2024-01", # 수정일
        "filetype": "application/pdf",
        "page_number": 45,          # 페이지 번호
        "coordinates": {            # 위치 좌표
            "points": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
            "system": "PixelSpace",
            "layout_width": 1700,
            "layout_height": 2200
        },
        "text_as_html": "<table>...</table>",  # 표의 HTML
        "parent_id": "b9e4c2d1...", # 부모 Element ID
        "category_depth": 1,        # 제목 레벨 (Title일 때)
        "languages": ["kor", "eng"], # 감지된 언어
        "image_base64": "iVBOR...", # 이미지 데이터 (Image일 때)
        "image_mime_type": "image/png"
    }
}
```

---

## 3. 🔧 partition() 함수 기본 사용

### 3.1 partition() 함수 개요

`partition()` 함수는 파일 형식을 자동 감지하여 적절한 파서를 호출하는 통합 함수입니다.

```python
from unstructured.partition.auto import partition

# 파일 경로로 파싱
elements = partition(filename="document.pdf")

# URL로 파싱
elements = partition(url="https://example.com/report.pdf")

# 파일 객체로 파싱
with open("document.pdf", "rb") as f:
    elements = partition(file=f, content_type="application/pdf")
```

### 3.2 형식별 파티션 함수

```python
# 형식별 전용 함수 (더 많은 옵션 지원)
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.image import partition_image
```

### 3.3 파싱 결과 탐색

```python
elements = partition(filename="report.pdf")

# 기본 정보 출력
print(f"총 Element 수: {len(elements)}")

# 타입별 분류
from collections import Counter
type_counts = Counter(type(el).__name__ for el in elements)
print("Element 타입 분포:", type_counts)

# 각 Element 탐색
for i, element in enumerate(elements[:5]):
    print(f"\n[Element {i+1}]")
    print(f"  타입: {type(element).__name__}")
    print(f"  내용: {str(element)[:100]}")
    print(f"  페이지: {element.metadata.page_number}")
```

---

## 4. 📄 PDF 파싱 전략

### 4.1 세 가지 파싱 전략 비교

```
PDF 파싱 전략 비교
═══════════════════════════════════════════════════════════
전략           속도    정확도   OCR     레이아웃   사용 케이스
───────────────────────────────────────────────────────────
fast           ★★★★★   ★★★     ✗       기본        텍스트 PDF
hi_res         ★★      ★★★★★   선택    정밀        복잡한 PDF
ocr_only       ★★★     ★★★★    ✓(강제) -          스캔 PDF
═══════════════════════════════════════════════════════════

fast:    pdfminer 기반 텍스트 레이어 직접 추출
hi_res:  detectron2 레이아웃 분석 + 표/이미지 정밀 처리
ocr_only: 모든 페이지를 이미지로 변환 후 OCR 적용
```

### 4.2 fast 전략

```python
from unstructured.partition.pdf import partition_pdf

# fast 전략 (기본값)
elements = partition_pdf(
    filename="report.pdf",
    strategy="fast",                    # 빠른 텍스트 추출
    include_page_breaks=True,           # 페이지 구분자 포함
    include_metadata=True,              # 메타데이터 포함
)
```

**fast 전략 특징**:
- pdfminer를 사용하여 텍스트 레이어 직접 추출
- 처리 속도가 가장 빠름 (100페이지 약 2-5초)
- 이미지 기반 PDF에서는 텍스트 추출 불가
- 복잡한 레이아웃 처리 능력 제한

### 4.3 hi_res 전략

```python
# hi_res 전략 (고정밀)
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",
    hi_res_model_name="yolox",          # 레이아웃 분석 모델
    infer_table_structure=True,         # 표 구조 분석
    extract_images_in_pdf=True,         # 이미지 추출
    extract_image_block_types=["Image", "Table"],  # 추출할 블록 타입
    extract_image_block_output_dir="/output/images",  # 이미지 저장 경로
    languages=["kor", "eng"],           # OCR 언어
)
```

**hi_res 전략 특징**:
- `detectron2` 또는 `yolox` 모델로 레이아웃 분석
- 표, 이미지, 텍스트 블록 정밀 경계 감지
- 처리 시간이 상대적으로 길음 (100페이지 약 30-120초)
- GPU 사용 시 대폭 속도 향상 가능

### 4.4 ocr_only 전략

```python
# ocr_only 전략 (스캔 문서용)
elements = partition_pdf(
    filename="scanned_report.pdf",
    strategy="ocr_only",
    ocr_languages="kor+eng",            # Tesseract 언어 코드
    ocr_mode="entire_page",             # 페이지 전체 OCR
)
```

**ocr_only 전략 특징**:
- 모든 페이지를 이미지로 변환 후 Tesseract OCR 적용
- 스캔된 문서, 이미지 기반 PDF에 적합
- 텍스트 레이어가 있어도 OCR 우선 적용
- 한국어 처리를 위해 `kor` 언어팩 필요

---

## 5. 📝 텍스트 요소 추출

### 5.1 Title 요소 추출

```python
from unstructured.documents.elements import Title, NarrativeText, ListItem
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(filename="report.pdf", strategy="fast")

# Title 요소만 추출
titles = [el for el in elements if isinstance(el, Title)]

print("문서 목차 구조:")
for title in titles:
    depth = title.metadata.category_depth or 0
    indent = "  " * depth
    print(f"{indent}{'#' * (depth + 1)} {title.text}")
```

### 5.2 NarrativeText 요소 처리

```python
# NarrativeText 추출 및 처리
narratives = [el for el in elements if isinstance(el, NarrativeText)]

# 짧은 텍스트 필터링 (의미 없는 단편 제거)
meaningful_texts = [
    el for el in narratives
    if len(el.text.split()) >= 10  # 10단어 이상
]

print(f"전체 서술 텍스트: {len(narratives)}개")
print(f"의미 있는 텍스트: {len(meaningful_texts)}개")

# 특정 섹션 텍스트 추출 (메타데이터 활용)
section_texts = {}
for el in meaningful_texts:
    page = el.metadata.page_number
    if page not in section_texts:
        section_texts[page] = []
    section_texts[page].append(el.text)
```

### 5.3 ListItem 처리

```python
# 목록 항목 추출 및 계층 구조 재구성
list_items = [el for el in elements if isinstance(el, ListItem)]

# 부모-자식 관계 재구성
def build_list_hierarchy(list_items):
    hierarchy = {}
    for item in list_items:
        parent_id = item.metadata.parent_id
        if parent_id not in hierarchy:
            hierarchy[parent_id] = []
        hierarchy[parent_id].append(item.text)
    return hierarchy

hierarchy = build_list_hierarchy(list_items)
print("목록 계층 구조:", hierarchy)
```

---

## 6. 📊 표(Table) 요소 추출

### 6.1 표 기본 추출

```python
from unstructured.documents.elements import Table

# 표 요소만 추출
tables = [el for el in elements if isinstance(el, Table)]

print(f"발견된 표 수: {len(tables)}")

for i, table in enumerate(tables):
    print(f"\n[표 {i+1}] (페이지 {table.metadata.page_number})")
    print(f"텍스트 요약: {table.text[:200]}")

    # HTML 형식으로 표 구조 확인
    if hasattr(table.metadata, 'text_as_html') and table.metadata.text_as_html:
        print(f"HTML 길이: {len(table.metadata.text_as_html)}자")
```

### 6.2 표 HTML을 DataFrame으로 변환

```python
import pandas as pd
from io import StringIO

def table_element_to_dataframe(table_element: Table) -> pd.DataFrame:
    """
    Table Element를 pandas DataFrame으로 변환

    Args:
        table_element: unstructured Table Element

    Returns:
        pd.DataFrame 또는 None (변환 실패 시)
    """
    if not hasattr(table_element.metadata, 'text_as_html'):
        return None

    html = table_element.metadata.text_as_html
    if not html:
        return None

    try:
        dfs = pd.read_html(StringIO(html))
        return dfs[0] if dfs else None
    except Exception as e:
        print(f"DataFrame 변환 실패: {e}")
        return None

# 모든 표를 DataFrame으로 변환
dataframes = []
for table in tables:
    df = table_element_to_dataframe(table)
    if df is not None:
        dataframes.append({
            "page": table.metadata.page_number,
            "dataframe": df
        })

print(f"DataFrame 변환 성공: {len(dataframes)}개")
```

### 6.3 표 품질 검증

```python
def validate_table_quality(table: Table) -> dict:
    """표 품질 검증"""
    quality = {
        "has_html": False,
        "has_headers": False,
        "row_count": 0,
        "col_count": 0,
        "empty_cell_ratio": 0.0,
        "quality_score": 0.0
    }

    if hasattr(table.metadata, 'text_as_html') and table.metadata.text_as_html:
        quality["has_html"] = True
        df = table_element_to_dataframe(table)

        if df is not None:
            quality["row_count"] = len(df)
            quality["col_count"] = len(df.columns)

            # 빈 셀 비율
            total_cells = df.size
            empty_cells = df.isnull().sum().sum()
            quality["empty_cell_ratio"] = empty_cells / total_cells if total_cells > 0 else 1.0

            # 헤더 존재 여부 (첫 행이 텍스트인지 확인)
            quality["has_headers"] = not df.columns.str.startswith('Unnamed').all()

            # 품질 점수 계산
            score = 0.0
            if quality["has_html"]: score += 0.3
            if quality["has_headers"]: score += 0.2
            if quality["row_count"] > 1: score += 0.2
            if quality["col_count"] > 1: score += 0.2
            if quality["empty_cell_ratio"] < 0.3: score += 0.1
            quality["quality_score"] = score

    return quality
```

---

## 7. 🖼️ 이미지 요소 추출

### 7.1 이미지 추출 설정

```python
# 이미지 추출 활성화 (hi_res 전략 필요)
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",
    extract_images_in_pdf=True,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_output_dir="./extracted_images",
    extract_image_block_to_payload=True,  # base64로 페이로드에 포함
)
```

### 7.2 이미지 Element 처리

```python
from unstructured.documents.elements import Image

# 이미지 요소 추출
images = [el for el in elements if isinstance(el, Image)]

print(f"추출된 이미지 수: {len(images)}")

for i, img in enumerate(images):
    print(f"\n[이미지 {i+1}]")
    print(f"  페이지: {img.metadata.page_number}")
    print(f"  설명: {img.text[:100] if img.text else '없음'}")

    if hasattr(img.metadata, 'image_base64') and img.metadata.image_base64:
        import base64
        from PIL import Image as PILImage
        from io import BytesIO

        # base64 → PIL Image
        img_bytes = base64.b64decode(img.metadata.image_base64)
        pil_image = PILImage.open(BytesIO(img_bytes))
        print(f"  크기: {pil_image.size}")
        print(f"  형식: {pil_image.format}")

        # 이미지 저장
        pil_image.save(f"./output/image_{i+1}.png")
```

---

## 8. 🏷️ 메타데이터 보존

### 8.1 메타데이터 스키마 확장

```python
def enrich_element_metadata(element, source_info: dict) -> dict:
    """
    Element에 추가 메타데이터를 결합하여 반환

    Args:
        element: unstructured Element
        source_info: 추가할 소스 정보 딕셔너리

    Returns:
        풍부한 메타데이터를 포함한 딕셔너리
    """
    return {
        "element_id": element.id,
        "element_type": type(element).__name__,
        "text": element.text,
        # 원본 메타데이터
        "page_number": element.metadata.page_number,
        "filename": element.metadata.filename,
        "file_directory": element.metadata.file_directory,
        # HTML 표현 (표인 경우)
        "text_as_html": getattr(element.metadata, 'text_as_html', None),
        # 좌표 정보
        "coordinates": element.metadata.coordinates.to_dict()
                      if element.metadata.coordinates else None,
        # 언어 정보
        "languages": element.metadata.languages,
        # 추가 소스 정보
        **source_info
    }

# 예시 사용
source_info = {
    "company": "삼성전자",
    "report_type": "사업보고서",
    "fiscal_year": 2023,
    "dart_code": "005930"
}

enriched_elements = [
    enrich_element_metadata(el, source_info)
    for el in elements
]
```

### 8.2 메타데이터 활용 필터링

```python
# 특정 페이지 범위의 요소만 추출
def filter_by_page(elements, start_page: int, end_page: int):
    return [
        el for el in elements
        if el.metadata.page_number
        and start_page <= el.metadata.page_number <= end_page
    ]

# 재무제표 섹션 (예: 45-80 페이지)
financial_elements = filter_by_page(elements, 45, 80)

# 표만 필터링하면서 메타데이터로 검색
def find_tables_with_keyword(elements, keyword: str):
    return [
        el for el in elements
        if isinstance(el, Table) and keyword in el.text
    ]

# 재무상태표 찾기
balance_sheets = find_tables_with_keyword(elements, "재무상태표")
income_statements = find_tables_with_keyword(elements, "손익계산서")
```

---

## 9. 📐 개념 다이어그램: Element 타입 분류 구조

```
unstructured Document Element 타입 완전 분류도
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

입력 PDF 페이지
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  III. 재무에 관한 사항                                │   │ ← Title
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. 연결재무제표                                      │   │ ← Title
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  당사의 연결재무제표는 한국채택국제회계기준(K-IFRS)에  │   │
│  │  따라 작성되었으며, 비교 목적으로 전기 재무제표와 함께 │   │ ← NarrativeText
│  │  제시됩니다.                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ● 2023년 매출액: 258.9조원                           │   │
│  │  ● 2023년 영업이익: 6.6조원                           │   │ ← ListItem(×2)
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  구분     │ 2023     │ 2022     │ 증감율              │   │
│  │  ─────────┼──────────┼──────────┼──────────          │   │ ← Table
│  │  매출액   │ 258.9조  │ 302.2조  │ -14.3%             │   │
│  │  영업이익 │ 6.6조    │ 43.4조   │ -84.8%             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  [매출 추이 그래프 이미지]                             │   │ ← Image
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ──────────────────── 페이지 45 ─────────────────────────   │ ← PageBreak
└─────────────────────────────────────────────────────────────┘

파싱 결과:
┌──────────────────────────────────────────────────────────────┐
│  elements = [                                                │
│    Title("III. 재무에 관한 사항", page=45, depth=0),          │
│    Title("1. 연결재무제표", page=45, depth=1),                │
│    NarrativeText("당사의 연결재무제표는...", page=45),        │
│    ListItem("2023년 매출액: 258.9조원", page=45),             │
│    ListItem("2023년 영업이익: 6.6조원", page=45),             │
│    Table("구분 2023 2022 증감율...", page=45,                 │
│           text_as_html="<table>...</table>"),                 │
│    Image("매출 추이 그래프", page=45, base64="iVBOR..."),     │
│    PageBreak(),                                              │
│  ]                                                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. 💻 코드 예제: 종합 파싱 워크플로우

코드 파일: `code/02_unstructured_1.py` 참조

핵심 코드 요약:
```python
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table, Image, ListItem

# 1. 파싱 실행
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    extract_images_in_pdf=True,
    languages=["kor", "eng"]
)

# 2. 타입별 분류
element_map = {
    "titles": [e for e in elements if isinstance(e, Title)],
    "texts": [e for e in elements if isinstance(e, NarrativeText)],
    "tables": [e for e in elements if isinstance(e, Table)],
    "images": [e for e in elements if isinstance(e, Image)],
    "lists": [e for e in elements if isinstance(e, ListItem)],
}

# 3. 결과 저장
import json
output = []
for el in elements:
    output.append({
        "type": type(el).__name__,
        "text": el.text,
        "page": el.metadata.page_number,
        "html": getattr(el.metadata, 'text_as_html', None)
    })

with open("parsed_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
```

---

## 💡 핵심 포인트

1. **전략 선택의 중요성**: 사업보고서는 텍스트/이미지/표가 혼재하므로 `hi_res` 전략이 권장되나, 처리 시간을 고려한 배치 처리 필요
2. **메타데이터 보존**: 페이지 번호, 좌표 정보는 나중에 출처 표시와 문서 구조 재구성에 필수적
3. **표의 HTML 보존**: `text_as_html`을 통해 표 구조를 보존하면 재무 데이터 처리 시 정확도 향상
4. **언어 설정**: 한국어 문서 처리 시 반드시 `languages=["kor"]` 또는 `languages=["kor", "eng"]` 설정
5. **Element ID 활용**: `element_id`를 메타데이터로 저장하면 청킹 후에도 원본 Element 추적 가능

## ⚠️ 주의사항

- `hi_res` 전략은 `detectron2` 또는 `unstructured-inference` 패키지가 필요하며, GPU 환경에서의 성능이 CPU 대비 10배 이상 우수
- 이미지 추출(`extract_images_in_pdf=True`) 시 `extract_image_block_output_dir` 경로가 사전에 생성되어 있어야 함
- 대용량 PDF(100페이지 이상)는 메모리 부족 문제 발생 가능 → 페이지 범위 분할 처리 고려
- Tesseract OCR의 한국어 정확도를 높이려면 `tesseract-ocr-kor` 언어팩 설치 필수

## 🔍 심화학습

- `unstructured` 소스 코드: [GitHub](https://github.com/Unstructured-IO/unstructured)
- Document Layout Analysis: [PubLayNet](https://arxiv.org/abs/1908.07490)
- Table Detection with Deep Learning: [TableNet](https://arxiv.org/abs/2001.01469)
- 한국어 OCR 성능 개선: [PaddleOCR Korean](https://github.com/PaddlePaddle/PaddleOCR)

## ❓ 차시별 질문

1. **전략 선택**: 100페이지 분량의 일반 텍스트 PDF 사업보고서를 파싱할 때 어떤 전략을 선택하겠습니까? 그 이유는?

2. **메타데이터 활용**: `element_id`와 `parent_id` 메타데이터를 활용하여 문서의 계층 구조(목차 트리)를 어떻게 재구성할 수 있을까요?

3. **표 처리**: `text_as_html`로 저장된 복잡한 재무제표를 LLM이 이해하기 좋은 형식으로 변환하는 방법을 설명하세요.

4. **품질 검증**: 파싱된 표에서 빈 셀 비율이 50%를 초과하는 경우 어떤 대처 방법이 있을까요?

5. **이미지 처리**: 추출된 이미지 Element를 RAG 시스템에 통합하기 위해 어떤 후처리 단계가 필요할까요?
