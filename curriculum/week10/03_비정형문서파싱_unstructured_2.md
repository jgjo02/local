# 3차시: unstructured 고급 활용

## 학습 목표
- 표 구조를 추출하고 마크다운으로 변환할 수 있다
- chunk_by_title로 섹션 기반 청킹을 구현할 수 있다
- OCR로 이미지 내 텍스트를 추출할 수 있다

---

## 1. 표(Table) 추출

```python
from unstructured.partition.pdf import partition_pdf

# 표 구조 인식 활성화
elements = partition_pdf(
    filename="business_report.pdf",
    strategy="hi_res",
    infer_table_structure=True,   # 표 구조 추론
    extract_images_in_pdf=False
)

# 표 요소 필터링
tables = [e for e in elements if e.category == "Table"]

for table in tables:
    print(f"표 제목: {table.metadata.text_as_html[:200]}")
    # HTML 형식으로 표 데이터 접근 가능
```

---

## 2. chunk_by_title 전략

```python
from unstructured.chunking.title import chunk_by_title

# 섹션 제목 기준으로 청킹
chunks = chunk_by_title(
    elements=elements,
    max_characters=2000,      # 최대 청크 크기
    new_after_n_chars=1800,   # 새 청크 시작 임계값
    combine_text_under_n_chars=500  # 작은 요소 결합
)

print(f"섹션 기반 청크 수: {len(chunks)}")
for chunk in chunks[:3]:
    print(f"  [{chunk.category}] {str(chunk)[:100]}")
```

---

## 3. OCR 텍스트 추출

```python
# OCR 전략 사용
elements = partition_pdf(
    filename="scanned_document.pdf",
    strategy="ocr_only",  # OCR 전용
    languages=["kor", "eng"]  # 한국어 + 영어
)

# 이미지에서 텍스트 추출
from unstructured.partition.image import partition_image

img_elements = partition_image(
    filename="chart.png",
    strategy="hi_res",
    languages=["kor"]
)
text = "\n".join(str(e) for e in img_elements)
```

---

## 4. 메타데이터 활용

```python
from langchain_core.documents import Document

def elements_to_documents(elements) -> list[Document]:
    """unstructured 요소를 LangChain Document로 변환"""
    docs = []
    for element in elements:
        if element.category in ["NarrativeText", "Title", "ListItem", "Table"]:
            doc = Document(
                page_content=str(element),
                metadata={
                    "source": element.metadata.filename,
                    "page_number": element.metadata.page_number,
                    "category": element.category,
                    "section": getattr(element.metadata, "section", ""),
                }
            )
            docs.append(doc)
    return docs
```

---

## 💡 핵심 포인트
- `infer_table_structure=True`로 표를 HTML 형식으로 추출 가능
- `chunk_by_title`은 문서 계층 구조를 유지하면서 청킹
- OCR은 느리므로 실제 텍스트가 있는 PDF에는 `fast` 전략 권장

## ❓ 차시별 질문
1. 표 데이터를 LLM이 이해하기 좋은 텍스트로 어떻게 변환할까요?
2. OCR 정확도가 낮을 때 어떻게 품질을 개선할 수 있나요?
