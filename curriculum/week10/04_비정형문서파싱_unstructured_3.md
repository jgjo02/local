# 4차시: 사업보고서 특화 파싱

## 학습 목표
- 사업보고서 구조를 이해하고 섹션별 파싱 전략을 적용할 수 있다
- 재무제표 표 데이터를 구조화할 수 있다
- 대용량 문서 배치 처리를 구현할 수 있다

---

## 1. 사업보고서 구조

```
사업보고서 (DART) 주요 섹션:
  I.   회사의 개요
  II.  사업의 내용 (주력 사업, 시장 현황)
  III. 재무에 관한 사항
         └── 재무상태표, 손익계산서, 현금흐름표
  IV.  감사인의 감사의견
  V.   이사회 등 회사의 기관
```

---

## 2. 섹션별 파싱 전략

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, NarrativeText, Table

def parse_business_report(pdf_path: str) -> dict:
    """사업보고서 섹션별 파싱"""
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True
    )

    sections = {}
    current_section = "기타"

    for element in elements:
        if element.category == "Title":
            title_text = str(element)
            # 주요 섹션 감지
            if any(kw in title_text for kw in ["사업의 내용", "재무", "감사"]):
                current_section = title_text[:50]
                sections.setdefault(current_section, [])

        if current_section in sections:
            sections[current_section].append({
                "type": element.category,
                "content": str(element)[:500],
                "page": element.metadata.page_number
            })

    return sections
```

---

## 3. 재무제표 처리

```python
import pandas as pd
from bs4 import BeautifulSoup

def extract_financial_data(table_html: str) -> pd.DataFrame:
    """HTML 표를 DataFrame으로 변환"""
    soup = BeautifulSoup(table_html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if row:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
    return pd.DataFrame()

def financial_table_to_text(df: pd.DataFrame) -> str:
    """DataFrame을 RAG용 텍스트로 변환"""
    lines = []
    for _, row in df.iterrows():
        items = [f"{col}: {val}" for col, val in row.items() if val.strip()]
        lines.append(" | ".join(items))
    return "\n".join(lines)
```

---

## 4. 배치 처리

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_report(pdf_path: str) -> list:
    """단일 보고서 처리"""
    elements = partition_pdf(filename=pdf_path, strategy="fast")
    return elements_to_documents(elements)

def batch_process_reports(pdf_dir: str, n_workers: int = 4) -> list:
    """여러 보고서 병렬 처리"""
    pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
    all_docs = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_report, pdf_files),
            total=len(pdf_files),
            desc="보고서 처리 중"
        ))

    for docs in results:
        all_docs.extend(docs)

    return all_docs
```

---

## 💡 핵심 포인트
- 사업보고서는 섹션별로 다른 파싱 전략 적용 (텍스트 vs 표 vs 이미지)
- 재무제표는 HTML → DataFrame → 텍스트 변환으로 RAG에 적합한 형식으로 변환
- 대용량 문서 처리 시 `ProcessPoolExecutor`로 병렬화

## ❓ 차시별 질문
1. 동일 페이지에 텍스트와 표가 섞여 있을 때 어떻게 처리할까요?
2. 배치 처리 시 OOM(Out of Memory) 방지 방법은?
