"""
Week 10 - 2차시: unstructured 라이브러리 활용 1
비정형 문서 파싱 기초 - 텍스트, 표, 이미지 요소 추출

필요 패키지:
    pip install "unstructured[all-docs]"
    pip install pymupdf pandas pillow
    apt-get install tesseract-ocr tesseract-ocr-kor poppler-utils
"""

import json
import base64
import logging
from pathlib import Path
from io import BytesIO, StringIO
from collections import Counter
from typing import List, Dict, Optional, Any

# unstructured 핵심 임포트
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem,
    Table, Image, Header, Footer,
    PageBreak, Element
)

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 기본 파싱 함수
# ──────────────────────────────────────────────

def parse_pdf_fast(filepath: str) -> List[Element]:
    """
    fast 전략으로 PDF 파싱 (빠른 텍스트 추출)

    Args:
        filepath: PDF 파일 경로

    Returns:
        파싱된 Element 리스트
    """
    logger.info(f"[fast] 파싱 시작: {filepath}")
    elements = partition_pdf(
        filename=filepath,
        strategy="fast",
        include_page_breaks=True,
        include_metadata=True,
        languages=["kor", "eng"],
    )
    logger.info(f"[fast] 완료: {len(elements)}개 Element 추출")
    return elements


def parse_pdf_hires(
    filepath: str,
    output_image_dir: Optional[str] = None,
    extract_images: bool = True
) -> List[Element]:
    """
    hi_res 전략으로 PDF 파싱 (고정밀 레이아웃 분석)

    Args:
        filepath: PDF 파일 경로
        output_image_dir: 이미지 저장 디렉토리 (None이면 저장하지 않음)
        extract_images: 이미지 추출 여부

    Returns:
        파싱된 Element 리스트
    """
    logger.info(f"[hi_res] 파싱 시작: {filepath}")

    kwargs = {
        "filename": filepath,
        "strategy": "hi_res",
        "hi_res_model_name": "yolox",
        "infer_table_structure": True,
        "include_page_breaks": True,
        "include_metadata": True,
        "languages": ["kor", "eng"],
    }

    if extract_images:
        kwargs["extract_images_in_pdf"] = True
        kwargs["extract_image_block_types"] = ["Image", "Table"]
        kwargs["extract_image_block_to_payload"] = True

        if output_image_dir:
            Path(output_image_dir).mkdir(parents=True, exist_ok=True)
            kwargs["extract_image_block_output_dir"] = output_image_dir

    elements = partition_pdf(**kwargs)
    logger.info(f"[hi_res] 완료: {len(elements)}개 Element 추출")
    return elements


def parse_pdf_ocr(filepath: str) -> List[Element]:
    """
    ocr_only 전략으로 PDF 파싱 (스캔 문서용)

    Args:
        filepath: PDF 파일 경로

    Returns:
        파싱된 Element 리스트
    """
    logger.info(f"[ocr_only] 파싱 시작: {filepath}")
    elements = partition_pdf(
        filename=filepath,
        strategy="ocr_only",
        ocr_languages="kor+eng",
        ocr_mode="entire_page",
        include_page_breaks=True,
    )
    logger.info(f"[ocr_only] 완료: {len(elements)}개 Element 추출")
    return elements


def parse_document_auto(filepath: str) -> List[Element]:
    """
    파일 형식 자동 감지 후 파싱 (PDF, DOCX, HTML 등)

    Args:
        filepath: 문서 파일 경로

    Returns:
        파싱된 Element 리스트
    """
    file_ext = Path(filepath).suffix.lower()
    logger.info(f"파일 형식: {file_ext}, 파싱 시작: {filepath}")

    elements = partition(
        filename=filepath,
        include_metadata=True,
    )
    logger.info(f"자동 파싱 완료: {len(elements)}개 Element 추출")
    return elements


# ──────────────────────────────────────────────
# 2. Element 분류 및 통계
# ──────────────────────────────────────────────

def classify_elements(elements: List[Element]) -> Dict[str, List[Element]]:
    """
    Element를 타입별로 분류

    Args:
        elements: Element 리스트

    Returns:
        타입별 분류된 딕셔너리
    """
    return {
        "titles": [e for e in elements if isinstance(e, Title)],
        "narratives": [e for e in elements if isinstance(e, NarrativeText)],
        "list_items": [e for e in elements if isinstance(e, ListItem)],
        "tables": [e for e in elements if isinstance(e, Table)],
        "images": [e for e in elements if isinstance(e, Image)],
        "headers": [e for e in elements if isinstance(e, Header)],
        "footers": [e for e in elements if isinstance(e, Footer)],
        "page_breaks": [e for e in elements if isinstance(e, PageBreak)],
        "others": [
            e for e in elements
            if not isinstance(e, (Title, NarrativeText, ListItem,
                                   Table, Image, Header, Footer, PageBreak))
        ]
    }


def print_element_statistics(elements: List[Element]) -> None:
    """Element 통계 정보 출력"""
    classified = classify_elements(elements)

    print("\n" + "=" * 60)
    print("문서 파싱 통계")
    print("=" * 60)
    print(f"총 Element 수: {len(elements)}")
    print()

    type_counts = Counter(type(e).__name__ for e in elements)
    print("Element 타입별 분포:")
    for elem_type, count in type_counts.most_common():
        bar = "█" * min(count, 50)
        print(f"  {elem_type:20s}: {count:4d} {bar}")

    # 페이지별 분포
    page_counts = Counter(
        e.metadata.page_number
        for e in elements
        if e.metadata.page_number
    )
    if page_counts:
        print(f"\n총 페이지 수: {max(page_counts.keys())}")
        print(f"평균 Elements/페이지: {len(elements) / max(page_counts.keys()):.1f}")

    print("=" * 60)


# ──────────────────────────────────────────────
# 3. 텍스트 요소 처리
# ──────────────────────────────────────────────

def extract_document_outline(elements: List[Element]) -> List[Dict]:
    """
    제목 요소에서 문서 목차 추출

    Args:
        elements: Element 리스트

    Returns:
        목차 구조 (딕셔너리 리스트)
    """
    outline = []
    titles = [e for e in elements if isinstance(e, Title)]

    for title in titles:
        depth = getattr(title.metadata, 'category_depth', 0) or 0
        outline.append({
            "level": depth,
            "text": title.text,
            "page": title.metadata.page_number,
            "id": title.id
        })

    return outline


def print_document_outline(elements: List[Element]) -> None:
    """문서 목차 구조 출력"""
    outline = extract_document_outline(elements)

    print("\n문서 목차:")
    print("-" * 50)
    for item in outline:
        indent = "  " * item["level"]
        marker = "#" * (item["level"] + 1)
        page_info = f"[p.{item['page']}]" if item['page'] else ""
        print(f"{indent}{marker} {item['text']} {page_info}")


def get_text_by_section(elements: List[Element]) -> Dict[str, List[str]]:
    """
    섹션(제목) 기준으로 텍스트 그룹화

    Args:
        elements: Element 리스트

    Returns:
        섹션별 텍스트 딕셔너리
    """
    sections = {}
    current_section = "서문"

    for element in elements:
        if isinstance(element, Title):
            current_section = element.text
            if current_section not in sections:
                sections[current_section] = []
        elif isinstance(element, (NarrativeText, ListItem)):
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(element.text)

    return sections


# ──────────────────────────────────────────────
# 4. 표(Table) 처리
# ──────────────────────────────────────────────

def table_to_dataframe(table: Table) -> Optional[pd.DataFrame]:
    """
    Table Element를 pandas DataFrame으로 변환

    Args:
        table: Table Element

    Returns:
        DataFrame 또는 None
    """
    html = getattr(table.metadata, 'text_as_html', None)
    if not html:
        return None

    try:
        dfs = pd.read_html(StringIO(html))
        return dfs[0] if dfs else None
    except Exception as e:
        logger.warning(f"DataFrame 변환 실패: {e}")
        return None


def extract_all_tables(elements: List[Element]) -> List[Dict]:
    """
    모든 Table Element를 구조화된 형태로 추출

    Args:
        elements: Element 리스트

    Returns:
        표 정보 딕셔너리 리스트
    """
    tables_data = []
    table_elements = [e for e in elements if isinstance(e, Table)]

    for i, table in enumerate(table_elements):
        df = table_to_dataframe(table)
        table_info = {
            "index": i,
            "page": table.metadata.page_number,
            "text_summary": table.text[:300],
            "html": getattr(table.metadata, 'text_as_html', None),
            "dataframe": df,
            "row_count": len(df) if df is not None else 0,
            "col_count": len(df.columns) if df is not None else 0,
        }
        tables_data.append(table_info)
        logger.info(
            f"표 {i+1}/{len(table_elements)}: "
            f"페이지={table_info['page']}, "
            f"크기={table_info['row_count']}×{table_info['col_count']}"
        )

    return tables_data


def search_financial_tables(elements: List[Element]) -> Dict[str, Optional[pd.DataFrame]]:
    """
    재무제표 관련 표를 찾아서 반환

    Args:
        elements: Element 리스트

    Returns:
        재무제표 유형별 DataFrame 딕셔너리
    """
    financial_keywords = {
        "재무상태표": ["재무상태표", "대차대조표", "Balance Sheet"],
        "손익계산서": ["손익계산서", "포괄손익", "Income Statement", "P&L"],
        "현금흐름표": ["현금흐름표", "Cash Flow"],
        "자본변동표": ["자본변동표", "자본변동"],
    }

    found_tables = {key: None for key in financial_keywords}

    # 제목과 표를 함께 탐색
    current_section = ""
    for element in elements:
        if isinstance(element, Title):
            current_section = element.text

        elif isinstance(element, Table):
            for report_type, keywords in financial_keywords.items():
                # 표 텍스트 또는 현재 섹션에 키워드 포함 여부 확인
                table_text = element.text + " " + current_section
                if any(kw in table_text for kw in keywords):
                    df = table_to_dataframe(element)
                    if df is not None and found_tables[report_type] is None:
                        found_tables[report_type] = df
                        logger.info(
                            f"'{report_type}' 발견: 페이지 {element.metadata.page_number}"
                        )
                    break

    return found_tables


def validate_table_quality(table: Table) -> Dict[str, Any]:
    """
    표 품질 검증

    Args:
        table: Table Element

    Returns:
        품질 지표 딕셔너리
    """
    quality = {
        "has_html": False,
        "has_dataframe": False,
        "row_count": 0,
        "col_count": 0,
        "empty_cell_ratio": 1.0,
        "quality_score": 0.0,
        "issues": []
    }

    html = getattr(table.metadata, 'text_as_html', None)
    if html:
        quality["has_html"] = True
    else:
        quality["issues"].append("HTML 구조 없음")

    df = table_to_dataframe(table)
    if df is not None:
        quality["has_dataframe"] = True
        quality["row_count"] = len(df)
        quality["col_count"] = len(df.columns)

        # 빈 셀 비율
        total = df.size
        empty = df.isnull().sum().sum() + (df == "").sum().sum()
        quality["empty_cell_ratio"] = float(empty / total) if total > 0 else 1.0

        if quality["row_count"] < 2:
            quality["issues"].append("행 수 부족 (2 미만)")
        if quality["col_count"] < 2:
            quality["issues"].append("열 수 부족 (2 미만)")
        if quality["empty_cell_ratio"] > 0.5:
            quality["issues"].append(f"빈 셀 비율 높음 ({quality['empty_cell_ratio']:.1%})")
    else:
        quality["issues"].append("DataFrame 변환 실패")

    # 품질 점수 계산 (0~1)
    score = 0.0
    if quality["has_html"]: score += 0.3
    if quality["has_dataframe"]: score += 0.2
    if quality["row_count"] >= 2: score += 0.2
    if quality["col_count"] >= 2: score += 0.2
    if quality["empty_cell_ratio"] <= 0.3: score += 0.1
    quality["quality_score"] = score

    return quality


# ──────────────────────────────────────────────
# 5. 이미지 처리
# ──────────────────────────────────────────────

def extract_images(elements: List[Element], output_dir: str = "./extracted_images") -> List[Dict]:
    """
    Image Element에서 이미지 데이터를 추출하고 저장

    Args:
        elements: Element 리스트
        output_dir: 이미지 저장 디렉토리

    Returns:
        이미지 정보 딕셔너리 리스트
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_elements = [e for e in elements if isinstance(e, Image)]
    extracted = []

    for i, img_el in enumerate(image_elements):
        img_info = {
            "index": i,
            "page": img_el.metadata.page_number,
            "description": img_el.text,
            "file_path": None,
            "width": None,
            "height": None,
            "mime_type": getattr(img_el.metadata, 'image_mime_type', None),
        }

        # base64 인코딩된 이미지 처리
        img_b64 = getattr(img_el.metadata, 'image_base64', None)
        if img_b64:
            try:
                img_bytes = base64.b64decode(img_b64)
                from PIL import Image as PILImage
                pil_img = PILImage.open(BytesIO(img_bytes))

                img_info["width"] = pil_img.width
                img_info["height"] = pil_img.height

                # 파일로 저장
                file_ext = "png"
                if img_info["mime_type"]:
                    mime_to_ext = {"image/jpeg": "jpg", "image/png": "png",
                                    "image/gif": "gif", "image/bmp": "bmp"}
                    file_ext = mime_to_ext.get(img_info["mime_type"], "png")

                save_path = output_path / f"image_p{img_el.metadata.page_number}_{i+1}.{file_ext}"
                pil_img.save(save_path)
                img_info["file_path"] = str(save_path)

                logger.info(
                    f"이미지 저장: {save_path} ({pil_img.width}×{pil_img.height})"
                )
            except Exception as e:
                logger.warning(f"이미지 {i+1} 처리 실패: {e}")

        extracted.append(img_info)

    return extracted


# ──────────────────────────────────────────────
# 6. 메타데이터 관리
# ──────────────────────────────────────────────

def element_to_dict(element: Element, extra_metadata: Optional[Dict] = None) -> Dict:
    """
    Element를 직렬화 가능한 딕셔너리로 변환

    Args:
        element: Element 객체
        extra_metadata: 추가 메타데이터

    Returns:
        직렬화된 딕셔너리
    """
    data = {
        "element_id": element.id,
        "element_type": type(element).__name__,
        "text": element.text,
        "metadata": {
            "page_number": element.metadata.page_number,
            "filename": element.metadata.filename,
            "file_directory": element.metadata.file_directory,
            "last_modified": str(element.metadata.last_modified)
                             if element.metadata.last_modified else None,
            "languages": element.metadata.languages,
            "text_as_html": getattr(element.metadata, 'text_as_html', None),
            "category_depth": getattr(element.metadata, 'category_depth', None),
            "parent_id": getattr(element.metadata, 'parent_id', None),
        }
    }

    # 좌표 정보 추가
    if element.metadata.coordinates:
        data["metadata"]["coordinates"] = {
            "points": element.metadata.coordinates.points,
            "system": element.metadata.coordinates.system,
            "layout_width": element.metadata.coordinates.layout_width,
            "layout_height": element.metadata.coordinates.layout_height,
        }

    # 추가 메타데이터 병합
    if extra_metadata:
        data["metadata"].update(extra_metadata)

    return data


def save_parsed_results(
    elements: List[Element],
    output_path: str,
    extra_metadata: Optional[Dict] = None
) -> None:
    """
    파싱 결과를 JSON 파일로 저장

    Args:
        elements: Element 리스트
        output_path: 출력 파일 경로
        extra_metadata: 모든 Element에 추가할 메타데이터
    """
    serialized = [
        element_to_dict(el, extra_metadata)
        for el in elements
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"파싱 결과 저장 완료: {output_path} ({len(serialized)}개 Element)")


def load_parsed_results(input_path: str) -> List[Dict]:
    """
    저장된 파싱 결과 로드

    Args:
        input_path: JSON 파일 경로

    Returns:
        딕셔너리 리스트
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"파싱 결과 로드 완료: {input_path} ({len(data)}개 Element)")
    return data


# ──────────────────────────────────────────────
# 7. 문서 분석 보고서 생성
# ──────────────────────────────────────────────

def generate_analysis_report(elements: List[Element], filepath: str) -> Dict:
    """
    파싱된 문서 분석 보고서 생성

    Args:
        elements: Element 리스트
        filepath: 원본 파일 경로

    Returns:
        분석 보고서 딕셔너리
    """
    classified = classify_elements(elements)

    # 표 품질 검증
    table_qualities = [
        validate_table_quality(t)
        for t in classified["tables"]
    ]

    # 페이지별 통계
    page_counts = Counter(
        e.metadata.page_number
        for e in elements
        if e.metadata.page_number
    )

    total_pages = max(page_counts.keys()) if page_counts else 0

    report = {
        "file": filepath,
        "total_elements": len(elements),
        "total_pages": total_pages,
        "element_distribution": {
            k: len(v) for k, v in classified.items()
        },
        "tables": {
            "total": len(classified["tables"]),
            "high_quality": sum(1 for q in table_qualities if q["quality_score"] >= 0.7),
            "avg_quality_score": (
                sum(q["quality_score"] for q in table_qualities) / len(table_qualities)
                if table_qualities else 0
            ),
        },
        "images": {
            "total": len(classified["images"]),
            "has_description": sum(
                1 for img in classified["images"]
                if img.text and len(img.text) > 10
            )
        },
        "text_stats": {
            "total_characters": sum(len(e.text) for e in elements if e.text),
            "avg_element_length": (
                sum(len(e.text) for e in elements if e.text) / len(elements)
                if elements else 0
            ),
        }
    }

    return report


# ──────────────────────────────────────────────
# 메인 실행 예제
# ──────────────────────────────────────────────

def demo_with_sample_pdf(pdf_path: str = "sample_report.pdf"):
    """
    샘플 PDF로 전체 파싱 워크플로우 시연

    실제 PDF 파일이 없는 경우 테스트용 PDF 생성 후 실행하세요:
        pip install fpdf2
        python -c "from fpdf import FPDF; pdf=FPDF(); pdf.add_page(); pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, 'Test Report', ln=True); pdf.output('sample_report.pdf')"
    """
    if not Path(pdf_path).exists():
        logger.error(f"PDF 파일을 찾을 수 없음: {pdf_path}")
        logger.info("샘플 PDF를 생성하거나 실제 PDF 경로를 지정하세요.")
        return

    print("\n" + "=" * 60)
    print("unstructured 파싱 데모 시작")
    print("=" * 60)

    # 1. fast 전략으로 빠른 파싱
    print("\n[1/4] fast 전략으로 파싱 중...")
    elements_fast = parse_pdf_fast(pdf_path)
    print_element_statistics(elements_fast)

    # 2. 문서 목차 추출
    print("\n[2/4] 문서 목차 추출...")
    print_document_outline(elements_fast)

    # 3. 섹션별 텍스트 추출
    print("\n[3/4] 섹션별 텍스트 그룹화...")
    sections = get_text_by_section(elements_fast)
    print(f"발견된 섹션 수: {len(sections)}")
    for section, texts in list(sections.items())[:3]:
        print(f"\n  [{section}]: {len(texts)}개 텍스트 블록")
        if texts:
            print(f"    첫 번째 텍스트: {texts[0][:100]}...")

    # 4. 분석 보고서 생성
    print("\n[4/4] 분석 보고서 생성...")
    report = generate_analysis_report(elements_fast, pdf_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 5. 결과 저장
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    save_parsed_results(
        elements_fast,
        str(output_dir / "parsed_elements.json"),
        extra_metadata={"source_strategy": "fast"}
    )

    print(f"\n파싱 결과 저장: {output_dir / 'parsed_elements.json'}")
    print("\n데모 완료!")


def demo_table_extraction(pdf_path: str = "sample_report.pdf"):
    """표 추출 심화 데모"""
    if not Path(pdf_path).exists():
        print("PDF 파일 없음. 데모를 건너뜁니다.")
        return

    print("\n" + "=" * 60)
    print("표 추출 심화 데모")
    print("=" * 60)

    # hi_res 전략으로 표 구조 분석
    elements = parse_pdf_hires(pdf_path, extract_images=False)

    # 모든 표 추출
    tables_data = extract_all_tables(elements)
    print(f"\n총 {len(tables_data)}개 표 발견")

    # 재무제표 찾기
    print("\n재무제표 탐색...")
    financial_tables = search_financial_tables(elements)
    for report_type, df in financial_tables.items():
        if df is not None:
            print(f"\n{report_type}:")
            print(df.head(5).to_string())
        else:
            print(f"{report_type}: 발견되지 않음")

    # 표 품질 검증
    print("\n표 품질 검증:")
    for i, table_info in enumerate(tables_data[:5]):
        # Element 객체가 필요하므로 다시 추출
        table_elements = [e for e in elements if isinstance(e, Table)]
        if i < len(table_elements):
            quality = validate_table_quality(table_elements[i])
            print(f"\n  표 {i+1} (페이지 {table_info['page']}):")
            print(f"    품질 점수: {quality['quality_score']:.2f}")
            print(f"    크기: {quality['row_count']}행 × {quality['col_count']}열")
            print(f"    빈 셀 비율: {quality['empty_cell_ratio']:.1%}")
            if quality["issues"]:
                print(f"    이슈: {', '.join(quality['issues'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="unstructured 문서 파싱 데모")
    parser.add_argument(
        "--pdf",
        default="sample_report.pdf",
        help="파싱할 PDF 파일 경로"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "tables", "full"],
        default="basic",
        help="실행 모드: basic(기본), tables(표 추출), full(전체)"
    )
    args = parser.parse_args()

    if args.mode in ("basic", "full"):
        demo_with_sample_pdf(args.pdf)

    if args.mode in ("tables", "full"):
        demo_table_extraction(args.pdf)
