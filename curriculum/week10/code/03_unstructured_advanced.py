"""Week 10 실습 3: unstructured 고급 - 표 추출 및 청크 전략"""

import os
from typing import List, Dict, Any

# ── 표 추출 예시 ──────────────────────────────────────────────────────────────

def demo_table_extraction():
    """PDF에서 표 추출 시뮬레이션"""
    # 실제로는 unstructured로 PDF에서 표 추출
    # from unstructured.partition.pdf import partition_pdf
    # elements = partition_pdf("report.pdf", infer_table_structure=True)

    # 시뮬레이션 데이터
    sample_table = {
        "type": "Table",
        "text": "구분 | 2023 | 2022 | 변화율\n매출액 | 258,935 | 302,231 | -14.3%\n영업이익 | 6,569 | 43,376 | -84.9%",
        "metadata": {
            "filename": "samsung_annual_2023.pdf",
            "page_number": 5,
            "table_type": "financial_summary",
        }
    }

    print("=== 표 추출 결과 ===")
    print(f"타입: {sample_table['type']}")
    print(f"내용:\n{sample_table['text']}")
    print(f"메타데이터: {sample_table['metadata']}")


def parse_financial_table(table_text: str) -> List[Dict]:
    """재무 표 텍스트를 구조화된 데이터로 변환"""
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if not lines:
        return []

    headers = [h.strip() for h in lines[0].split("|")]
    rows = []

    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))

    return rows


def demo_chunk_strategies():
    """청크 전략 비교"""
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )

    sample_text = """## 사업 개요
삼성전자는 1969년 설립된 글로벌 전자기업입니다.

## 주요 사업 부문
반도체 부문은 메모리와 시스템 반도체를 생산합니다.
가전 부문은 TV, 냉장고 등을 제조합니다.

## 재무 현황
2023년 매출액은 258.9조원입니다.
영업이익은 6.6조원으로 전년 대비 크게 감소했습니다."""

    print("=== 청크 전략 비교 ===\n")

    # 1. 고정 크기 분할
    fixed_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    fixed_chunks = fixed_splitter.split_text(sample_text)
    print(f"고정 크기 분할 ({len(fixed_chunks)}청크):")
    for i, c in enumerate(fixed_chunks, 1):
        print(f"  [{i}] {c[:60]}...")

    # 2. 재귀적 분할
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", " "],
    )
    recursive_chunks = recursive_splitter.split_text(sample_text)
    print(f"\n재귀적 분할 ({len(recursive_chunks)}청크):")
    for i, c in enumerate(recursive_chunks, 1):
        print(f"  [{i}] {c[:80]}...")


def build_rag_with_tables():
    """표 포함 문서 RAG 구축 시뮬레이션"""
    from langchain.schema import Document

    # 텍스트 + 표를 혼합한 문서 목록
    documents = [
        Document(
            page_content="삼성전자 2023년 매출액은 258.9조원, 영업이익 6.6조원입니다.",
            metadata={"type": "text", "source": "samsung_2023", "page": 1}
        ),
        Document(
            page_content="재무표: 매출액 258,935억원 | 영업이익 6,569억원 | 순이익 15,487억원",
            metadata={"type": "table", "source": "samsung_2023", "page": 5}
        ),
        Document(
            page_content="SK하이닉스 2023년 매출은 32.8조원으로 전년 대비 26% 감소했습니다.",
            metadata={"type": "text", "source": "skhynix_2023", "page": 1}
        ),
    ]

    print("\n=== RAG 문서 구성 ===")
    for doc in documents:
        print(f"[{doc.metadata['type']}] {doc.page_content[:80]}...")

    return documents


if __name__ == "__main__":
    demo_table_extraction()

    # 표 파싱 테스트
    table_text = """구분 | 2023 | 2022
매출액 | 258,935 | 302,231
영업이익 | 6,569 | 43,376"""

    rows = parse_financial_table(table_text)
    print(f"\n=== 파싱된 재무 데이터 ({len(rows)}행) ===")
    for row in rows:
        print(f"  {row}")

    demo_chunk_strategies()
    build_rag_with_tables()

    print("\n✅ 완료 (실제 unstructured 사용은 PDF 파일과 라이브러리 설치 필요)")
