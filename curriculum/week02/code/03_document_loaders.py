"""
2주차 3차시: 문서 로더 & 텍스트 분할 실습
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document

load_dotenv()


def demo_text_loader():
    """텍스트 파일 로더"""
    # 샘플 파일 생성
    sample_path = "/tmp/sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("""주택청약 안내서

제1장 청약 자격
청약에 참여하려면 만 19세 이상 무주택 세대구성원이어야 합니다.
청약통장 가입 기간에 따라 1순위 자격이 결정됩니다.

제2장 청약 신청 방법
청약홈(applyhome.co.kr)에서 온라인으로 신청합니다.
본인인증 후 희망 주택 선택 → 청약 신청 → 서류 제출 순서로 진행됩니다.

제3장 당첨자 선정
가점제와 추첨제로 당첨자를 선정합니다.
가점제는 무주택기간, 부양가족수, 통장가입기간 3가지 항목으로 계산합니다.
""")

    loader = TextLoader(sample_path, encoding="utf-8")
    docs = loader.load()

    print("텍스트 로더 결과:")
    for doc in docs:
        print(f"  내용 길이: {len(doc.page_content)}자")
        print(f"  메타데이터: {doc.metadata}")
        print(f"  미리보기: {doc.page_content[:100]}...")
    return docs


def demo_text_splitter():
    """RecursiveCharacterTextSplitter 실습"""
    text = """인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템입니다.

머신러닝은 AI의 한 분야로, 데이터를 통해 스스로 학습하는 알고리즘입니다.
지도학습, 비지도학습, 강화학습으로 나뉩니다.

딥러닝은 머신러닝의 한 분야로, 신경망을 여러 층으로 쌓아 학습합니다.
이미지 인식, 자연어 처리 등 다양한 분야에서 활용됩니다.

자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.
챗봇, 번역, 요약 등에 활용됩니다."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(f"\n텍스트 분할 결과 (chunk_size=200, overlap=50):")
    print(f"  총 청크 수: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  [청크 {i}] ({len(chunk)}자)")
        print(f"  {chunk[:100]}...")


def demo_token_splitter():
    """TokenTextSplitter 실습"""
    import tiktoken

    text = "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다. " * 20

    splitter = TokenTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    chunks = splitter.split_text(text)
    print(f"\nToken Splitter 결과 (50 tokens/chunk):")
    print(f"  총 청크 수: {len(chunks)}")

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    for i, chunk in enumerate(chunks[:3], 1):
        tokens = enc.encode(chunk)
        print(f"  [청크 {i}] {len(tokens)} 토큰: {chunk[:60]}...")


def demo_chunk_analysis():
    """청크 품질 분석"""
    sample_docs = [
        Document(
            page_content="제1조 계약의 목적: 본 계약은 갑과 을 사이의 소프트웨어 개발 용역에 관한 사항을 정함을 목적으로 한다.",
            metadata={"source": "contract.pdf", "page": 1}
        ),
        Document(
            page_content="제2조 계약 기간: 계약 기간은 계약 체결일로부터 1년으로 하되, 쌍방 합의에 따라 연장할 수 있다. 계약 종료 30일 전에 서면으로 통보해야 한다.",
            metadata={"source": "contract.pdf", "page": 1}
        ),
        Document(
            page_content="제3조 대금 지급: 을은 계약 금액의 50%를 착수금으로, 나머지 50%를 납품 완료 후 30일 이내에 지급한다.",
            metadata={"source": "contract.pdf", "page": 2}
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(sample_docs)

    print(f"\n청크 품질 분석:")
    sizes = [len(c.page_content) for c in chunks]
    print(f"  총 청크 수: {len(chunks)}")
    print(f"  평균 크기: {sum(sizes)/len(sizes):.0f}자")
    print(f"  최소/최대: {min(sizes)}자 / {max(sizes)}자")


if __name__ == "__main__":
    print("2주차 3차시: 문서 로더 & 텍스트 분할\n")
    demo_text_loader()
    demo_text_splitter()
    demo_token_splitter()
    demo_chunk_analysis()
