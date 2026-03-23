"""
Legal Document Processor for Korean legal texts.

Handles loading, parsing, and chunking of Korean legal documents.
Understands the Korean law structure: 조(Article) > 항(Paragraph) > 호(Item).
Produces LangChain Documents enriched with legal metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import get_config


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LegalArticle:
    """Represents a single 조(Article) parsed from a Korean law."""

    law_name: str
    article_number: str           # e.g. "750"  (숫자 only)
    article_label: str            # e.g. "제750조"
    article_title: str            # e.g. "불법행위의 내용"
    paragraphs: list[str] = field(default_factory=list)    # 항
    items: list[str] = field(default_factory=list)          # 호
    raw_text: str = ""
    law_type: str = ""            # e.g. "민법", "근로기준법"
    enforcement_date: str = ""
    article_category: str = ""    # e.g. "계약", "불법행위"


@dataclass
class ParsedLegalDocument:
    """A fully parsed legal document with its articles."""

    law_name: str
    law_type: str
    raw_text: str
    articles: list[LegalArticle] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


# ---------------------------------------------------------------------------
# Korean legal structure patterns
# ---------------------------------------------------------------------------

# Matches "제750조(불법행위의 내용)" or "제750조"
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)(제\s*(\d+(?:의\d+)?)\s*조(?:의\d+)?\s*(?:\(([^)]+)\))?)",
    re.MULTILINE,
)

# Matches paragraphs: "①", "②", etc.
PARAGRAPH_PATTERN = re.compile(
    r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*(.+?)(?=[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|$)",
    re.DOTALL,
)

# Matches items: "1.", "2." or "가.", "나." or "1)" etc.
ITEM_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+\.|[가-힣]\.|[가-힣]\))\s*(.+?)(?=\n\s*(?:\d+\.|[가-힣]\.|[가-힣]\))|$)",
    re.DOTALL,
)

# Matches law name header lines
LAW_NAME_PATTERN = re.compile(
    r"^(?:【|【\s*)?([\w\s]+법(?:률)?(?:\s+\([^)]+\))?)(?:\s*】)?$",
    re.MULTILINE,
)

# Numeric circle characters map to integers
CIRCLE_TO_INT: dict[str, int] = {
    "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
    "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9, "⑩": 10,
    "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15,
    "⑯": 16, "⑰": 17, "⑱": 18, "⑲": 19, "⑳": 20,
}

# Hanja → Hangul law category mapping
CATEGORY_MAP: dict[str, str] = {
    "계약": "계약법",
    "불법행위": "불법행위법",
    "근로": "노동법",
    "임대": "임대차법",
    "손해배상": "손해배상법",
    "해고": "노동법",
    "임금": "노동법",
}


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------


class LegalDocumentProcessor:
    """
    Processes raw Korean legal text files into LangChain Documents.

    Usage::

        processor = LegalDocumentProcessor()
        documents = processor.batch_process(["data/sample_legal_docs.txt"])
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_legal_documents(self, path: str | Path) -> list[str]:
        """
        Load raw text from a legal document file.

        Supports .txt, .pdf (via pypdf), and .json formats.
        Returns a list of raw text strings (one per document page/section).

        Args:
            path: Path to the legal document file.

        Returns:
            List of raw text strings.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Legal document not found: {path}")

        logger.info(f"Loading legal document: {path}")

        suffix = path.suffix.lower()
        if suffix in (".txt", ".text"):
            return self._load_text(path)
        elif suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix == ".json":
            return self._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def parse_law_structure(self, document: str) -> ParsedLegalDocument:
        """
        Parse a raw Korean legal text into structured articles.

        Extracts 조(articles), 항(paragraphs), and 호(items).

        Args:
            document: Raw Korean legal text.

        Returns:
            ParsedLegalDocument with articles list populated.
        """
        # Normalise unicode (e.g. full-width to half-width)
        document = unicodedata.normalize("NFC", document)

        law_name = self._extract_law_name(document)
        law_type = self._classify_law_type(law_name, document)

        articles = self.split_by_article(document)

        parsed_articles: list[LegalArticle] = []
        for art_text in articles:
            art = self._parse_single_article(art_text, law_name, law_type)
            if art:
                parsed_articles.append(art)

        logger.info(
            f"Parsed '{law_name}': {len(parsed_articles)} articles found"
        )

        return ParsedLegalDocument(
            law_name=law_name,
            law_type=law_type,
            raw_text=document,
            articles=parsed_articles,
            metadata={
                "total_articles": len(parsed_articles),
                "law_type": law_type,
            },
        )

    def split_by_article(self, document: str) -> list[str]:
        """
        Split a legal document into individual article (조) segments.

        Args:
            document: Raw Korean legal text.

        Returns:
            List of text segments, each containing one article.
        """
        matches = list(ARTICLE_PATTERN.finditer(document))
        if not matches:
            logger.warning("No article patterns found – returning document as single chunk")
            return [document]

        segments: list[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(document)
            segment = document[start:end].strip()
            if segment:
                segments.append(segment)

        logger.debug(f"split_by_article produced {len(segments)} segments")
        return segments

    def create_legal_metadata(self, chunk: Document) -> Document:
        """
        Enrich a Document chunk with legal-domain metadata.

        Extracts law name, article number, article title, and category
        from the document content and existing metadata, then adds
        a unique doc_id, source_type, and timestamp.

        Args:
            chunk: A LangChain Document (may already have partial metadata).

        Returns:
            The same Document with enriched metadata.
        """
        content = chunk.page_content
        existing = chunk.metadata or {}

        # Try to parse article info from content
        article_match = ARTICLE_PATTERN.search(content)
        article_label = ""
        article_number = ""
        article_title = ""
        if article_match:
            article_label = article_match.group(1).strip()
            article_number = article_match.group(2)
            article_title = article_match.group(3) or ""

        # Determine law name
        law_name: str = existing.get("law_name", "")
        if not law_name:
            law_name = self._extract_law_name(content)

        # Determine category
        category = self._classify_category(content)

        # Compute stable doc_id
        doc_id = hashlib.sha256(
            f"{law_name}{article_number}{content[:100]}".encode("utf-8")
        ).hexdigest()[:16]

        enriched_metadata = {
            **existing,
            "doc_id": doc_id,
            "law_name": law_name,
            "law_type": existing.get("law_type", self._classify_law_type(law_name, content)),
            "article_label": article_label,
            "article_number": article_number,
            "article_title": article_title,
            "category": category,
            "source_type": existing.get("source_type", "vector_store"),
            "char_count": len(content),
        }

        return Document(page_content=content, metadata=enriched_metadata)

    def batch_process(self, paths: list[str | Path]) -> list[Document]:
        """
        Load, parse, chunk, and enrich all documents at the given paths.

        This is the primary entry point for populating the vector store.

        Args:
            paths: List of file paths to process.

        Returns:
            Flat list of enriched LangChain Documents ready for indexing.
        """
        all_documents: list[Document] = []

        for path in paths:
            try:
                raw_texts = self.load_legal_documents(path)
                for raw_text in raw_texts:
                    parsed = self.parse_law_structure(raw_text)
                    docs = self._convert_to_documents(parsed)
                    enriched = [self.create_legal_metadata(doc) for doc in docs]
                    all_documents.extend(enriched)
                    logger.info(
                        f"{path}: produced {len(enriched)} document chunks "
                        f"from '{parsed.law_name}'"
                    )
            except Exception as exc:
                logger.error(f"Failed to process {path}: {exc}")
                continue

        logger.info(f"batch_process complete: {len(all_documents)} total chunks")
        return all_documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_text(self, path: Path) -> list[str]:
        """Load a plain text file, splitting on double-newlines between law sections."""
        text = path.read_text(encoding="utf-8")
        # Split by section headers (lines with only law names)
        sections = re.split(r"\n{3,}", text)
        return [s.strip() for s in sections if s.strip()]

    def _load_pdf(self, path: Path) -> list[str]:
        """Load a PDF and return one string per page."""
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
            return pages
        except ImportError:
            logger.error("pypdf not installed; cannot load PDF files")
            return []

    def _load_json(self, path: Path) -> list[str]:
        """Load a JSON file containing a list of legal text strings."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict) and "documents" in data:
            return [str(item) for item in data["documents"]]
        else:
            return [json.dumps(data, ensure_ascii=False)]

    def _extract_law_name(self, text: str) -> str:
        """Extract the law name from the first few lines of a text segment."""
        for line in text.splitlines()[:10]:
            line = line.strip()
            # Korean law names typically end in 법, 법률, 규정, 령
            if re.match(r"^[\w\s()]+(?:법률?|규정|령|조례)$", line):
                return line
            match = LAW_NAME_PATTERN.match(line)
            if match:
                return match.group(1).strip()
        return "미상 법률"

    def _classify_law_type(self, law_name: str, text: str) -> str:
        """Classify the broad type of law based on name and content."""
        combined = (law_name + " " + text[:200]).lower()
        if "민법" in combined:
            return "민법"
        elif "근로기준법" in combined or "근로" in law_name:
            return "근로기준법"
        elif "주택임대차" in combined or "임대차" in law_name:
            return "임대차보호법"
        elif "상법" in combined:
            return "상법"
        elif "형법" in combined:
            return "형법"
        elif "헌법" in combined:
            return "헌법"
        return "기타"

    def _classify_category(self, text: str) -> str:
        """Classify the legal category of a chunk based on keyword heuristics."""
        for keyword, category in CATEGORY_MAP.items():
            if keyword in text:
                return category
        return "일반"

    def _parse_single_article(
        self, text: str, law_name: str, law_type: str
    ) -> Optional[LegalArticle]:
        """Parse a single article text into a LegalArticle dataclass."""
        article_match = ARTICLE_PATTERN.search(text)
        if not article_match:
            return None

        article_label = article_match.group(1).strip()
        article_number = article_match.group(2)
        article_title = article_match.group(3) or ""

        # Extract paragraphs (항)
        paragraphs: list[str] = []
        for pm in PARAGRAPH_PATTERN.finditer(text):
            circle = pm.group(1)
            para_text = pm.group(2).strip()
            paragraphs.append(f"{circle} {para_text}")

        # Extract items (호) – look within each paragraph
        items: list[str] = []
        for im in ITEM_PATTERN.finditer(text):
            item_marker = im.group(1)
            item_text = im.group(2).strip()
            items.append(f"{item_marker} {item_text}")

        category = self._classify_category(text)

        return LegalArticle(
            law_name=law_name,
            article_number=article_number,
            article_label=article_label,
            article_title=article_title,
            paragraphs=paragraphs,
            items=items,
            raw_text=text.strip(),
            law_type=law_type,
            article_category=category,
        )

    def _convert_to_documents(self, parsed: ParsedLegalDocument) -> list[Document]:
        """
        Convert a ParsedLegalDocument into a list of LangChain Documents.

        Each article becomes one Document. Articles that exceed chunk_size
        are further split by the text splitter.
        """
        documents: list[Document] = []

        for article in parsed.articles:
            base_metadata = {
                "law_name": article.law_name,
                "law_type": article.law_type,
                "article_label": article.article_label,
                "article_number": article.article_number,
                "article_title": article.article_title,
                "article_category": article.article_category,
                "source_path": parsed.source_path,
                "source_type": "vector_store",
            }

            if len(article.raw_text) <= self._config.chunk_size:
                documents.append(
                    Document(
                        page_content=article.raw_text,
                        metadata=base_metadata,
                    )
                )
            else:
                # Split long articles
                sub_docs = self._text_splitter.create_documents(
                    texts=[article.raw_text],
                    metadatas=[base_metadata],
                )
                documents.extend(sub_docs)

        # If no articles were parsed, fall back to generic chunking
        if not documents:
            logger.warning(
                f"No articles found in '{parsed.law_name}'; using fallback chunking"
            )
            sub_docs = self._text_splitter.create_documents(
                texts=[parsed.raw_text],
                metadatas=[{"law_name": parsed.law_name, "law_type": parsed.law_type}],
            )
            documents.extend(sub_docs)

        return documents
