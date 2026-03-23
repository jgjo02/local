"""
Document Parser for Korean Business Reports (사업보고서).

Uses the unstructured library to parse PDF documents and extract:
- Text sections with hierarchy
- Financial tables
- Charts and images
- Document structure
"""

import json
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

from .config import config


class ElementType(str, Enum):
    """Classified types of document elements."""
    TITLE = "title"
    SECTION_HEADER = "section_header"
    NARRATIVE_TEXT = "narrative_text"
    FINANCIAL_TABLE = "financial_table"
    GENERAL_TABLE = "general_table"
    CHART_IMAGE = "chart_image"
    LOGO_IMAGE = "logo_image"
    GENERAL_IMAGE = "general_image"
    FOOTER = "footer"
    HEADER = "header"
    LIST_ITEM = "list_item"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


# Korean section keywords for classification
KOREAN_FINANCIAL_KEYWORDS = [
    "매출액", "영업이익", "당기순이익", "자산총계", "부채총계", "자본총계",
    "영업활동", "투자활동", "재무활동", "현금흐름", "손익계산서", "재무상태표",
    "자본변동표", "현금흐름표", "주석", "EPS", "BPS", "ROE", "ROA", "PER", "PBR",
]

KOREAN_SECTION_KEYWORDS = [
    "I.", "II.", "III.", "IV.", "V.", "VI.",
    "제1", "제2", "제3", "제4", "제5",
    "1.", "2.", "3.", "4.", "5.",
    "가.", "나.", "다.", "라.", "마.",
    "사업의 개요", "사업의 내용", "재무에 관한 사항",
    "이사회 등 회사의 기관", "주주에 관한 사항",
    "임원 및 직원 등에 관한 사항", "계열회사 등에 관한 사항",
]


@dataclass
class ParsedElement:
    """A single parsed element from a document."""
    element_id: str
    element_type: ElementType
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)
    image_path: Optional[str] = None
    table_data: Optional[list[list[str]]] = None
    section_hierarchy: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["element_type"] = self.element_type.value
        return d


@dataclass
class DocumentSection:
    """A logical section of the document."""
    section_id: str
    title: str
    level: int  # 0=top, 1=chapter, 2=section, 3=subsection
    elements: list[ParsedElement] = field(default_factory=list)
    subsections: list["DocumentSection"] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

    def get_full_text(self) -> str:
        return "\n".join(
            e.text for e in self.elements if e.text
        )

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "elements": [e.to_dict() for e in self.elements],
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class ParsedDocument:
    """Full parsed document structure."""
    document_id: str
    source_path: str
    company_name: str
    total_pages: int
    sections: list[DocumentSection] = field(default_factory=list)
    all_elements: list[ParsedElement] = field(default_factory=list)
    tables: list[ParsedElement] = field(default_factory=list)
    images: list[ParsedElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "company_name": self.company_name,
            "total_pages": self.total_pages,
            "metadata": self.metadata,
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "images": [i.to_dict() for i in self.images],
        }


class UnstructuredDocumentParser:
    """
    Parses Korean business reports (사업보고서) using the unstructured library.

    Supports both local processing and the Unstructured API.
    Extracts and classifies text, tables, images, and charts.
    """

    def __init__(self):
        self.use_api = config.document_parsing.use_unstructured_api
        self.api_key = config.document_parsing.unstructured_api_key
        self.api_url = config.document_parsing.unstructured_api_url
        self.image_dpi = config.document_parsing.pdf_image_dpi
        self._element_counter = 0

    def _next_id(self, prefix: str = "elem") -> str:
        self._element_counter += 1
        return f"{prefix}_{self._element_counter:06d}"

    def parse_business_report(
        self,
        pdf_path: Union[str, Path],
        company_name: str = "",
        extract_images: bool = True,
        output_image_dir: Optional[Path] = None,
    ) -> ParsedDocument:
        """
        Parse a Korean business report PDF.

        Args:
            pdf_path: Path to the PDF file
            company_name: Company name (used for metadata)
            extract_images: Whether to extract embedded images
            output_image_dir: Directory to save extracted images

        Returns:
            ParsedDocument with all extracted elements
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing business report: {pdf_path.name}")
        self._element_counter = 0

        # Set up image output directory
        if output_image_dir is None:
            output_image_dir = (
                config.paths.parsed_docs_dir
                / pdf_path.stem
                / "images"
            )
        output_image_dir.mkdir(parents=True, exist_ok=True)

        # Parse with unstructured
        raw_elements = self._parse_with_unstructured(
            pdf_path, output_image_dir, extract_images
        )

        # Classify each element
        classified = [
            self.classify_element_type(elem) for elem in raw_elements
        ]

        # Build document structure
        document = self.build_document_structure(classified)
        document.document_id = self._next_id("doc")
        document.source_path = str(pdf_path)
        document.company_name = company_name or pdf_path.stem

        logger.info(
            f"Parsed {len(document.all_elements)} elements, "
            f"{len(document.tables)} tables, "
            f"{len(document.images)} images, "
            f"{len(document.sections)} top-level sections"
        )
        return document

    def _parse_with_unstructured(
        self,
        pdf_path: Path,
        image_dir: Path,
        extract_images: bool,
    ) -> list[dict]:
        """Call unstructured to get raw elements."""
        try:
            if self.use_api and self.api_key:
                return self._parse_via_api(pdf_path)
            else:
                return self._parse_locally(pdf_path, image_dir, extract_images)
        except ImportError as e:
            logger.error(f"unstructured not installed: {e}")
            raise

    def _parse_locally(
        self,
        pdf_path: Path,
        image_dir: Path,
        extract_images: bool,
    ) -> list[dict]:
        """Parse locally using unstructured."""
        try:
            from unstructured.partition.pdf import partition_pdf

            strategy = "hi_res" if extract_images else "fast"

            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=strategy,
                extract_images_in_pdf=extract_images,
                extract_image_block_output_dir=str(image_dir),
                extract_image_block_types=["Image", "Table"],
                include_page_breaks=True,
                infer_table_structure=True,
                chunking_strategy=None,  # We handle chunking ourselves
                languages=["kor", "eng"],
            )

            raw = []
            for elem in elements:
                elem_dict = {
                    "type": type(elem).__name__,
                    "text": getattr(elem, "text", "") or "",
                    "metadata": {},
                }

                # Extract metadata
                if hasattr(elem, "metadata"):
                    meta = elem.metadata
                    elem_dict["metadata"] = {
                        "page_number": getattr(meta, "page_number", 1) or 1,
                        "filename": getattr(meta, "filename", "") or "",
                        "image_path": getattr(meta, "image_path", None),
                        "coordinates": str(getattr(meta, "coordinates", "")),
                        "category_depth": getattr(meta, "category_depth", 0) or 0,
                        "text_as_html": getattr(meta, "text_as_html", None),
                    }

                raw.append(elem_dict)

            return raw

        except ImportError:
            logger.warning(
                "unstructured not available. Falling back to pdfplumber parser."
            )
            return self._parse_with_pdfplumber(pdf_path, image_dir, extract_images)

    def _parse_with_pdfplumber(
        self, pdf_path: Path, image_dir: Path, extract_images: bool
    ) -> list[dict]:
        """Fallback parser using pdfplumber."""
        import pdfplumber

        raw = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                if text.strip():
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        raw.append({
                            "type": "NarrativeText",
                            "text": para,
                            "metadata": {
                                "page_number": page_num,
                                "filename": pdf_path.name,
                                "image_path": None,
                                "coordinates": "",
                                "category_depth": 0,
                                "text_as_html": None,
                            },
                        })

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # Convert table to HTML-like string
                        rows = []
                        for row in table:
                            cells = [str(c) if c is not None else "" for c in row]
                            rows.append(" | ".join(cells))
                        table_text = "\n".join(rows)
                        raw.append({
                            "type": "Table",
                            "text": table_text,
                            "metadata": {
                                "page_number": page_num,
                                "filename": pdf_path.name,
                                "image_path": None,
                                "coordinates": "",
                                "category_depth": 0,
                                "text_as_html": table_text,
                                "table_data": table,
                            },
                        })

                # Add page break marker
                raw.append({
                    "type": "PageBreak",
                    "text": "",
                    "metadata": {
                        "page_number": page_num,
                        "filename": pdf_path.name,
                        "image_path": None,
                        "coordinates": "",
                        "category_depth": 0,
                        "text_as_html": None,
                    },
                })

        return raw

    def _parse_via_api(self, pdf_path: Path) -> list[dict]:
        """Parse using Unstructured cloud API."""
        import httpx

        with open(pdf_path, "rb") as f:
            files = {"files": (pdf_path.name, f, "application/pdf")}
            headers = {"unstructured-api-key": self.api_key}
            params = {
                "strategy": "hi_res",
                "pdf_infer_table_structure": True,
                "languages": ["kor", "eng"],
            }

            response = httpx.post(
                self.api_url,
                headers=headers,
                files=files,
                params=params,
                timeout=300,
            )
            response.raise_for_status()
            return response.json()

    def classify_element_type(self, raw_element: dict) -> ParsedElement:
        """
        Classify a raw unstructured element into a typed ParsedElement.

        Uses heuristics based on element type, text content, and metadata.
        """
        elem_type_str = raw_element.get("type", "")
        text = raw_element.get("text", "").strip()
        meta = raw_element.get("metadata", {})
        page_number = meta.get("page_number", 1)
        image_path = meta.get("image_path")
        table_data_raw = meta.get("table_data")
        text_as_html = meta.get("text_as_html")

        # Determine element type
        element_type = self._infer_element_type(
            elem_type_str, text, image_path, text_as_html, table_data_raw
        )

        # Parse table data if present
        table_data = None
        if element_type in (ElementType.FINANCIAL_TABLE, ElementType.GENERAL_TABLE):
            table_data = self._parse_table_data(text_as_html, table_data_raw, text)

        return ParsedElement(
            element_id=self._next_id("elem"),
            element_type=element_type,
            text=text,
            page_number=page_number,
            metadata=meta,
            image_path=image_path,
            table_data=table_data,
            section_hierarchy=[],
        )

    def _infer_element_type(
        self,
        elem_type_str: str,
        text: str,
        image_path: Optional[str],
        text_as_html: Optional[str],
        table_data: Any,
    ) -> ElementType:
        """Infer the semantic type of an element."""
        etype = elem_type_str.lower()

        # Page breaks
        if etype == "pagebreak":
            return ElementType.PAGE_BREAK

        # Images
        if etype == "image" or image_path:
            if self._looks_like_chart(text):
                return ElementType.CHART_IMAGE
            if self._looks_like_logo(text, image_path or ""):
                return ElementType.LOGO_IMAGE
            return ElementType.GENERAL_IMAGE

        # Tables
        if etype == "table" or table_data is not None or text_as_html:
            if self._is_financial_table(text):
                return ElementType.FINANCIAL_TABLE
            return ElementType.GENERAL_TABLE

        # Titles and headers
        if etype in ("title",):
            return ElementType.TITLE

        if etype in ("header",):
            return ElementType.HEADER

        if etype in ("footer",):
            return ElementType.FOOTER

        if etype in ("listitem", "list_item"):
            return ElementType.LIST_ITEM

        # Section headers heuristic
        if etype in ("title", "header") or self._is_section_header(text):
            return ElementType.SECTION_HEADER

        # Default
        return ElementType.NARRATIVE_TEXT

    def _is_financial_table(self, text: str) -> bool:
        """Check if text contains financial table indicators."""
        text_lower = text.lower()
        matches = sum(
            1 for kw in KOREAN_FINANCIAL_KEYWORDS if kw in text
        )
        # Also check for numeric patterns typical in financial tables
        has_numbers = bool(re.search(r"[\d,]+\s*(억|원|천|만|백)", text))
        return matches >= 2 or (matches >= 1 and has_numbers)

    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header."""
        if not text or len(text) > 200:
            return False
        for kw in KOREAN_SECTION_KEYWORDS:
            if text.startswith(kw):
                return True
        # Short text ending with nothing (no period) often headers
        if len(text) < 60 and not text.endswith((".", "다", "요", "임")):
            if re.match(r"^[\dIVX가-힣\s\.\-]+$", text):
                return True
        return False

    def _looks_like_chart(self, text: str) -> bool:
        """Heuristic: does the element description suggest a chart?"""
        chart_keywords = [
            "chart", "graph", "그래프", "차트", "도표", "추이", "비율",
            "매출", "영업이익", "성장률", "점유율",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in chart_keywords)

    def _looks_like_logo(self, text: str, path: str) -> bool:
        """Heuristic: does the element look like a logo?"""
        return "logo" in path.lower() or "logo" in text.lower()

    def _parse_table_data(
        self,
        text_as_html: Optional[str],
        raw_table: Any,
        fallback_text: str,
    ) -> list[list[str]]:
        """Convert table data to list of rows."""
        if raw_table and isinstance(raw_table, list):
            return [
                [str(cell) if cell is not None else "" for cell in row]
                for row in raw_table
            ]

        if text_as_html:
            return self._parse_html_table(text_as_html)

        # Fallback: split pipe-delimited text
        rows = []
        for line in fallback_text.split("\n"):
            if "|" in line:
                cells = [c.strip() for c in line.split("|")]
                rows.append(cells)
        return rows if rows else [[fallback_text]]

    def _parse_html_table(self, html: str) -> list[list[str]]:
        """Parse simple HTML table into list of rows."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            rows = []
            for tr in soup.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            return rows
        except Exception:
            # Simple regex fallback
            rows = []
            row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
            cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
            for row_match in row_pattern.finditer(html):
                cells = [
                    re.sub(r"<[^>]+>", "", c).strip()
                    for c in cell_pattern.findall(row_match.group(1))
                ]
                if cells:
                    rows.append(cells)
            return rows

    def extract_tables(self, document: ParsedDocument) -> list[ParsedElement]:
        """Extract all table elements from a parsed document."""
        return [
            elem for elem in document.all_elements
            if elem.element_type in (
                ElementType.FINANCIAL_TABLE, ElementType.GENERAL_TABLE
            )
        ]

    def extract_images(self, document: ParsedDocument) -> list[ParsedElement]:
        """Extract all image elements from a parsed document."""
        return [
            elem for elem in document.all_elements
            if elem.element_type in (
                ElementType.CHART_IMAGE,
                ElementType.LOGO_IMAGE,
                ElementType.GENERAL_IMAGE,
            )
        ]

    def extract_text_sections(
        self, document: ParsedDocument
    ) -> dict[str, list[ParsedElement]]:
        """
        Extract text elements organized by section.

        Returns:
            Dict mapping section title to list of text elements
        """
        sections: dict[str, list[ParsedElement]] = {}

        def collect_section(section: DocumentSection, prefix: str = ""):
            title = f"{prefix}{section.title}".strip() or "서문"
            text_elems = [
                e for e in section.elements
                if e.element_type in (
                    ElementType.NARRATIVE_TEXT,
                    ElementType.LIST_ITEM,
                )
            ]
            if text_elems:
                sections[title] = text_elems
            for sub in section.subsections:
                collect_section(sub, prefix=f"{title} > ")

        for section in document.sections:
            collect_section(section)

        return sections

    def build_document_structure(
        self, elements: list[ParsedElement]
    ) -> ParsedDocument:
        """
        Reconstruct hierarchical document structure from flat element list.

        Uses section header detection to build a tree of sections.
        """
        document = ParsedDocument(
            document_id="",
            source_path="",
            company_name="",
            total_pages=max((e.page_number for e in elements), default=1),
        )
        document.all_elements = elements

        # Separate tables and images
        document.tables = self.extract_tables(document)
        document.images = self.extract_images(document)

        # Build section tree
        current_sections: list[DocumentSection] = []  # stack
        current_level = 0

        def get_section_level(text: str) -> int:
            """Determine heading level from text content."""
            # Level 1: Roman numerals or major Korean headings
            if re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.", text):
                return 1
            if re.match(r"^제\d+[부편장]", text):
                return 1
            # Level 2: Arabic numerals
            if re.match(r"^\d+\.", text):
                return 2
            # Level 3: Korean alphabet
            if re.match(r"^[가나다라마바사아자차카타파하]\.", text):
                return 3
            # Level 4: parenthetical
            if re.match(r"^\(\d+\)", text):
                return 4
            return 2  # default to level 2

        root_section = DocumentSection(
            section_id=self._next_id("sec"),
            title="문서 전체",
            level=0,
        )
        document.sections.append(root_section)
        active_path: list[DocumentSection] = [root_section]

        for elem in elements:
            if elem.element_type == ElementType.PAGE_BREAK:
                continue

            if elem.element_type in (
                ElementType.TITLE,
                ElementType.SECTION_HEADER,
            ) and elem.text:
                level = get_section_level(elem.text)
                new_section = DocumentSection(
                    section_id=self._next_id("sec"),
                    title=elem.text[:200],
                    level=level,
                    page_start=elem.page_number,
                )

                # Find appropriate parent
                while len(active_path) > 1 and active_path[-1].level >= level:
                    active_path.pop()

                parent = active_path[-1]
                parent.subsections.append(new_section)
                active_path.append(new_section)

                # Set section hierarchy on element
                elem.section_hierarchy = [s.title for s in active_path]

            else:
                # Assign element to current deepest section
                if active_path:
                    active_path[-1].elements.append(elem)
                    elem.section_hierarchy = [s.title for s in active_path]

        # Update page ranges for all sections
        self._update_page_ranges(document.sections)

        return document

    def _update_page_ranges(self, sections: list[DocumentSection]) -> None:
        """Recursively update page_start and page_end for all sections."""
        for section in sections:
            pages = [e.page_number for e in section.elements if e.page_number]
            if pages:
                if not section.page_start:
                    section.page_start = min(pages)
                section.page_end = max(pages)

            self._update_page_ranges(section.subsections)

            # Parent page range encompasses children
            child_pages = []
            for sub in section.subsections:
                if sub.page_start:
                    child_pages.append(sub.page_start)
                if sub.page_end:
                    child_pages.append(sub.page_end)
            if child_pages:
                section.page_start = min(
                    section.page_start or 9999, min(child_pages)
                )
                section.page_end = max(section.page_end or 0, max(child_pages))

    def save_parsed(
        self,
        document: ParsedDocument,
        output_dir: Union[str, Path],
    ) -> Path:
        """
        Save the parsed document structure to disk.

        Saves:
        - document_structure.json : full structure
        - sections/<section_title>.txt : text per section
        - tables/<table_id>.json : table data
        - metadata.json : document metadata

        Args:
            document: Parsed document
            output_dir: Directory to save output

        Returns:
            Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full structure JSON
        structure_path = output_dir / "document_structure.json"
        with open(structure_path, "w", encoding="utf-8") as f:
            json.dump(document.to_dict(), f, ensure_ascii=False, indent=2)

        # Sections as text files
        sections_dir = output_dir / "sections"
        sections_dir.mkdir(exist_ok=True)
        text_sections = self.extract_text_sections(document)
        for section_title, elements in text_sections.items():
            # Sanitize filename
            safe_name = re.sub(r"[<>:/\\|?*\n\r]", "_", section_title)[:100]
            section_file = sections_dir / f"{safe_name}.txt"
            with open(section_file, "w", encoding="utf-8") as f:
                f.write(f"# {section_title}\n\n")
                for elem in elements:
                    f.write(elem.text + "\n\n")

        # Tables as JSON
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        for i, table in enumerate(document.tables):
            table_path = tables_dir / f"table_{i:04d}_page{table.page_number}.json"
            with open(table_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "element_id": table.element_id,
                        "page_number": table.page_number,
                        "element_type": table.element_type.value,
                        "text": table.text,
                        "table_data": table.table_data,
                        "section_hierarchy": table.section_hierarchy,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # Metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "document_id": document.document_id,
                    "source_path": document.source_path,
                    "company_name": document.company_name,
                    "total_pages": document.total_pages,
                    "total_elements": len(document.all_elements),
                    "total_tables": len(document.tables),
                    "total_images": len(document.images),
                    "total_sections": len(document.sections),
                    "metadata": document.metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved parsed document to: {output_dir}")
        return output_dir
