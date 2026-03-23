"""
Document processing module for AI Consulting Chatbot.

Handles loading, cleaning, splitting, and persisting documents
for use in the RAG pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Loads, cleans, and splits documents for the RAG pipeline.

    Supported formats:
    - PDF  (.pdf) via pypdf / langchain PyPDFLoader
    - Plain text (.txt)
    - Batch processing of mixed file lists

    Example usage::

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        docs = processor.load_text("data/sample_faq.txt")
        chunks = processor.split_documents(docs)
        processor.save_processed(chunks, "data/processed.json")
    """

    # Characters considered noise that should be stripped
    _NOISE_PATTERN = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # ASCII control chars (keep \t \n \r)
        r"|"
        r"[\ufff0-\uffff]"  # Unicode specials
    )

    # Collapse runs of more than 3 blank lines into 2
    _MULTI_BLANK_LINES = re.compile(r"\n{3,}")

    # Collapse runs of spaces / tabs (but not newlines)
    _MULTI_SPACES = re.compile(r"[ \t]{2,}")

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            chunk_size:    Target size of each text chunk (characters).
            chunk_overlap: Number of characters shared between consecutive chunks.
            separators:    Custom separator list for RecursiveCharacterTextSplitter.
                           Defaults to Korean-aware separators.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",   # Paragraph break
            "\n",     # Line break
            "。",     # CJK full stop
            ".",      # ASCII full stop
            "？",     # CJK question mark
            "?",
            "！",
            "!",
            "，",
            ",",
            " ",
            "",
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            "DocumentProcessor initialised (chunk_size=%d, chunk_overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Load a PDF file and return a list of LangChain Documents.

        Each page in the PDF becomes one Document with metadata
        containing the source path and page number.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of Document objects, one per PDF page.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError:       If pypdf is not installed.
            RuntimeError:      On any other loading failure.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as exc:
            raise ImportError(
                "pypdf and langchain-community are required to load PDFs. "
                "Install with: pip install pypdf langchain-community"
            ) from exc

        logger.info("Loading PDF: %s", file_path)
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to load PDF '{file_path}': {exc}") from exc

        # Clean text in every page document
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
            doc.metadata.setdefault("source", str(file_path))
            doc.metadata.setdefault("file_type", "pdf")

        # Drop empty pages
        documents = [d for d in documents if d.page_content.strip()]
        logger.info("Loaded %d non-empty pages from %s", len(documents), file_path.name)
        return documents

    def load_text(self, file_path: str | Path) -> list[Document]:
        """
        Load a plain-text file and return it as LangChain Documents.

        Each Q&A block (separated by blank lines) becomes a separate
        Document to preserve semantic boundaries.

        Args:
            file_path: Absolute or relative path to the text file.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file encoding is not UTF-8.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        logger.info("Loading text file: %s", file_path)
        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to cp949 for Korean text files
            raw = file_path.read_text(encoding="cp949")
            logger.warning(
                "UTF-8 decode failed for %s; fell back to cp949", file_path.name
            )

        cleaned = self.clean_text(raw)

        # Split on double newlines to get semantic paragraphs / Q&A pairs
        blocks = [b.strip() for b in re.split(r"\n{2,}", cleaned) if b.strip()]

        documents: list[Document] = []
        for i, block in enumerate(blocks):
            doc = Document(
                page_content=block,
                metadata={
                    "source": str(file_path),
                    "file_type": "text",
                    "block_index": i,
                },
            )
            documents.append(doc)

        logger.info(
            "Loaded %d text blocks from %s", len(documents), file_path.name
        )
        return documents

    def split_documents(
        self,
        documents: list[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> list[Document]:
        """
        Split a list of Documents into smaller chunks.

        If chunk_size / chunk_overlap are provided they override the
        instance defaults for this call only.

        Args:
            documents:     Source documents to split.
            chunk_size:    Override chunk size (characters).
            chunk_overlap: Override chunk overlap (characters).

        Returns:
            List of chunk Documents with updated metadata.
        """
        if not documents:
            logger.warning("split_documents called with empty document list")
            return []

        splitter = self._splitter
        if chunk_size is not None or chunk_overlap is not None:
            effective_size = chunk_size or self.chunk_size
            effective_overlap = chunk_overlap or self.chunk_overlap
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=effective_size,
                chunk_overlap=effective_overlap,
                separators=self.separators,
                length_function=len,
            )

        logger.info("Splitting %d documents into chunks…", len(documents))
        chunks = splitter.split_documents(documents)

        # Enrich metadata with chunk index
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_total"] = len(chunks)

        logger.info(
            "Produced %d chunks (avg %.0f chars)",
            len(chunks),
            sum(len(c.page_content) for c in chunks) / max(len(chunks), 1),
        )
        return chunks

    def clean_text(self, text: str) -> str:
        """
        Normalise and clean raw text.

        Operations performed:
        1. Unicode NFC normalisation
        2. Remove control characters / Unicode specials
        3. Collapse excessive whitespace
        4. Strip leading/trailing whitespace

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        if not text:
            return text

        # 1. Normalise Unicode to NFC (important for Korean)
        text = unicodedata.normalize("NFC", text)

        # 2. Remove noise characters
        text = self._NOISE_PATTERN.sub("", text)

        # 3. Collapse multiple blank lines
        text = self._MULTI_BLANK_LINES.sub("\n\n", text)

        # 4. Collapse multiple spaces/tabs on a single line
        text = self._MULTI_SPACES.sub(" ", text)

        # 5. Strip
        return text.strip()

    def batch_process(
        self,
        file_paths: list[str | Path],
        split: bool = True,
    ) -> list[Document]:
        """
        Load and optionally split multiple files in one call.

        Supports mixed PDF and text files. Skips unsupported formats
        with a warning rather than raising.

        Args:
            file_paths: List of file paths to process.
            split:      If True (default), chunks are returned.
                        If False, raw page/block documents are returned.

        Returns:
            Aggregated list of Document objects from all files.
        """
        if not file_paths:
            logger.warning("batch_process called with empty file list")
            return []

        all_documents: list[Document] = []
        failed: list[str] = []

        for fp in tqdm(file_paths, desc="Processing files", unit="file"):
            fp = Path(fp)
            try:
                ext = fp.suffix.lower()
                if ext == ".pdf":
                    docs = self.load_pdf(fp)
                elif ext in {".txt", ".md", ".text"}:
                    docs = self.load_text(fp)
                else:
                    logger.warning(
                        "Unsupported file extension '%s' for file: %s — skipping",
                        ext,
                        fp,
                    )
                    continue
                all_documents.extend(docs)
            except Exception as exc:
                logger.error("Failed to process %s: %s", fp, exc)
                failed.append(str(fp))

        if failed:
            logger.warning(
                "%d file(s) failed to load: %s", len(failed), ", ".join(failed)
            )

        logger.info(
            "Batch processing complete: %d raw documents from %d files",
            len(all_documents),
            len(file_paths) - len(failed),
        )

        if split and all_documents:
            all_documents = self.split_documents(all_documents)

        return all_documents

    def save_processed(
        self,
        documents: list[Document],
        output_path: str | Path,
    ) -> None:
        """
        Persist processed Document objects to a JSON file.

        The JSON format is a list of objects with ``page_content``
        and ``metadata`` keys — compatible with LangChain's
        ``Document`` constructor.

        Args:
            documents:   Documents to save.
            output_path: Destination path (.json).

        Raises:
            IOError: If the file cannot be written.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = [
            {
                "page_content": doc.page_content,
                "metadata": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in doc.metadata.items()
                },
            }
            for doc in documents
        ]

        try:
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(serialisable, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            raise IOError(f"Failed to write processed docs to {output_path}: {exc}") from exc

        logger.info(
            "Saved %d documents to %s (%.1f KB)",
            len(documents),
            output_path,
            output_path.stat().st_size / 1024,
        )

    @staticmethod
    def load_processed(input_path: str | Path) -> list[Document]:
        """
        Load previously saved Document objects from a JSON file.

        Args:
            input_path: Path to the JSON file written by ``save_processed``.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Processed document file not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as fh:
            data: list[dict[str, Any]] = json.load(fh)

        documents = [
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
            )
            for item in data
        ]
        logger.info("Loaded %d documents from %s", len(documents), input_path)
        return documents

    def get_stats(self, documents: list[Document]) -> dict[str, Any]:
        """
        Return basic statistics about a list of documents.

        Args:
            documents: Document list to analyse.

        Returns:
            Dictionary with keys: count, total_chars, avg_chars,
            min_chars, max_chars, sources.
        """
        if not documents:
            return {
                "count": 0,
                "total_chars": 0,
                "avg_chars": 0,
                "min_chars": 0,
                "max_chars": 0,
                "sources": [],
            }
        lengths = [len(d.page_content) for d in documents]
        sources = list({d.metadata.get("source", "unknown") for d in documents})
        return {
            "count": len(documents),
            "total_chars": sum(lengths),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "sources": sources,
        }
