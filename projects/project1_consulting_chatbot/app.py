"""
Main entry point for the AI Consulting Chatbot (주택청약 AI 상담 챗봇).

Initialises the full RAG pipeline and launches the Gradio web interface.

Usage:
    # Basic (uses .env for all settings)
    python app.py

    # Rebuild vector store from scratch
    python app.py --rebuild

    # Override port and share
    python app.py --port 8080 --share

    # Use specific data file(s)
    python app.py --data-files data/sample_faq.txt data/extra.pdf

    # Debug mode (verbose logging)
    python app.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="주택청약 AI 상담 챗봇 — RAG + LangChain + Gradio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server options
    server = parser.add_argument_group("Gradio server options")
    server.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the Gradio server on (overrides GRADIO_SERVER_PORT in .env)",
    )
    server.add_argument(
        "--host",
        type=str,
        default=None,
        help="Bind address for the Gradio server (overrides GRADIO_SERVER_NAME in .env)",
    )
    server.add_argument(
        "--share",
        action="store_true",
        default=None,
        help="Create a public Gradio share link",
    )
    server.add_argument(
        "--no-share",
        dest="share",
        action="store_false",
        help="Do NOT create a share link (default)",
    )

    # Data / vector store options
    data = parser.add_argument_group("Data options")
    data.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help=(
            "Rebuild the vector store from source documents even if one "
            "already exists on disk"
        ),
    )
    data.add_argument(
        "--data-files",
        nargs="*",
        metavar="FILE",
        default=None,
        help=(
            "One or more .txt / .pdf files to index. "
            "If omitted, all .txt and .pdf files in the data/ directory are used."
        ),
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override the LOG_LEVEL setting from .env",
    )

    return parser


# ---------------------------------------------------------------------------
# Pipeline initialisation helpers
# ---------------------------------------------------------------------------

def collect_data_files(data_dir: Path, extra_files: list[str] | None) -> list[Path]:
    """
    Collect data files to index.

    Args:
        data_dir:    Default data directory.
        extra_files: Optional list of explicit file paths from CLI.

    Returns:
        List of resolved Path objects.
    """
    if extra_files:
        paths = [Path(f).resolve() for f in extra_files]
        missing = [p for p in paths if not p.exists()]
        if missing:
            logger.warning(
                "The following files were not found and will be skipped: %s",
                ", ".join(str(p) for p in missing),
            )
        return [p for p in paths if p.exists()]

    # Auto-discover in data/
    files = sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.pdf"))
    if not files:
        logger.warning(
            "No .txt or .pdf files found in %s. "
            "Place FAQ data there or use --data-files.",
            data_dir,
        )
    return files


def initialise_pipeline(
    config: "Config",  # noqa: F821 – forward ref for type hint only
    data_files: list[Path],
    rebuild: bool,
) -> tuple["VectorStoreManager", "RAGChain"]:  # noqa: F821
    """
    Build or load the vector store and construct the RAG chain.

    Args:
        config:     Application config.
        data_files: Files to embed if building from scratch.
        rebuild:    If True, always rebuild even if a persisted store exists.

    Returns:
        A tuple of (VectorStoreManager, RAGChain).
    """
    from src.document_processor import DocumentProcessor
    from src.rag_chain import RAGChain
    from src.vector_store import VectorStoreManager

    vsm = VectorStoreManager(config)

    # Try to load existing store first (unless --rebuild)
    loaded = False
    if not rebuild:
        try:
            loaded = vsm.load_existing()
        except Exception as exc:
            logger.warning("Could not load existing vector store: %s", exc)

    if not loaded:
        if not data_files:
            logger.error(
                "No data files found and no existing vector store. "
                "Provide data files or run with --data-files."
            )
            sys.exit(1)

        logger.info(
            "Building vector store from %d file(s): %s",
            len(data_files),
            ", ".join(f.name for f in data_files),
        )
        processor = DocumentProcessor(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        documents = processor.batch_process(data_files, split=True)

        if not documents:
            logger.error("No documents were produced from the provided data files.")
            sys.exit(1)

        stats = processor.get_stats(documents)
        logger.info(
            "Processed %d chunks | avg %.0f chars | sources: %s",
            stats["count"],
            stats["avg_chars"],
            ", ".join(Path(s).name for s in stats["sources"]),
        )

        vsm.build_from_documents(documents)
        vsm.save()

    logger.info(
        "Vector store ready: %d documents indexed.",
        vsm.get_document_count(),
    )

    # Build the RAG chain
    rag = RAGChain(config, vsm)
    rag.build_chain()
    logger.info("RAG chain built and ready.")

    return vsm, rag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    from src.config import get_config

    config = get_config()

    # CLI log-level override
    if args.log_level:
        numeric = getattr(logging, args.log_level)
        logging.getLogger().setLevel(numeric)
        logger.info("Log level overridden to %s via CLI.", args.log_level)

    # ------------------------------------------------------------------
    # Validate API key
    # ------------------------------------------------------------------
    if not config.validate_openai_key():
        print(
            "\n[ERROR] OPENAI_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Set OPENAI_API_KEY=sk-...\n"
            "  3. Re-run this script.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Configuration loaded: %s", config)

    # ------------------------------------------------------------------
    # Collect data files
    # ------------------------------------------------------------------
    data_files = collect_data_files(config.app.data_dir, args.data_files)
    logger.info(
        "Data files: [%s]",
        ", ".join(f.name for f in data_files) if data_files else "none",
    )

    # ------------------------------------------------------------------
    # Initialise pipeline (vector store + RAG chain)
    # ------------------------------------------------------------------
    from src.document_processor import DocumentProcessor

    vsm, rag = initialise_pipeline(config, data_files, rebuild=args.rebuild)

    doc_processor = DocumentProcessor(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )

    # ------------------------------------------------------------------
    # Build and launch UI
    # ------------------------------------------------------------------
    from src.chatbot_ui import ChatbotUI

    ui = ChatbotUI(
        rag_chain=rag,
        config=config,
        doc_processor=doc_processor,
        vsm=vsm,
    )

    # Override share/port from CLI if provided
    share_override = args.share  # None means "use config"
    port_override = args.port    # None means "use config"
    host_override = args.host    # None means "use config"

    print(
        f"\n{'='*60}\n"
        f"  {config.app.app_name}\n"
        f"  Model : {config.model.chat_model}\n"
        f"  Vector DB : {config.vector_store.store_type} "
        f"({vsm.get_document_count()} docs)\n"
        f"  Retriever : {config.retriever.search_type} (k={config.retriever.k})\n"
        f"{'='*60}\n"
    )

    ui.launch(
        share=share_override,
        port=port_override,
        server_name=host_override,
    )


if __name__ == "__main__":
    main()
