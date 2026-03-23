"""
Vector store management module for AI Consulting Chatbot.

Provides a unified interface over Chroma and FAISS backends,
handling creation, persistence, retrieval and incremental updates.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]


class VectorStoreManager:
    """
    Manages the lifecycle of a vector store for the RAG pipeline.

    Supports Chroma (default, persistent) and FAISS (in-memory / file).

    Example usage::

        from src.config import get_config
        from src.vector_store import VectorStoreManager

        cfg = get_config()
        vsm = VectorStoreManager(cfg)
        vsm.build_from_documents(chunks)
        retriever = vsm.get_retriever(search_type="mmr", k=5)
    """

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: A ``Config`` instance (from ``src.config``).
        """
        self.config = config
        self._store: Any = None  # The underlying LangChain vector store object
        self._embeddings: Optional[OpenAIEmbeddings] = None

        logger.info(
            "VectorStoreManager initialised (backend=%s)",
            config.vector_store.store_type,
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _get_embeddings(self) -> OpenAIEmbeddings:
        """Return (and cache) the OpenAI embeddings model."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.config.model.embedding_model,
                openai_api_key=self.config.openai.api_key,
                openai_organization=self.config.openai.org_id,
            )
            logger.debug(
                "Embeddings model created: %s", self.config.model.embedding_model
            )
        return self._embeddings

    # ------------------------------------------------------------------
    # Build / Load
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[Document]) -> None:
        """
        Create a vector store from a list of documents.

        Embeds all documents and, for Chroma, persists to disk.
        Any existing store at the same persist path is overwritten.

        Args:
            documents: Chunked Document objects to embed.

        Raises:
            ValueError:   If documents list is empty.
            RuntimeError: On embedding or store creation failure.
        """
        if not documents:
            raise ValueError("Cannot build vector store from an empty document list.")

        logger.info(
            "Building vector store from %d documents (backend=%s)…",
            len(documents),
            self.config.vector_store.store_type,
        )

        embeddings = self._get_embeddings()

        try:
            if self.config.vector_store.store_type == "chroma":
                self._store = self._build_chroma(documents, embeddings)
            else:
                self._store = self._build_faiss(documents, embeddings)
        except Exception as exc:
            raise RuntimeError(
                f"Vector store creation failed: {exc}"
            ) from exc

        logger.info("Vector store built successfully.")

    def _build_chroma(
        self, documents: list[Document], embeddings: OpenAIEmbeddings
    ) -> Any:
        """Build and persist a Chroma vector store."""
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "langchain-chroma is required. Install with: pip install langchain-chroma"
            ) from exc

        persist_dir = str(self.config.vector_store.persist_dir)
        collection_name = self.config.vector_store.collection_name

        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        logger.info(
            "Chroma store persisted to %s (collection=%s)",
            persist_dir,
            collection_name,
        )
        return store

    def _build_faiss(
        self, documents: list[Document], embeddings: OpenAIEmbeddings
    ) -> Any:
        """Build an in-memory FAISS vector store."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu and langchain-community are required. "
                "Install with: pip install faiss-cpu langchain-community"
            ) from exc

        store = FAISS.from_documents(documents=documents, embedding=embeddings)
        logger.info("FAISS store built in memory (%d vectors).", len(documents))
        return store

    def load_existing(self, persist_dir: Optional[str | Path] = None) -> bool:
        """
        Load a previously persisted vector store from disk.

        For Chroma: reads from ``persist_dir`` (defaults to config value).
        For FAISS:  loads from ``<persist_dir>/faiss_index.pkl``.

        Args:
            persist_dir: Override the config persist directory.

        Returns:
            True if loading succeeded, False if no store was found.

        Raises:
            RuntimeError: On loading failure.
        """
        effective_dir = Path(persist_dir) if persist_dir else self.config.vector_store.persist_dir
        embeddings = self._get_embeddings()

        try:
            if self.config.vector_store.store_type == "chroma":
                return self._load_chroma(effective_dir, embeddings)
            else:
                return self._load_faiss(effective_dir, embeddings)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load existing vector store from {effective_dir}: {exc}"
            ) from exc

    def _load_chroma(self, persist_dir: Path, embeddings: OpenAIEmbeddings) -> bool:
        """Load an existing Chroma store."""
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "langchain-chroma is required. Install with: pip install langchain-chroma"
            ) from exc

        # Chroma directory is considered "existing" when it contains sqlite file
        chroma_sqlite = persist_dir / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            logger.info("No existing Chroma store found at %s.", persist_dir)
            return False

        self._store = Chroma(
            collection_name=self.config.vector_store.collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        count = self._store._collection.count()
        logger.info(
            "Loaded existing Chroma store from %s (%d documents).",
            persist_dir,
            count,
        )
        return True

    def _load_faiss(self, persist_dir: Path, embeddings: OpenAIEmbeddings) -> bool:
        """Load an existing FAISS index."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu and langchain-community are required."
            ) from exc

        index_path = persist_dir / "faiss_index"
        if not index_path.exists():
            logger.info("No existing FAISS index found at %s.", persist_dir)
            return False

        self._store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded existing FAISS index from %s.", index_path)
        return True

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add new documents to the existing vector store.

        Args:
            documents: New Document chunks to embed and insert.

        Raises:
            RuntimeError: If no store has been built/loaded yet.
        """
        if self._store is None:
            raise RuntimeError(
                "No vector store is loaded. Call build_from_documents() or "
                "load_existing() first."
            )
        if not documents:
            logger.warning("add_documents called with empty list — nothing to do.")
            return

        self._store.add_documents(documents)
        logger.info("Added %d documents to the vector store.", len(documents))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_retriever(
        self,
        search_type: SearchType = "mmr",
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
    ) -> Any:
        """
        Return a LangChain BaseRetriever configured for the given strategy.

        Args:
            search_type:      "similarity", "mmr", or
                              "similarity_score_threshold".
            k:                Number of documents to return.
            score_threshold:  Minimum score (only for threshold mode).
            fetch_k:          Documents fetched before MMR reranking.

        Returns:
            A LangChain retriever object.

        Raises:
            RuntimeError: If no store has been built/loaded yet.
        """
        if self._store is None:
            raise RuntimeError(
                "No vector store is loaded. Call build_from_documents() or "
                "load_existing() first."
            )

        cfg_r = self.config.retriever
        effective_k = k or cfg_r.k
        effective_threshold = score_threshold or cfg_r.score_threshold
        effective_fetch_k = fetch_k or cfg_r.fetch_k

        search_kwargs: dict[str, Any] = {"k": effective_k}

        if search_type == "mmr":
            search_kwargs["fetch_k"] = effective_fetch_k
        elif search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = effective_threshold

        retriever = self._store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        logger.debug(
            "Retriever created (type=%s, k=%d)", search_type, effective_k
        )
        return retriever

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """
        Direct similarity search without going through a retriever chain.

        Args:
            query:           Search query string.
            k:               Number of results to return.
            score_threshold: Optional minimum score filter.

        Returns:
            List of the most similar Documents.

        Raises:
            RuntimeError: If no store has been built/loaded yet.
        """
        if self._store is None:
            raise RuntimeError(
                "No vector store is loaded. Call build_from_documents() or "
                "load_existing() first."
            )

        effective_k = k or self.config.retriever.k

        if score_threshold is not None:
            results_with_scores = self._store.similarity_search_with_score(
                query, k=effective_k
            )
            results = [
                doc
                for doc, score in results_with_scores
                if score >= score_threshold
            ]
            logger.debug(
                "Similarity search: query=%r, candidates=%d, above_threshold=%d",
                query[:60],
                len(results_with_scores),
                len(results),
            )
            return results

        results = self._store.similarity_search(query, k=effective_k)
        logger.debug(
            "Similarity search: query=%r, results=%d", query[:60], len(results)
        )
        return results

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Similarity search that also returns relevance scores.

        Args:
            query: Search query string.
            k:     Number of results.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        if self._store is None:
            raise RuntimeError("No vector store is loaded.")

        effective_k = k or self.config.retriever.k
        results = self._store.similarity_search_with_score(query, k=effective_k)
        logger.debug(
            "Similarity search with score: query=%r, results=%d",
            query[:60],
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, persist_dir: Optional[str | Path] = None) -> None:
        """
        Persist the vector store to disk.

        For Chroma this is a no-op (it auto-persists), but calling it
        explicitly is safe and harmless.
        For FAISS this saves the index to ``<persist_dir>/faiss_index``.

        Args:
            persist_dir: Override the config persist directory.

        Raises:
            RuntimeError: If no store is loaded.
        """
        if self._store is None:
            raise RuntimeError("No vector store is loaded — nothing to save.")

        effective_dir = Path(persist_dir) if persist_dir else self.config.vector_store.persist_dir
        effective_dir.mkdir(parents=True, exist_ok=True)

        if self.config.vector_store.store_type == "chroma":
            # Chroma persists automatically; calling persist() explicitly
            # if the method exists (older langchain-chroma versions)
            if hasattr(self._store, "persist"):
                self._store.persist()
            logger.info("Chroma store is persisted at %s.", effective_dir)
        else:
            index_path = effective_dir / "faiss_index"
            self._store.save_local(str(index_path))
            logger.info("FAISS index saved to %s.", index_path)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if a vector store has been built or loaded."""
        return self._store is not None

    def get_document_count(self) -> int:
        """
        Return the number of vectors in the current store.

        Returns:
            Document count, or 0 if no store is loaded.
        """
        if self._store is None:
            return 0

        if self.config.vector_store.store_type == "chroma":
            try:
                return self._store._collection.count()
            except Exception:
                return 0
        else:
            try:
                return self._store.index.ntotal
            except Exception:
                return 0

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "empty"
        return (
            f"VectorStoreManager("
            f"backend={self.config.vector_store.store_type!r}, "
            f"status={status!r}, "
            f"documents={self.get_document_count()})"
        )
