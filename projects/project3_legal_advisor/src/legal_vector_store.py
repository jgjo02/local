"""
Legal Vector Store management.

Provides a unified interface over ChromaDB and FAISS backends.
Supports article-level retrieval, hybrid search, and law-scoped queries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from src.config import get_config


class LegalVectorStore:
    """
    Manages a persistent vector store for Korean legal documents.

    Supports two backends (configurable via VECTOR_STORE_TYPE):
    - ``chroma``: ChromaDB with optional persistence (default)
    - ``faiss``: FAISS in-memory (faster for small corpora)

    Example usage::

        store = LegalVectorStore()
        store.build_from_documents(docs)

        results = store.hybrid_search("임대차 계약 해지 통보 기간", k=5)
    """

    def __init__(self) -> None:
        self._config = get_config()
        self._embeddings = OpenAIEmbeddings(**self._config.get_embedding_kwargs())
        self._store: Optional[VectorStore] = None
        self._initialized: bool = False

        # Attempt to load an existing store from disk
        self._try_load_existing()

    # ------------------------------------------------------------------
    # Build / initialise
    # ------------------------------------------------------------------

    def build_from_documents(self, documents: list[Document]) -> None:
        """
        Create (or rebuild) the vector store from a list of Documents.

        If a store already exists on disk it will be cleared and rebuilt.

        Args:
            documents: Enriched LangChain Documents to index.

        Raises:
            ValueError: If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("Cannot build vector store from empty document list")

        logger.info(
            f"Building {self._config.vector_store_type} vector store "
            f"with {len(documents)} documents …"
        )

        backend = self._config.vector_store_type
        if backend == "chroma":
            self._store = self._build_chroma(documents)
        elif backend == "faiss":
            self._store = self._build_faiss(documents)
        else:
            raise ValueError(f"Unknown vector store backend: {backend}")

        self._initialized = True
        logger.info("Vector store built successfully")

    def _build_chroma(self, documents: list[Document]) -> VectorStore:
        """Build a ChromaDB vector store."""
        from langchain_community.vectorstores import Chroma  # type: ignore

        persist_dir = str(self._config.vector_store_dir)
        os.makedirs(persist_dir, exist_ok=True)

        store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            collection_name=self._config.chroma_collection_name,
            persist_directory=persist_dir,
        )
        return store

    def _build_faiss(self, documents: list[Document]) -> VectorStore:
        """Build a FAISS vector store and save to disk."""
        from langchain_community.vectorstores import FAISS  # type: ignore

        store = FAISS.from_documents(
            documents=documents,
            embedding=self._embeddings,
        )
        save_dir = str(self._config.vector_store_dir)
        os.makedirs(save_dir, exist_ok=True)
        store.save_local(save_dir)
        return store

    def _try_load_existing(self) -> None:
        """Attempt to load a previously persisted vector store."""
        backend = self._config.vector_store_type
        store_dir = self._config.vector_store_dir

        if not store_dir.exists():
            logger.debug("No existing vector store found — will build on first use")
            return

        try:
            if backend == "chroma":
                from langchain_community.vectorstores import Chroma  # type: ignore

                self._store = Chroma(
                    collection_name=self._config.chroma_collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=str(store_dir),
                )
                self._initialized = True
                logger.info("Loaded existing ChromaDB vector store from disk")
            elif backend == "faiss":
                from langchain_community.vectorstores import FAISS  # type: ignore

                self._store = FAISS.load_local(
                    str(store_dir),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._initialized = True
                logger.info("Loaded existing FAISS vector store from disk")
        except Exception as exc:
            logger.warning(f"Could not load existing vector store: {exc}")
            self._initialized = False

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def search_by_law_name(
        self, law_name: str, k: Optional[int] = None
    ) -> list[Document]:
        """
        Retrieve all indexed chunks that belong to a specific law.

        Uses a metadata filter so only exact law name matches are returned.

        Args:
            law_name: Korean law name, e.g. "민법", "근로기준법".
            k: Maximum number of documents to return.

        Returns:
            List of matching Documents.
        """
        self._require_initialized()
        k = k or self._config.retriever_k

        logger.debug(f"search_by_law_name: law='{law_name}', k={k}")

        # ChromaDB supports metadata filtering natively
        if self._config.vector_store_type == "chroma":
            from langchain_community.vectorstores import Chroma  # type: ignore

            assert isinstance(self._store, Chroma)
            results = self._store.get(
                where={"law_name": {"$eq": law_name}},
                limit=k,
                include=["documents", "metadatas"],
            )
            docs: list[Document] = []
            for content, metadata in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                if content:
                    docs.append(Document(page_content=content, metadata=metadata or {}))
            return docs[:k]
        else:
            # FAISS: use similarity search and post-filter
            query = f"{law_name} 법률 조항"
            results = self._store.similarity_search(query, k=k * 3)  # type: ignore
            filtered = [
                doc
                for doc in results
                if doc.metadata.get("law_name") == law_name
            ]
            return filtered[:k]

    def search_by_article(
        self, law_name: str, article_number: str, k: int = 3
    ) -> list[Document]:
        """
        Look up a specific article within a law.

        Args:
            law_name: e.g. "민법"
            article_number: Numeric part of the article, e.g. "750" (for 제750조).
            k: Maximum results.

        Returns:
            Documents matching that article.
        """
        self._require_initialized()

        logger.debug(
            f"search_by_article: law='{law_name}', article='{article_number}'"
        )

        if self._config.vector_store_type == "chroma":
            from langchain_community.vectorstores import Chroma  # type: ignore

            assert isinstance(self._store, Chroma)
            results = self._store.get(
                where={
                    "$and": [
                        {"law_name": {"$eq": law_name}},
                        {"article_number": {"$eq": article_number}},
                    ]
                },
                limit=k,
                include=["documents", "metadatas"],
            )
            docs: list[Document] = []
            for content, metadata in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                if content:
                    docs.append(Document(page_content=content, metadata=metadata or {}))
            return docs

        else:
            # FAISS fallback: use article label as query
            article_label = f"제{article_number}조"
            query = f"{law_name} {article_label}"
            results = self._store.similarity_search(query, k=k * 3)  # type: ignore
            return [
                doc
                for doc in results
                if doc.metadata.get("article_number") == article_number
                and doc.metadata.get("law_name") == law_name
            ][:k]

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """
        Perform a hybrid semantic search with optional metadata filtering.

        Returns documents sorted by semantic similarity descending.

        Args:
            query: The search query string.
            k: Number of documents to return.
            filter_metadata: Optional dict of metadata key/value pairs to filter by.
            score_threshold: If set, only return documents with similarity ≥ threshold.

        Returns:
            Ranked list of Documents.
        """
        self._require_initialized()
        k = k or self._config.retriever_k
        threshold = score_threshold if score_threshold is not None else 0.0

        logger.debug(f"hybrid_search: query='{query[:60]}…', k={k}")

        try:
            docs_with_scores = self._store.similarity_search_with_score(  # type: ignore
                query, k=k * 2
            )
        except Exception as exc:
            logger.error(f"Similarity search failed: {exc}")
            return []

        results: list[Document] = []
        for doc, score in docs_with_scores:
            # ChromaDB returns L2 distance (lower = better); normalise to 0-1
            # FAISS also returns L2 distance by default
            normalised_score = 1.0 / (1.0 + float(score))

            if normalised_score < threshold:
                continue

            # Apply metadata filter
            if filter_metadata:
                meta = doc.metadata or {}
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            doc.metadata["relevance_score"] = round(normalised_score, 4)
            results.append(doc)

        # Sort by score descending
        results.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return results[:k]

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> list[Document]:
        """Simple similarity search (delegates to hybrid_search)."""
        return self.hybrid_search(query, k=k)

    def get_legal_retriever(
        self,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Return a LangChain-compatible retriever object.

        The returned retriever can be used in LangChain expression language
        (LCEL) chains.

        Args:
            k: Number of documents to retrieve per query.
            score_threshold: Minimum relevance score filter.

        Returns:
            A BaseRetriever instance backed by the vector store.
        """
        self._require_initialized()
        k = k or self._config.retriever_k

        search_kwargs: dict[str, Any] = {"k": k}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        return self._store.as_retriever(  # type: ignore
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add new documents to an existing vector store (incremental update).

        Args:
            documents: New Documents to add.
        """
        self._require_initialized()
        self._store.add_documents(documents)  # type: ignore
        logger.info(f"Added {len(documents)} documents to vector store")

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the collection."""
        if not self._initialized or self._store is None:
            return {"initialized": False, "count": 0}

        try:
            if self._config.vector_store_type == "chroma":
                from langchain_community.vectorstores import Chroma  # type: ignore

                assert isinstance(self._store, Chroma)
                count = self._store._collection.count()
                return {
                    "initialized": True,
                    "backend": "chroma",
                    "collection_name": self._config.chroma_collection_name,
                    "count": count,
                    "persist_directory": str(self._config.vector_store_dir),
                }
            else:
                return {
                    "initialized": True,
                    "backend": "faiss",
                    "count": self._store.index.ntotal,  # type: ignore
                }
        except Exception as exc:
            logger.warning(f"Could not retrieve collection stats: {exc}")
            return {"initialized": True, "count": -1}

    def _require_initialized(self) -> None:
        """Raise RuntimeError if the store has not been built yet."""
        if not self._initialized or self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. "
                "Call build_from_documents() or ensure a persisted store exists."
            )
