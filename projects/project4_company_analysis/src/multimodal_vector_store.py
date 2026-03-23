"""
Multimodal Vector Store using FAISS.

Maintains separate indexes for:
- Text documents (OpenAI embeddings, 1536-dim)
- Image documents (CLIP embeddings, 768-dim)
- Cross-modal text-to-image and image-to-text retrieval

Supports hybrid search combining both modalities.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from loguru import logger

from .config import config
from .multimodal_processor import MultimodalDocument, MultimodalProcessor


@dataclass
class SearchResult:
    """A single result from a vector store search."""
    doc_id: str
    score: float
    text: str
    image_path: Optional[str] = None
    caption: Optional[str] = None
    image_description: Optional[str] = None
    chart_data: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    modality: str = "text"  # "text", "image", or "multimodal"

    def __repr__(self) -> str:
        return (
            f"SearchResult(id={self.doc_id}, score={self.score:.4f}, "
            f"modality={self.modality}, text={self.text[:80]!r})"
        )


class MultimodalVectorStore:
    """
    FAISS-backed vector store for multimodal documents.

    Supports:
    - Text-only documents indexed by OpenAI embeddings
    - Image documents indexed by CLIP embeddings
    - Cross-modal queries (text query -> image results, image -> text)
    - Hybrid search combining both indexes
    """

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        processor: Optional[MultimodalProcessor] = None,
    ):
        self.index_path = Path(index_path or config.paths.faiss_index_path)
        self.processor = processor or MultimodalProcessor()

        # FAISS indexes
        self._text_index = None       # OpenAI embeddings (1536-dim)
        self._image_index = None      # CLIP embeddings (768-dim)
        self._text_clip_index = None  # CLIP text embeddings (768-dim) for cross-modal

        # Document stores (parallel to index entries)
        self._text_docs: list[MultimodalDocument] = []
        self._image_docs: list[MultimodalDocument] = []

        self.text_embedding_dim = 1536
        self.clip_embedding_dim = None  # determined at first encode

    def _get_clip_dim(self) -> int:
        """Get CLIP embedding dimension from model config."""
        if self.clip_embedding_dim is None:
            # ViT-L/14 -> 768, ViT-B/32 -> 512
            dim_map = {
                "ViT-L/14": 768,
                "ViT-B/32": 512,
                "ViT-B/16": 512,
            }
            self.clip_embedding_dim = dim_map.get(
                config.clip.model_name, 768
            )
        return self.clip_embedding_dim

    def _build_faiss_index(self, dim: int, use_ivf: bool = False):
        """Create a FAISS index."""
        import faiss

        if use_ivf:
            # IVF index for large-scale (>10k vectors)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(dim)  # Inner product (for normalized vectors)
        return index

    def add_text_documents(
        self,
        documents: list[MultimodalDocument],
        batch_size: int = 32,
    ) -> None:
        """
        Add text documents to the text FAISS index.

        Uses OpenAI text-embedding-3-large for semantic text search.

        Args:
            documents: List of MultimodalDocument objects
            batch_size: Number of documents to embed at once
        """
        import faiss

        if not documents:
            return

        logger.info(f"Adding {len(documents)} text documents to vector store")

        # Initialize index if needed
        if self._text_index is None:
            self._text_index = self._build_faiss_index(self.text_embedding_dim)

        # Process in batches
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start: batch_start + batch_size]
            texts = [doc.text for doc in batch]

            try:
                embeddings = self.processor.encode_text_batch_openai(texts)
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                # Fallback: encode one by one
                embeddings = np.array([
                    self.processor.encode_text_openai(t) for t in texts
                ])

            # Normalize for cosine similarity via inner product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

            self._text_index.add(embeddings.astype(np.float32))

            # Store documents
            for i, doc in enumerate(batch):
                doc.text_embedding = embeddings[i]
                self._text_docs.append(doc)

            logger.debug(
                f"Processed batch {batch_start // batch_size + 1}, "
                f"total text docs: {len(self._text_docs)}"
            )

    def add_image_documents(
        self,
        image_paths: list[Union[str, Path]],
        captions: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
        generate_descriptions: bool = False,
    ) -> None:
        """
        Add image documents to the CLIP FAISS index.

        Also builds a parallel CLIP text index from captions for cross-modal retrieval.

        Args:
            image_paths: Paths to image files
            captions: Optional captions for each image
            metadata_list: Optional metadata dicts for each image
            generate_descriptions: Use GPT-4V to describe images
        """
        import faiss

        if not image_paths:
            return

        captions = captions or [""] * len(image_paths)
        metadata_list = metadata_list or [{}] * len(image_paths)

        logger.info(f"Adding {len(image_paths)} image documents")

        clip_dim = self._get_clip_dim()

        if self._image_index is None:
            self._image_index = self._build_faiss_index(clip_dim)
        if self._text_clip_index is None:
            self._text_clip_index = self._build_faiss_index(clip_dim)

        for i, (img_path, caption, meta) in enumerate(
            zip(image_paths, captions, metadata_list)
        ):
            img_path = Path(img_path)

            # Create document
            doc = self.processor.create_multimodal_document(
                text=caption or f"이미지 {i}",
                image_path=img_path,
                caption=caption,
                generate_description=generate_descriptions,
            )
            doc.metadata.update(meta)

            # Encode image with CLIP
            try:
                img_emb = self.processor.encode_image_clip(img_path)
                img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
                self._image_index.add(
                    img_emb.reshape(1, -1).astype(np.float32)
                )
                doc.image_embedding = img_emb
            except Exception as e:
                logger.warning(f"Failed to encode image {img_path}: {e}")
                img_emb = np.zeros(clip_dim, dtype=np.float32)
                self._image_index.add(img_emb.reshape(1, -1))

            # Encode caption/description with CLIP text encoder
            text_for_clip = caption or (doc.image_description or f"이미지 {i}")
            try:
                txt_emb = self.processor.encode_text_clip(text_for_clip)
                txt_emb = txt_emb / (np.linalg.norm(txt_emb) + 1e-8)
                self._text_clip_index.add(
                    txt_emb.reshape(1, -1).astype(np.float32)
                )
            except Exception as e:
                logger.warning(f"Failed to encode caption: {e}")
                txt_emb = np.zeros(clip_dim, dtype=np.float32)
                self._text_clip_index.add(txt_emb.reshape(1, -1))

            self._image_docs.append(doc)

            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(image_paths)} images")

    def text_search(self, query: str, k: int = 10) -> list[SearchResult]:
        """
        Search text documents by semantic similarity.

        Args:
            query: Text query
            k: Number of results

        Returns:
            Sorted list of SearchResult
        """
        if self._text_index is None or len(self._text_docs) == 0:
            return []

        import faiss

        k = min(k, len(self._text_docs))
        query_emb = self.processor.encode_text_openai(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        scores, indices = self._text_index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._text_docs):
                continue
            doc = self._text_docs[idx]
            results.append(SearchResult(
                doc_id=doc.doc_id,
                score=float(score),
                text=doc.text,
                image_path=doc.image_path,
                caption=doc.caption,
                image_description=doc.image_description,
                chart_data=doc.chart_data,
                metadata=doc.metadata,
                modality="text",
            ))

        return results

    def text_to_image_search(
        self, query: str, k: int = 5
    ) -> list[SearchResult]:
        """
        Find images relevant to a text query using CLIP.

        Uses CLIP to project the text query into the shared embedding space
        and finds the most similar image embeddings.

        Args:
            query: Text query describing desired images
            k: Number of results

        Returns:
            Sorted list of SearchResult with image paths
        """
        if self._image_index is None or len(self._image_docs) == 0:
            logger.warning("Image index is empty")
            return []

        k = min(k, len(self._image_docs))

        # Encode query with CLIP text encoder
        query_emb = self.processor.encode_text_clip(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        scores, indices = self._image_index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._image_docs):
                continue
            doc = self._image_docs[idx]
            results.append(SearchResult(
                doc_id=doc.doc_id,
                score=float(score),
                text=doc.text,
                image_path=doc.image_path,
                caption=doc.caption,
                image_description=doc.image_description,
                chart_data=doc.chart_data,
                metadata=doc.metadata,
                modality="image",
            ))

        return results

    def image_to_text_search(
        self, image_embedding: np.ndarray, k: int = 5
    ) -> list[SearchResult]:
        """
        Find text documents relevant to an image using CLIP embeddings.

        Projects image embedding into text space via the shared CLIP embedding.

        Args:
            image_embedding: CLIP image embedding (normalized)
            k: Number of results

        Returns:
            List of SearchResult with text documents
        """
        if self._text_clip_index is None or len(self._image_docs) == 0:
            return []

        k = min(k, len(self._image_docs))
        query_emb = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)

        scores, indices = self._text_clip_index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._image_docs):
                continue
            doc = self._image_docs[idx]
            results.append(SearchResult(
                doc_id=doc.doc_id,
                score=float(score),
                text=doc.text,
                image_path=doc.image_path,
                caption=doc.caption,
                image_description=doc.image_description,
                chart_data=doc.chart_data,
                metadata=doc.metadata,
                modality="image",
            ))

        return results

    def hybrid_multimodal_search(
        self,
        query: str,
        k: int = 10,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ) -> list[SearchResult]:
        """
        Hybrid search combining text semantic search and CLIP-based image search.

        Merges results from both modalities using weighted score combination.

        Args:
            query: Text query
            k: Total number of results
            text_weight: Weight for text search scores
            image_weight: Weight for image search scores

        Returns:
            Combined sorted list of SearchResult
        """
        # Get results from both indexes
        text_results = self.text_search(query, k=k)
        image_results = self.text_to_image_search(query, k=k)

        # Normalize scores within each modality to [0, 1]
        def normalize_scores(results: list[SearchResult]) -> list[SearchResult]:
            if not results:
                return results
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                for r in results:
                    r.score = 1.0
            else:
                for r in results:
                    r.score = (r.score - min_s) / (max_s - min_s)
            return results

        text_results = normalize_scores(text_results)
        image_results = normalize_scores(image_results)

        # Apply weights
        combined: dict[str, SearchResult] = {}

        for result in text_results:
            result.score *= text_weight
            combined[result.doc_id] = result

        for result in image_results:
            result.score *= image_weight
            if result.doc_id in combined:
                combined[result.doc_id].score += result.score
                combined[result.doc_id].modality = "multimodal"
            else:
                combined[result.doc_id] = result

        # Sort and return top-k
        sorted_results = sorted(
            combined.values(), key=lambda r: r.score, reverse=True
        )
        return sorted_results[:k]

    def build(
        self,
        text_docs: list[MultimodalDocument],
        image_docs: Optional[list[tuple[Union[str, Path], str]]] = None,
    ) -> None:
        """
        Build the complete vector store from documents.

        Args:
            text_docs: List of MultimodalDocument for text index
            image_docs: Optional list of (image_path, caption) tuples
        """
        logger.info("Building multimodal vector store...")

        if text_docs:
            self.add_text_documents(text_docs)
            logger.info(f"Text index: {len(self._text_docs)} documents")

        if image_docs:
            paths = [p for p, _ in image_docs]
            captions = [c for _, c in image_docs]
            self.add_image_documents(paths, captions)
            logger.info(f"Image index: {len(self._image_docs)} documents")

        logger.info("Vector store built successfully")

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Persist the vector store to disk.

        Args:
            path: Directory to save. Defaults to config.paths.faiss_index_path

        Returns:
            Path to save directory
        """
        import faiss

        save_dir = Path(path or self.index_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._text_index is not None:
            faiss.write_index(
                self._text_index, str(save_dir / "text_index.faiss")
            )
        if self._image_index is not None:
            faiss.write_index(
                self._image_index, str(save_dir / "image_index.faiss")
            )
        if self._text_clip_index is not None:
            faiss.write_index(
                self._text_clip_index, str(save_dir / "text_clip_index.faiss")
            )

        # Save document stores (without embeddings to save space)
        docs_data = {
            "text_docs": [self._serialize_doc(d) for d in self._text_docs],
            "image_docs": [self._serialize_doc(d) for d in self._image_docs],
            "clip_embedding_dim": self.clip_embedding_dim,
        }
        with open(save_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Vector store saved to: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load a previously saved vector store."""
        import faiss

        load_dir = Path(path or self.index_path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Vector store not found: {load_dir}")

        text_idx_path = load_dir / "text_index.faiss"
        if text_idx_path.exists():
            self._text_index = faiss.read_index(str(text_idx_path))

        image_idx_path = load_dir / "image_index.faiss"
        if image_idx_path.exists():
            self._image_index = faiss.read_index(str(image_idx_path))

        clip_idx_path = load_dir / "text_clip_index.faiss"
        if clip_idx_path.exists():
            self._text_clip_index = faiss.read_index(str(clip_idx_path))

        docs_path = load_dir / "documents.json"
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.clip_embedding_dim = data.get("clip_embedding_dim")
            self._text_docs = [
                self._deserialize_doc(d) for d in data.get("text_docs", [])
            ]
            self._image_docs = [
                self._deserialize_doc(d) for d in data.get("image_docs", [])
            ]

        logger.info(
            f"Vector store loaded: {len(self._text_docs)} text docs, "
            f"{len(self._image_docs)} image docs"
        )

    def _serialize_doc(self, doc: MultimodalDocument) -> dict:
        return {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "image_path": doc.image_path,
            "caption": doc.caption,
            "image_description": doc.image_description,
            "chart_data": doc.chart_data,
            "metadata": doc.metadata,
        }

    def _deserialize_doc(self, data: dict) -> MultimodalDocument:
        return MultimodalDocument(
            doc_id=data["doc_id"],
            text=data["text"],
            image_path=data.get("image_path"),
            caption=data.get("caption"),
            image_description=data.get("image_description"),
            chart_data=data.get("chart_data"),
            metadata=data.get("metadata", {}),
        )

    @property
    def text_count(self) -> int:
        return len(self._text_docs)

    @property
    def image_count(self) -> int:
        return len(self._image_docs)

    def stats(self) -> dict:
        return {
            "text_documents": self.text_count,
            "image_documents": self.image_count,
            "text_index_trained": self._text_index is not None,
            "image_index_trained": self._image_index is not None,
            "clip_dim": self.clip_embedding_dim,
            "text_embedding_dim": self.text_embedding_dim,
        }
