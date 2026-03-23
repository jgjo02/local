"""
Multimodal Processor: CLIP embeddings + GPT-4V for image understanding.

Handles:
- CLIP image and text encoding for cross-modal retrieval
- GPT-4V based image description
- Chart data extraction (numerical values from visualizations)
- Table OCR from images
- Unified multimodal document creation
"""

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config


@dataclass
class MultimodalDocument:
    """Document combining text, image, and caption information."""
    doc_id: str
    text: str
    image_path: Optional[str] = None
    caption: Optional[str] = None
    image_description: Optional[str] = None
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    chart_data: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "image_path": self.image_path,
            "caption": self.caption,
            "image_description": self.image_description,
            "chart_data": self.chart_data,
            "metadata": self.metadata,
            # embeddings are not serialized to dict
        }


@dataclass
class ChartData:
    """Extracted numerical data from a chart."""
    chart_type: str  # bar, line, pie, scatter, etc.
    title: Optional[str]
    x_axis_label: Optional[str]
    y_axis_label: Optional[str]
    series: list[dict]  # [{"name": ..., "values": [...], "labels": [...]}]
    raw_description: str


class MultimodalProcessor:
    """
    Multimodal processing pipeline using CLIP and GPT-4V.

    Provides unified embedding space for text and images, enabling
    cross-modal search and retrieval for company reports.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.clip.device
        self.clip_model_name = config.clip.model_name
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._openai_client = None
        self._doc_counter = 0

        # Embedding dimensions
        self.text_embedding_dim = 1536  # text-embedding-3-large
        self.clip_embedding_dim = 768   # ViT-L/14

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=config.openai.api_key)
        return self._openai_client

    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is not None:
            return

        try:
            import open_clip

            logger.info(f"Loading CLIP model: {self.clip_model_name} on {self.device}")
            # Map short name to open_clip format
            model_name_map = {
                "ViT-L/14": ("ViT-L-14", "openai"),
                "ViT-B/32": ("ViT-B-32", "openai"),
                "ViT-B/16": ("ViT-B-16", "openai"),
            }
            arch, pretrained = model_name_map.get(
                self.clip_model_name, ("ViT-L-14", "openai")
            )

            self._clip_model, _, self._clip_preprocess = (
                open_clip.create_model_and_transforms(
                    arch, pretrained=pretrained, device=self.device
                )
            )
            self._clip_tokenizer = open_clip.get_tokenizer(arch)
            self._clip_model.eval()
            logger.info("CLIP model loaded successfully")

        except ImportError:
            logger.warning(
                "open_clip not available. Trying original CLIP library..."
            )
            try:
                import clip

                self._clip_model, self._clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
                self._clip_tokenizer = clip.tokenize
                self._clip_model.eval()
                logger.info("CLIP (openai) model loaded successfully")
            except ImportError:
                logger.error(
                    "Neither open_clip nor clip is installed. "
                    "Install: pip install open-clip-torch"
                )
                raise

    def encode_image_clip(
        self, image_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Encode an image using CLIP to get a semantic embedding.

        Args:
            image_path: Path to image file (PNG, JPG, etc.)

        Returns:
            Normalized embedding vector of shape (embedding_dim,)
        """
        import torch
        from PIL import Image

        self._load_clip()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_tensor)
            # Normalize
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        embedding = image_features.cpu().numpy().squeeze()
        return embedding

    def encode_text_clip(self, text: str) -> np.ndarray:
        """
        Encode text using CLIP to get an embedding in the same space as images.

        Args:
            text: Input text (truncated to 77 tokens by CLIP)

        Returns:
            Normalized embedding vector of shape (embedding_dim,)
        """
        import torch

        self._load_clip()

        # Truncate text to avoid CLIP token limit
        text = text[:300]

        tokens = self._clip_tokenizer([text]).to(self.device)

        with torch.no_grad():
            text_features = self._clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        embedding = text_features.cpu().numpy().squeeze()
        return embedding

    def encode_text_openai(self, text: str) -> np.ndarray:
        """
        Encode text using OpenAI text-embedding-3-large.

        Args:
            text: Input text

        Returns:
            Embedding vector of shape (1536,)
        """
        response = self.openai_client.embeddings.create(
            model=config.openai.embedding_model,
            input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def encode_text_batch_openai(self, texts: list[str]) -> np.ndarray:
        """
        Batch encode texts using OpenAI embeddings.

        Args:
            texts: List of texts

        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        response = self.openai_client.embeddings.create(
            model=config.openai.embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def describe_image_with_vlm(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a detailed description of an image using GPT-4o vision.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt. Defaults to Korean business report context.

        Returns:
            Text description of the image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image as base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        if prompt is None:
            prompt = (
                "이 이미지는 한국 기업의 사업보고서에서 추출되었습니다. "
                "이미지의 내용을 상세하게 설명해 주세요. "
                "차트나 그래프가 있다면 데이터 트렌드, 수치, 레이블을 포함하여 설명하세요. "
                "표가 있다면 행과 열의 구조와 주요 데이터를 설명하세요. "
                "비즈니스적으로 중요한 인사이트도 포함해 주세요."
            )

        response = self.openai_client.chat.completions.create(
            model=config.openai.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1500,
        )

        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def extract_chart_data(
        self, image_path: Union[str, Path]
    ) -> ChartData:
        """
        Extract structured numerical data from a chart image.

        Uses GPT-4V to interpret chart visualizations and return
        structured data including series names, labels, and values.

        Args:
            image_path: Path to chart image

        Returns:
            ChartData with extracted series and metadata
        """
        image_path = Path(image_path)

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        extraction_prompt = """이 차트 이미지에서 데이터를 추출하세요.
다음 JSON 형식으로 응답하세요:
{
  "chart_type": "bar|line|pie|scatter|area|unknown",
  "title": "차트 제목",
  "x_axis_label": "X축 레이블",
  "y_axis_label": "Y축 레이블",
  "series": [
    {
      "name": "시리즈 이름",
      "labels": ["레이블1", "레이블2", ...],
      "values": [숫자1, 숫자2, ...]
    }
  ]
}
값을 읽을 수 없으면 null을 사용하세요. JSON만 응답하세요."""

        response = self.openai_client.chat.completions.create(
            model=config.openai.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": extraction_prompt},
                    ],
                }
            ],
            max_tokens=1000,
        )

        raw_text = response.choices[0].message.content or "{}"

        # Extract JSON from response
        try:
            # Remove markdown code blocks if present
            json_text = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
            data = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse chart data JSON: {raw_text[:200]}")
            data = {}

        return ChartData(
            chart_type=data.get("chart_type", "unknown"),
            title=data.get("title"),
            x_axis_label=data.get("x_axis_label"),
            y_axis_label=data.get("y_axis_label"),
            series=data.get("series", []),
            raw_description=raw_text,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def extract_table_from_image(
        self, image_path: Union[str, Path]
    ) -> list[list[str]]:
        """
        Extract tabular data from an image using GPT-4V.

        Handles cases where tables are embedded as images rather than
        structured text in the PDF.

        Args:
            image_path: Path to image containing a table

        Returns:
            List of rows, each row is a list of cell strings
        """
        image_path = Path(image_path)

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = image_path.suffix.lower()
        mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        prompt = """이 이미지에서 표(테이블) 데이터를 추출하세요.
다음 JSON 형식으로 응답하세요:
{
  "headers": ["열1", "열2", "열3", ...],
  "rows": [
    ["값1", "값2", "값3", ...],
    ...
  ]
}
표가 없으면 {"headers": [], "rows": []}를 반환하세요. JSON만 응답하세요."""

        response = self.openai_client.chat.completions.create(
            model=config.openai.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=2000,
        )

        raw_text = response.choices[0].message.content or "{}"
        json_text = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()

        try:
            data = json.loads(json_text)
            headers = data.get("headers", [])
            rows = data.get("rows", [])
            if headers:
                return [headers] + rows
            return rows
        except json.JSONDecodeError:
            logger.warning(f"Could not parse table JSON: {raw_text[:200]}")
            return []

    def extract_table_with_ocr(
        self, image_path: Union[str, Path]
    ) -> list[list[str]]:
        """
        Extract table data from image using pytesseract OCR (offline).

        Args:
            image_path: Path to image

        Returns:
            List of rows with cell values
        """
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            # OCR with Korean + English
            text = pytesseract.image_to_string(
                image,
                lang="kor+eng",
                config="--psm 6",
            )

            # Simple row/column parsing
            rows = []
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    # Split on multiple spaces or tabs
                    cells = re.split(r"\s{2,}|\t", line)
                    rows.append([c.strip() for c in cells if c.strip()])

            return rows

        except ImportError:
            logger.warning("pytesseract not available for OCR")
            return []

    def create_multimodal_document(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        caption: Optional[str] = None,
        generate_description: bool = False,
        extract_charts: bool = False,
    ) -> MultimodalDocument:
        """
        Create a unified multimodal document combining text and image.

        Optionally generates image descriptions and chart data using GPT-4V.

        Args:
            text: Text content
            image_path: Optional image path
            caption: Optional pre-existing caption
            generate_description: If True, use GPT-4V to describe the image
            extract_charts: If True, extract chart data from image

        Returns:
            MultimodalDocument with optional embeddings
        """
        self._doc_counter += 1
        doc_id = f"mmdoc_{self._doc_counter:06d}"

        image_path_str = str(image_path) if image_path else None
        image_description = None
        chart_data = None

        if image_path and Path(image_path).exists():
            if generate_description:
                try:
                    image_description = self.describe_image_with_vlm(image_path)
                except Exception as e:
                    logger.warning(f"Could not generate image description: {e}")

            if extract_charts:
                try:
                    chart_result = self.extract_chart_data(image_path)
                    chart_data = {
                        "chart_type": chart_result.chart_type,
                        "title": chart_result.title,
                        "x_axis_label": chart_result.x_axis_label,
                        "y_axis_label": chart_result.y_axis_label,
                        "series": chart_result.series,
                    }
                except Exception as e:
                    logger.warning(f"Could not extract chart data: {e}")

        # Build combined text for the document
        combined_text_parts = [text]
        if caption:
            combined_text_parts.append(f"캡션: {caption}")
        if image_description:
            combined_text_parts.append(f"이미지 설명: {image_description}")
        combined_text = "\n".join(combined_text_parts)

        return MultimodalDocument(
            doc_id=doc_id,
            text=combined_text,
            image_path=image_path_str,
            caption=caption,
            image_description=image_description,
            chart_data=chart_data,
            metadata={
                "has_image": image_path_str is not None,
                "has_chart_data": chart_data is not None,
                "original_text": text,
            },
        )

    def batch_encode_images(
        self, image_paths: list[Union[str, Path]]
    ) -> np.ndarray:
        """
        Batch encode multiple images using CLIP.

        Args:
            image_paths: List of image paths

        Returns:
            Array of shape (n_images, clip_embedding_dim)
        """
        import torch
        from PIL import Image

        self._load_clip()

        embeddings = []
        for path in image_paths:
            try:
                emb = self.encode_image_clip(path)
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Failed to encode image {path}: {e}")
                # Use zero vector as fallback
                zero = np.zeros(self.clip_embedding_dim, dtype=np.float32)
                embeddings.append(zero)

        return np.array(embeddings, dtype=np.float32)

    def compute_image_text_similarity(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between image and text CLIP embeddings.

        Args:
            image_embedding: CLIP image embedding (normalized)
            text_embedding: CLIP text embedding (normalized)

        Returns:
            Similarity score in [-1, 1]
        """
        # Both should be normalized; dot product = cosine similarity
        return float(np.dot(image_embedding, text_embedding))

    def get_most_relevant_images(
        self,
        query: str,
        image_paths: list[Union[str, Path]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find most relevant images for a text query using CLIP.

        Args:
            query: Text query
            image_paths: List of candidate image paths
            top_k: Number of top results to return

        Returns:
            List of (image_path, similarity_score) sorted by score descending
        """
        if not image_paths:
            return []

        query_embedding = self.encode_text_clip(query)
        image_embeddings = self.batch_encode_images(image_paths)

        # Compute similarities
        similarities = image_embeddings @ query_embedding

        # Sort by similarity
        indexed = list(enumerate(similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:top_k]:
            results.append((str(image_paths[idx]), float(score)))

        return results
