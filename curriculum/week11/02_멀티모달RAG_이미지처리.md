# Week 11-2: 멀티모달 RAG 및 이미지 처리

## 학습 목표
- 멀티모달 RAG 아키텍처 이해
- CLIP 임베딩으로 이미지-텍스트 통합 검색
- GPT-4o Vision으로 이미지 분석

---

## 1. 멀티모달 RAG란?

### 1.1 기존 RAG vs 멀티모달 RAG

```
텍스트 RAG:
문서 텍스트 → 임베딩 → FAISS → 텍스트 답변

멀티모달 RAG:
문서 텍스트 + 이미지 + 표 → 멀티모달 임베딩
→ 통합 검색 → Vision LLM → 멀티모달 답변
```

### 1.2 활용 사례

- 사업보고서의 차트/그래프 분석
- 제품 이미지 + 설명 통합 검색
- 의료 이미지 + 텍스트 진단
- 법률 문서의 인감/서명 검증

---

## 2. CLIP 임베딩

### 2.1 CLIP 기본 사용법

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path: str) -> np.ndarray:
    """이미지 임베딩 생성"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.numpy()[0]

def get_text_embedding(text: str) -> np.ndarray:
    """텍스트 CLIP 임베딩 생성"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.numpy()[0]

def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    norm1 = emb1 / np.linalg.norm(emb1)
    norm2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(norm1, norm2))
```

### 2.2 이미지-텍스트 교차 검색

```python
# 텍스트로 이미지 검색
query = "반도체 공장 시설"
query_emb = get_text_embedding(query)

image_paths = ["chart1.png", "factory.png", "report.png"]
image_embs = [get_image_embedding(p) for p in image_paths]

scores = [(path, compute_similarity(query_emb, emb))
          for path, emb in zip(image_paths, image_embs)]
scores.sort(key=lambda x: -x[1])

print("가장 관련 있는 이미지:")
for path, score in scores[:3]:
    print(f"  {path}: {score:.3f}")
```

---

## 3. GPT-4o Vision 분석

### 3.1 이미지 분석

```python
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_with_gpt4o(image_path: str, question: str) -> str:
    """GPT-4o Vision으로 이미지 분석"""
    llm = ChatOpenAI(model="gpt-4o", max_tokens=1000)

    image_data = encode_image_base64(image_path)

    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": "high",
            }
        }
    ])

    response = llm.invoke([message])
    return response.content

# 사업보고서 차트 분석
result = analyze_image_with_gpt4o(
    "quarterly_revenue_chart.png",
    "이 차트에서 매출 추이를 분석하고 주요 인사이트를 3가지 제시하세요."
)
print(result)
```

### 3.2 표 이미지 → 구조화 데이터

```python
import json

def extract_table_from_image(image_path: str) -> dict:
    """이미지의 표를 JSON으로 추출"""
    question = """이 이미지의 표 데이터를 JSON으로 추출하세요.
    형식: {"headers": [...], "rows": [[...], ...]}"""

    result = analyze_image_with_gpt4o(image_path, question)

    try:
        # JSON 추출
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
    except:
        pass

    return {"headers": [], "rows": [], "raw": result}
```

---

## 4. 멀티모달 벡터 스토어

```python
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Union

@dataclass
class MultimodalDocument:
    content: str           # 텍스트 내용 또는 이미지 경로
    content_type: str      # "text" or "image"
    metadata: dict
    embedding: np.ndarray = None

class MultimodalVectorStore:
    def __init__(self, embedding_dim: int = 512):
        self.dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[MultimodalDocument] = []

    def add_text(self, text: str, metadata: dict = None) -> None:
        emb = get_text_embedding(text)
        emb = emb / np.linalg.norm(emb)  # 정규화
        doc = MultimodalDocument(
            content=text,
            content_type="text",
            metadata=metadata or {},
            embedding=emb,
        )
        self.documents.append(doc)
        self.index.add(emb.reshape(1, -1).astype(np.float32))

    def add_image(self, image_path: str, metadata: dict = None) -> None:
        emb = get_image_embedding(image_path)
        emb = emb / np.linalg.norm(emb)
        doc = MultimodalDocument(
            content=image_path,
            content_type="image",
            metadata=metadata or {},
            embedding=emb,
        )
        self.documents.append(doc)
        self.index.add(emb.reshape(1, -1).astype(np.float32))

    def search(self, query: str, k: int = 5) -> List[MultimodalDocument]:
        """텍스트 쿼리로 텍스트+이미지 통합 검색"""
        query_emb = get_text_embedding(query)
        query_emb = query_emb / np.linalg.norm(query_emb)

        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype(np.float32), k
        )

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results
```

---

## 5. 멀티모달 RAG 체인

```python
def multimodal_rag_chain(query: str, store: MultimodalVectorStore) -> str:
    """멀티모달 RAG: 텍스트+이미지 통합 검색 후 답변"""
    docs = store.search(query, k=5)

    text_contexts = []
    image_contents = []

    for doc in docs:
        if doc.content_type == "text":
            text_contexts.append(doc.content)
        elif doc.content_type == "image":
            image_contents.append(doc.content)

    # 텍스트 컨텍스트 구성
    text_context = "\n\n".join(text_contexts[:3])

    # Vision LLM으로 이미지 분석 (첫 번째 관련 이미지)
    image_analysis = ""
    if image_contents:
        image_analysis = analyze_image_with_gpt4o(
            image_contents[0],
            f"이 이미지를 '{query}'에 관련하여 설명해주세요."
        )

    # 통합 답변 생성
    llm = ChatOpenAI(model="gpt-4o-mini")
    final_prompt = f"""다음 정보를 바탕으로 질문에 답하세요.

텍스트 자료:
{text_context}

이미지 분석:
{image_analysis}

질문: {query}"""

    response = llm.invoke([HumanMessage(content=final_prompt)])
    return response.content
```

---

## 핵심 정리

**멀티모달 RAG 구성요소**:
1. **CLIP**: 이미지-텍스트 공유 임베딩 공간
2. **GPT-4o Vision**: 이미지 내용 이해 및 분석
3. **통합 벡터 스토어**: 텍스트+이미지 혼합 인덱스
4. **교차 검색**: 텍스트로 이미지 검색, 이미지로 텍스트 검색

**Project 4 활용**:
- 사업보고서의 재무 차트 자동 분석
- 제품 사진과 텍스트 설명 통합 검색
