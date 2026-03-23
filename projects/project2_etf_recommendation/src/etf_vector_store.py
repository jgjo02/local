"""ETF 벡터 저장소 - 하이브리드 검색 (BM25 + FAISS)"""

import os
import pickle
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from .config import Config
from .etf_data_collector import ETFDataCollector


class ETFVectorStore:
    """ETF 하이브리드 검색 벡터 저장소"""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
        )
        self.faiss_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.documents: List[Document] = []

    def build_from_collector(self, collector: ETFDataCollector) -> None:
        """ETFDataCollector로부터 벡터 스토어 구축"""
        self.documents = collector.create_documents()
        self._build_faiss()
        self._build_bm25()
        print("✅ 하이브리드 검색 인덱스 구축 완료")

    def _build_faiss(self) -> None:
        """FAISS 벡터 인덱스 구축"""
        print("🔧 FAISS 인덱스 구축 중...")
        self.faiss_store = FAISS.from_documents(self.documents, self.embeddings)
        print(f"  ✅ FAISS: {len(self.documents)}개 벡터 인덱싱")

    def _build_bm25(self) -> None:
        """BM25 키워드 인덱스 구축"""
        print("🔧 BM25 인덱스 구축 중...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = self.config.top_k
        print(f"  ✅ BM25: {len(self.documents)}개 문서 인덱싱")

    def save(self, path: Optional[str] = None) -> None:
        """벡터 스토어 저장"""
        save_path = path or self.config.vector_store_path
        os.makedirs(save_path, exist_ok=True)

        if self.faiss_store:
            self.faiss_store.save_local(save_path)
            print(f"✅ FAISS 저장: {save_path}")

        if self.documents:
            docs_path = os.path.join(save_path, "documents.pkl")
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"✅ Documents 저장: {docs_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """저장된 벡터 스토어 로드"""
        load_path = path or self.config.vector_store_path
        faiss_index = os.path.join(load_path, "index.faiss")

        if not os.path.exists(faiss_index):
            return False

        try:
            self.faiss_store = FAISS.load_local(
                load_path, self.embeddings, allow_dangerous_deserialization=True
            )
            docs_path = os.path.join(load_path, "documents.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                self._build_bm25()
            print(f"✅ 벡터 스토어 로드 완료: {len(self.documents)}개 ETF")
            return True
        except Exception as e:
            print(f"⚠️ 로드 실패: {e}")
            return False

    def get_hybrid_retriever(self) -> EnsembleRetriever:
        """하이브리드 검색기 반환 (BM25 + Dense Vector)"""
        if not self.faiss_store or not self.bm25_retriever:
            raise RuntimeError("벡터 스토어가 초기화되지 않았습니다.")

        faiss_retriever = self.faiss_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )

        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[self.config.bm25_weight, self.config.vector_weight],
        )
        return ensemble

    def get_compressed_retriever(self) -> ContextualCompressionRetriever:
        """임베딩 기반 필터링이 적용된 압축 검색기"""
        base_retriever = self.get_hybrid_retriever()
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.75,
        )
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever,
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """순수 벡터 유사도 검색"""
        if not self.faiss_store:
            raise RuntimeError("FAISS 스토어가 초기화되지 않았습니다.")
        return self.faiss_store.similarity_search(query, k=k)

    def hybrid_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """하이브리드 검색 실행"""
        retriever = self.get_hybrid_retriever()
        docs = retriever.invoke(query)
        if k:
            docs = docs[:k]
        return docs

    def get_all_documents(self) -> List[Document]:
        """모든 ETF 문서 반환"""
        return self.documents
