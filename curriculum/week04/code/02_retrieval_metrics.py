"""
Week 4 - 2차시: 정보 검색 평가 지표 구현
=====================================
Hit Rate, MRR, NDCG, MAP 등 주요 검색 평가 지표를 완전히 구현합니다.
실제 RAG 시스템 평가에 바로 적용 가능한 코드입니다.
"""

import math
import numpy as np
from typing import List, Dict, Set, Optional, Tuple, Union
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────
# 1. 데이터 구조 정의
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """단일 쿼리에 대한 검색 결과"""
    query: str
    retrieved_docs: List[str]          # 검색된 문서 ID 목록 (순위 순서)
    relevant_docs: Set[str]            # 실제 관련 문서 ID 집합
    relevance_scores: Dict[str, float] # 각 문서의 관련성 점수 (0~3)


# ─────────────────────────────────────────────────────────────
# 2. 기본 지표 구현
# ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K: 상위 K개 결과 중 관련 문서 비율

    Args:
        retrieved: 검색된 문서 ID 목록 (순위 순서)
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        0~1 사이의 Precision 값
    """
    if k <= 0:
        raise ValueError("K는 양수여야 합니다.")

    top_k = retrieved[:k]
    if len(top_k) == 0:
        return 0.0

    relevant_count = sum(1 for doc in top_k if doc in relevant)
    return relevant_count / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K: 상위 K개 결과 중 찾은 관련 문서의 비율

    Args:
        retrieved: 검색된 문서 ID 목록
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        0~1 사이의 Recall 값
    """
    if len(relevant) == 0:
        return 0.0

    top_k = retrieved[:k]
    found = sum(1 for doc in top_k if doc in relevant)
    return found / len(relevant)


def f1_score_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    F1@K: Precision@K와 Recall@K의 조화 평균

    Args:
        retrieved: 검색된 문서 ID 목록
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        0~1 사이의 F1 값
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)

    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─────────────────────────────────────────────────────────────
# 3. Hit Rate 구현
# ─────────────────────────────────────────────────────────────

def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    HR@K: 상위 K개 결과 중 하나라도 관련 문서가 있으면 1

    Args:
        retrieved: 검색된 문서 ID 목록
        relevant: 관련 문서 ID 집합
        k: 상위 K개

    Returns:
        1.0 (hit) 또는 0.0 (miss)
    """
    top_k = set(retrieved[:k])
    return 1.0 if top_k & relevant else 0.0


def mean_hit_rate_at_k(
    results: List[QueryResult],
    k: int
) -> float:
    """
    평균 HR@K: 여러 쿼리에 대한 Hit Rate 평균

    Args:
        results: QueryResult 목록
        k: 상위 K개

    Returns:
        평균 Hit Rate
    """
    if not results:
        return 0.0

    hits = [
        hit_rate_at_k(r.retrieved_docs, r.relevant_docs, k)
        for r in results
    ]
    return np.mean(hits)


# ─────────────────────────────────────────────────────────────
# 4. MRR 구현
# ─────────────────────────────────────────────────────────────

def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Reciprocal Rank: 첫 번째 관련 문서 순위의 역수

    Args:
        retrieved: 검색된 문서 ID 목록 (순위 순서)
        relevant: 관련 문서 ID 집합

    Returns:
        1/rank 값 (관련 문서 없으면 0)
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(results: List[QueryResult]) -> float:
    """
    MRR: 여러 쿼리에 대한 Reciprocal Rank 평균

    Args:
        results: QueryResult 목록

    Returns:
        MRR 값
    """
    if not results:
        return 0.0

    rr_scores = [
        reciprocal_rank(r.retrieved_docs, r.relevant_docs)
        for r in results
    ]
    return np.mean(rr_scores)


# ─────────────────────────────────────────────────────────────
# 5. NDCG 구현
# ─────────────────────────────────────────────────────────────

def dcg_at_k(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    DCG@K: Discounted Cumulative Gain

    공식: DCG@K = Σ rel_i / log2(i+1) for i=1..K

    Args:
        retrieved: 검색된 문서 ID 목록
        relevance_scores: 각 문서의 관련성 점수
        k: 상위 K개

    Returns:
        DCG 값
    """
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc, 0.0)
        # i=0부터 시작하므로 log2(i+2) 사용
        dcg += rel / math.log2(i + 2)
    return dcg


def idcg_at_k(relevance_scores: Dict[str, float], k: int) -> float:
    """
    IDCG@K: Ideal DCG - 최적 순위일 때의 DCG

    Args:
        relevance_scores: 모든 문서의 관련성 점수
        k: 상위 K개

    Returns:
        IDCG 값
    """
    # 관련성 점수를 내림차순 정렬
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)
    return idcg


def ndcg_at_k(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain

    공식: NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: 검색된 문서 ID 목록
        relevance_scores: 각 문서의 관련성 점수 (높을수록 더 관련)
        k: 상위 K개

    Returns:
        0~1 사이의 NDCG 값
    """
    dcg = dcg_at_k(retrieved, relevance_scores, k)
    idcg = idcg_at_k(relevance_scores, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_ndcg_at_k(results: List[QueryResult], k: int) -> float:
    """
    평균 NDCG@K: 여러 쿼리에 대한 NDCG 평균

    Args:
        results: QueryResult 목록
        k: 상위 K개

    Returns:
        평균 NDCG 값
    """
    if not results:
        return 0.0

    ndcg_scores = [
        ndcg_at_k(r.retrieved_docs, r.relevance_scores, k)
        for r in results
    ]
    return np.mean(ndcg_scores)


# ─────────────────────────────────────────────────────────────
# 6. MAP 구현
# ─────────────────────────────────────────────────────────────

def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    AP (Average Precision): 각 관련 문서 발견 시점의 Precision 평균

    공식: AP = (1/|R|) Σ P(k) * rel(k)

    Args:
        retrieved: 검색된 문서 ID 목록
        relevant: 관련 문서 ID 집합

    Returns:
        AP 값
    """
    if not relevant:
        return 0.0

    num_relevant = 0
    cumulative_precision = 0.0

    for i, doc in enumerate(retrieved):
        if doc in relevant:
            num_relevant += 1
            precision_at_this_rank = num_relevant / (i + 1)
            cumulative_precision += precision_at_this_rank

    if num_relevant == 0:
        return 0.0

    return cumulative_precision / len(relevant)


def mean_average_precision(results: List[QueryResult]) -> float:
    """
    MAP: 여러 쿼리에 대한 Average Precision 평균

    Args:
        results: QueryResult 목록

    Returns:
        MAP 값
    """
    if not results:
        return 0.0

    ap_scores = [
        average_precision(r.retrieved_docs, r.relevant_docs)
        for r in results
    ]
    return np.mean(ap_scores)


# ─────────────────────────────────────────────────────────────
# 7. 종합 평가 클래스
# ─────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """검색 성능을 종합적으로 평가하는 클래스"""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Args:
            k_values: 평가할 K 값 목록
        """
        self.k_values = k_values
        self.results: List[QueryResult] = []

    def add_result(self, result: QueryResult):
        """평가 결과 추가"""
        self.results.append(result)

    def evaluate(self) -> Dict[str, float]:
        """
        모든 지표를 종합 계산

        Returns:
            지표명 → 값 딕셔너리
        """
        metrics = {}

        # Hit Rate @ K
        for k in self.k_values:
            metrics[f"HR@{k}"] = mean_hit_rate_at_k(self.results, k)

        # MRR
        metrics["MRR"] = mean_reciprocal_rank(self.results)

        # NDCG @ K
        for k in self.k_values:
            metrics[f"NDCG@{k}"] = mean_ndcg_at_k(self.results, k)

        # MAP
        metrics["MAP"] = mean_average_precision(self.results)

        # Precision @ K
        for k in self.k_values:
            p_scores = [
                precision_at_k(r.retrieved_docs, r.relevant_docs, k)
                for r in self.results
            ]
            metrics[f"P@{k}"] = np.mean(p_scores)

        # Recall @ K
        for k in self.k_values:
            r_scores = [
                recall_at_k(r.retrieved_docs, r.relevant_docs, k)
                for r in self.results
            ]
            metrics[f"R@{k}"] = np.mean(r_scores)

        return metrics

    def print_report(self):
        """평가 결과를 읽기 쉬운 형식으로 출력"""
        metrics = self.evaluate()

        print("\n" + "=" * 60)
        print("📊 검색 성능 평가 리포트")
        print("=" * 60)
        print(f"총 쿼리 수: {len(self.results)}")
        print("-" * 60)

        # Hit Rate 출력
        print("\n[Hit Rate]")
        for k in self.k_values:
            val = metrics[f"HR@{k}"]
            bar = "█" * int(val * 20)
            print(f"  HR@{k:2d}: {val:.4f}  {bar}")

        # MRR 출력
        print("\n[Mean Reciprocal Rank]")
        val = metrics["MRR"]
        bar = "█" * int(val * 20)
        print(f"  MRR:   {val:.4f}  {bar}")

        # NDCG 출력
        print("\n[NDCG]")
        for k in self.k_values:
            val = metrics[f"NDCG@{k}"]
            bar = "█" * int(val * 20)
            print(f"  NDCG@{k:2d}: {val:.4f}  {bar}")

        # MAP 출력
        print("\n[MAP]")
        val = metrics["MAP"]
        bar = "█" * int(val * 20)
        print(f"  MAP:   {val:.4f}  {bar}")

        # Precision/Recall 출력
        print("\n[Precision & Recall]")
        for k in self.k_values:
            p = metrics[f"P@{k}"]
            r = metrics[f"R@{k}"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"  @{k:2d}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

        print("=" * 60)

    def compare_systems(
        self,
        other_evaluator: "RetrievalEvaluator",
        system_names: Tuple[str, str] = ("System A", "System B")
    ):
        """두 검색 시스템의 성능 비교"""
        metrics_a = self.evaluate()
        metrics_b = other_evaluator.evaluate()

        all_keys = [f"HR@{k}" for k in self.k_values] + \
                   ["MRR"] + \
                   [f"NDCG@{k}" for k in self.k_values] + \
                   ["MAP"]

        print("\n" + "=" * 70)
        print(f"📊 검색 시스템 비교: {system_names[0]} vs {system_names[1]}")
        print("=" * 70)
        print(f"{'지표':<12} {system_names[0]:>12} {system_names[1]:>12} {'개선도':>10}")
        print("-" * 70)

        for key in all_keys:
            a_val = metrics_a.get(key, 0)
            b_val = metrics_b.get(key, 0)
            improvement = (b_val - a_val) / a_val * 100 if a_val > 0 else 0
            symbol = "▲" if improvement > 0 else "▼" if improvement < 0 else "─"
            print(f"{key:<12} {a_val:>12.4f} {b_val:>12.4f} {symbol}{abs(improvement):>8.1f}%")

        print("=" * 70)


# ─────────────────────────────────────────────────────────────
# 8. LangChain RAG 시스템 평가 통합
# ─────────────────────────────────────────────────────────────

def evaluate_rag_retriever(
    retriever,
    test_data: List[Dict],
    k: int = 5
) -> Dict[str, float]:
    """
    LangChain Retriever를 평가하는 함수

    Args:
        retriever: LangChain Retriever 객체
        test_data: [{"query": str, "relevant_docs": List[str]}] 형식의 테스트 데이터
        k: 상위 K개

    Returns:
        평가 지표 딕셔너리
    """
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5, 10])

    for item in test_data:
        query = item["query"]
        relevant_doc_ids = set(item["relevant_docs"])

        # Retriever로 문서 검색
        try:
            docs = retriever.get_relevant_documents(query)
            retrieved_ids = [doc.metadata.get("doc_id", doc.page_content[:50])
                           for doc in docs]
        except Exception as e:
            print(f"검색 오류 (쿼리: {query}): {e}")
            retrieved_ids = []

        # 관련성 점수 (이진 관련성)
        relevance_scores = {
            doc_id: 1.0 if doc_id in relevant_doc_ids else 0.0
            for doc_id in retrieved_ids
        }

        result = QueryResult(
            query=query,
            retrieved_docs=retrieved_ids,
            relevant_docs=relevant_doc_ids,
            relevance_scores=relevance_scores
        )
        evaluator.add_result(result)

    evaluator.print_report()
    return evaluator.evaluate()


# ─────────────────────────────────────────────────────────────
# 9. 실행 예시
# ─────────────────────────────────────────────────────────────

def run_demonstration():
    """평가 지표 시연"""
    print("=" * 60)
    print("📊 검색 평가 지표 시연")
    print("=" * 60)

    # 시나리오: ETF 추천 시스템 검색 결과 평가

    # 쿼리 1: ETF 기초 설명 쿼리
    print("\n[쿼리 1] ETF란 무엇인가요?")
    retrieved_1 = ["etf_intro", "stock_basics", "fund_types", "etf_history", "index_funds"]
    relevant_1 = {"etf_intro", "fund_types", "etf_history"}

    print(f"검색 결과: {retrieved_1}")
    print(f"관련 문서: {relevant_1}")
    print(f"HR@5:  {hit_rate_at_k(retrieved_1, relevant_1, 5):.4f}")
    print(f"RR:    {reciprocal_rank(retrieved_1, relevant_1):.4f}")
    print(f"AP:    {average_precision(retrieved_1, relevant_1):.4f}")

    # NDCG를 위한 등급 관련성
    rel_scores_1 = {
        "etf_intro": 3.0,    # 매우 관련
        "stock_basics": 1.0,  # 약간 관련
        "fund_types": 2.0,    # 관련
        "etf_history": 2.0,   # 관련
        "index_funds": 1.0    # 약간 관련
    }
    print(f"NDCG@5: {ndcg_at_k(retrieved_1, rel_scores_1, 5):.4f}")

    # 쿼리 2: 테크 ETF 추천 쿼리
    print("\n[쿼리 2] 미국 기술주에 투자하는 ETF 추천해주세요")
    retrieved_2 = ["bond_etf", "tech_etf_list", "sector_etfs", "qqq_info", "spy_info"]
    relevant_2 = {"tech_etf_list", "qqq_info", "sector_etfs"}

    print(f"검색 결과: {retrieved_2}")
    print(f"관련 문서: {relevant_2}")
    print(f"HR@5:  {hit_rate_at_k(retrieved_2, relevant_2, 5):.4f}")
    print(f"RR:    {reciprocal_rank(retrieved_2, relevant_2):.4f}")
    print(f"AP:    {average_precision(retrieved_2, relevant_2):.4f}")

    rel_scores_2 = {
        "bond_etf": 0.0,
        "tech_etf_list": 3.0,
        "sector_etfs": 2.0,
        "qqq_info": 3.0,
        "spy_info": 1.0
    }
    print(f"NDCG@5: {ndcg_at_k(retrieved_2, rel_scores_2, 5):.4f}")

    # 쿼리 3: 관련 문서 없는 경우
    print("\n[쿼리 3] 부동산 투자 방법 (ETF DB에 없는 쿼리)")
    retrieved_3 = ["etf_intro", "bond_etf", "stock_basics", "index_funds", "fund_types"]
    relevant_3 = set()  # 관련 문서 없음

    print(f"검색 결과: {retrieved_3}")
    print(f"관련 문서: {relevant_3} (없음)")
    print(f"HR@5:  {hit_rate_at_k(retrieved_3, relevant_3, 5):.4f}")
    print(f"RR:    {reciprocal_rank(retrieved_3, relevant_3):.4f}")

    # 종합 평가
    print("\n" + "=" * 60)
    print("종합 평가 (쿼리 1, 2 기준)")
    print("=" * 60)

    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    # 쿼리 1 결과 추가
    evaluator.add_result(QueryResult(
        query="ETF란 무엇인가요?",
        retrieved_docs=retrieved_1,
        relevant_docs=relevant_1,
        relevance_scores=rel_scores_1
    ))

    # 쿼리 2 결과 추가
    evaluator.add_result(QueryResult(
        query="미국 기술주 ETF 추천",
        retrieved_docs=retrieved_2,
        relevant_docs=relevant_2,
        relevance_scores=rel_scores_2
    ))

    evaluator.print_report()

    # 시스템 A vs 시스템 B 비교
    print("\n🔄 시스템 비교: Naive RAG vs Hybrid RAG")

    # 기존 시스템 (Naive RAG) - 위와 동일
    system_a = evaluator

    # 개선된 시스템 (Hybrid RAG) - 더 좋은 결과 시뮬레이션
    system_b = RetrievalEvaluator(k_values=[1, 3, 5])
    system_b.add_result(QueryResult(
        query="ETF란 무엇인가요?",
        retrieved_docs=["etf_intro", "fund_types", "etf_history", "stock_basics", "index_funds"],
        relevant_docs=relevant_1,
        relevance_scores=rel_scores_1
    ))
    system_b.add_result(QueryResult(
        query="미국 기술주 ETF 추천",
        retrieved_docs=["tech_etf_list", "qqq_info", "sector_etfs", "spy_info", "bond_etf"],
        relevant_docs=relevant_2,
        relevance_scores=rel_scores_2
    ))

    system_a.compare_systems(system_b, ("Naive RAG", "Hybrid RAG"))


def demonstrate_k_sensitivity():
    """K 값에 따른 지표 변화 분석"""
    print("\n" + "=" * 60)
    print("K 값에 따른 지표 변화 분석")
    print("=" * 60)

    # 10개 검색 결과: 1위, 4위, 7위에 관련 문서
    retrieved = ["r1", "d2", "d3", "r4", "d5", "d6", "r7", "d8", "d9", "d10"]
    relevant = {"r1", "r4", "r7"}
    rel_scores = {doc: (1.0 if doc.startswith("r") else 0.0) for doc in retrieved}

    print(f"검색 결과: {retrieved}")
    print(f"관련 문서 위치: 1위, 4위, 7위")
    print()
    print(f"{'K':>4} {'HR@K':>8} {'P@K':>8} {'R@K':>8} {'NDCG@K':>8}")
    print("-" * 44)

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        hr = hit_rate_at_k(retrieved, relevant, k)
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        ndcg = ndcg_at_k(retrieved, rel_scores, k)
        print(f"{k:>4} {hr:>8.4f} {p:>8.4f} {r:>8.4f} {ndcg:>8.4f}")


if __name__ == "__main__":
    run_demonstration()
    demonstrate_k_sensitivity()
