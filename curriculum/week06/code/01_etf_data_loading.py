"""Week 6 실습 1: ETF 데이터 로드 및 전처리"""

import json
from typing import List, Dict, Any
from langchain.schema import Document


def load_etf_data(path: str = "../../projects/project2_etf_recommendation/data/sample_etf_data.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["etfs"]


def build_etf_document(etf: Dict) -> Document:
    """ETF 딕셔너리를 LangChain Document로 변환"""
    text = f"""ETF명: {etf['name']} (티커: {etf['ticker']})
카테고리: {etf['category']} | 투자지역: {etf['region']}
위험등급: {etf['risk_level']} | 운용보수: {etf['expense_ratio']}%
운용규모: {etf['aum_billion']:,}억원
수익률 - 1개월: {etf['1m_return']:+.1f}%, 1년: {etf['1y_return']:+.1f}%, 3년: {etf['3y_return']:+.1f}%
설명: {etf['description']}
테마: {', '.join(etf.get('theme', []))}"""

    return Document(
        page_content=text,
        metadata={
            "ticker": etf["ticker"],
            "name": etf["name"],
            "category": etf["category"],
            "region": etf["region"],
            "risk_level": etf["risk_level"],
            "expense_ratio": etf["expense_ratio"],
            "aum_billion": etf["aum_billion"],
            "1y_return": etf["1y_return"],
            "3y_return": etf["3y_return"],
        }
    )


def analyze_etf_data(etfs: List[Dict]) -> None:
    """ETF 데이터 통계 분석"""
    print(f"📊 총 ETF 수: {len(etfs)}")

    # 카테고리별
    categories = {}
    for etf in etfs:
        cat = etf["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\n카테고리별 분포:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}개")

    # 위험등급별
    risk_order = ["저위험", "중립", "중고위험", "고위험", "매우고위험"]
    risk_counts = {r: 0 for r in risk_order}
    for etf in etfs:
        risk = etf["risk_level"]
        if risk in risk_counts:
            risk_counts[risk] += 1
    print("\n위험등급별 분포:")
    for risk, cnt in risk_counts.items():
        if cnt > 0:
            bar = "█" * cnt
            print(f"  {risk:10s}: {bar} ({cnt})")

    # 수익률 TOP 5
    print("\n🏆 1년 수익률 TOP 5:")
    top5 = sorted(etfs, key=lambda x: x["1y_return"], reverse=True)[:5]
    for i, etf in enumerate(top5, 1):
        print(f"  {i}. {etf['name']}: {etf['1y_return']:+.1f}%")

    # 수수료 최저 3개
    print("\n💰 운용보수 최저 3개:")
    cheap = sorted(etfs, key=lambda x: x["expense_ratio"])[:3]
    for etf in cheap:
        print(f"  {etf['name']}: {etf['expense_ratio']}%")


if __name__ == "__main__":
    etfs = load_etf_data()
    analyze_etf_data(etfs)

    print("\n📄 Document 변환 예시:")
    doc = build_etf_document(etfs[0])
    print(f"내용:\n{doc.page_content}")
    print(f"\n메타데이터: {doc.metadata}")
