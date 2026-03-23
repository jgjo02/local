"""ETF 데이터 수집 및 전처리 모듈"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.schema import Document


class ETFDataCollector:
    """ETF 데이터를 로드하고 LangChain Document로 변환하는 클래스"""

    def __init__(self, data_path: str = "data/sample_etf_data.json"):
        self.data_path = data_path
        self.etf_data: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        """JSON 파일에서 ETF 데이터 로드"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"ETF 데이터 파일을 찾을 수 없습니다: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.etf_data = data.get("etfs", [])
        print(f"✅ {len(self.etf_data)}개 ETF 데이터 로드 완료")
        return self.etf_data

    def build_etf_text(self, etf: Dict[str, Any]) -> str:
        """ETF 정보를 검색 가능한 텍스트로 변환"""
        risk_emoji = {
            "저위험": "🟢",
            "중립": "🟡",
            "중고위험": "🟠",
            "고위험": "🔴",
            "매우고위험": "⛔",
        }
        risk_icon = risk_emoji.get(etf.get("risk_level", ""), "⚪")

        themes_str = ", ".join(etf.get("theme", []))

        text = f"""ETF명: {etf['name']} (티커: {etf['ticker']})
카테고리: {etf['category']} | 투자지역: {etf['region']}
위험등급: {risk_icon} {etf['risk_level']} | 운용보수: {etf['expense_ratio']}%
운용규모: {etf['aum_billion']:,}억원

수익률 현황:
- 1개월: {etf['1m_return']:+.1f}%
- 3개월: {etf['3m_return']:+.1f}%
- 6개월: {etf['6m_return']:+.1f}%
- 1년: {etf['1y_return']:+.1f}%
- 3년: {etf['3y_return']:+.1f}%

상품 설명: {etf['description']}
투자 테마: {themes_str}"""

        return text

    def create_documents(self) -> List[Document]:
        """ETF 데이터를 LangChain Document 리스트로 변환"""
        if not self.etf_data:
            self.load_data()

        documents = []
        for etf in self.etf_data:
            text = self.build_etf_text(etf)
            metadata = {
                "ticker": etf["ticker"],
                "name": etf["name"],
                "category": etf["category"],
                "region": etf["region"],
                "risk_level": etf["risk_level"],
                "expense_ratio": etf["expense_ratio"],
                "aum_billion": etf["aum_billion"],
                "1y_return": etf["1y_return"],
                "3y_return": etf["3y_return"],
                "theme": etf.get("theme", []),
            }
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        print(f"✅ {len(documents)}개 ETF Document 생성 완료")
        return documents

    def get_etf_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """티커로 ETF 정보 조회"""
        if not self.etf_data:
            self.load_data()
        for etf in self.etf_data:
            if etf["ticker"] == ticker:
                return etf
        return None

    def filter_by_risk(self, risk_levels: List[str]) -> List[Dict[str, Any]]:
        """위험등급으로 ETF 필터링"""
        if not self.etf_data:
            self.load_data()
        return [e for e in self.etf_data if e["risk_level"] in risk_levels]

    def filter_by_category(self, categories: List[str]) -> List[Dict[str, Any]]:
        """카테고리로 ETF 필터링"""
        if not self.etf_data:
            self.load_data()
        return [e for e in self.etf_data if e["category"] in categories]

    def get_top_performers(self, period: str = "1y", top_n: int = 5) -> List[Dict[str, Any]]:
        """수익률 상위 ETF 반환"""
        if not self.etf_data:
            self.load_data()
        key = f"{period}_return"
        sorted_etfs = sorted(self.etf_data, key=lambda x: x.get(key, 0), reverse=True)
        return sorted_etfs[:top_n]

    def get_statistics(self) -> Dict[str, Any]:
        """ETF 데이터 통계 정보"""
        if not self.etf_data:
            self.load_data()

        categories = {}
        risk_levels = {}
        regions = {}

        for etf in self.etf_data:
            cat = etf.get("category", "기타")
            categories[cat] = categories.get(cat, 0) + 1

            risk = etf.get("risk_level", "기타")
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

            region = etf.get("region", "기타")
            regions[region] = regions.get(region, 0) + 1

        return {
            "total_count": len(self.etf_data),
            "categories": categories,
            "risk_levels": risk_levels,
            "regions": regions,
            "avg_expense_ratio": sum(e["expense_ratio"] for e in self.etf_data) / len(self.etf_data),
            "total_aum_billion": sum(e["aum_billion"] for e in self.etf_data),
        }


if __name__ == "__main__":
    collector = ETFDataCollector("../data/sample_etf_data.json")
    collector.load_data()

    stats = collector.get_statistics()
    print("\n📊 ETF 데이터 통계:")
    print(f"  총 ETF 수: {stats['total_count']}")
    print(f"  카테고리별: {stats['categories']}")
    print(f"  위험등급별: {stats['risk_levels']}")
    print(f"  평균 운용보수: {stats['avg_expense_ratio']:.3f}%")

    print("\n🏆 1년 수익률 TOP 5:")
    for etf in collector.get_top_performers("1y", 5):
        print(f"  {etf['name']}: {etf['1y_return']:+.1f}%")

    docs = collector.create_documents()
    print(f"\n📄 생성된 Document 수: {len(docs)}")
    print(f"\n첫 번째 Document 내용:\n{docs[0].page_content[:300]}...")
