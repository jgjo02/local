"""ETF 추천 품질 평가 모듈 - LLM-as-Judge"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_core.messages import HumanMessage

from .config import Config


RELEVANCE_EVAL_PROMPT = """당신은 ETF 추천 품질을 평가하는 전문가입니다.

## 평가 기준
사용자 질문과 추천된 ETF가 얼마나 관련성이 높은지 평가합니다.

## 사용자 질문
{question}

## 추천된 ETF 정보
{etf_info}

## 평가
1점(전혀 관련없음)부터 5점(매우 관련있음)까지 점수를 부여하고 이유를 설명해주세요.

응답 형식:
점수: [1-5]
이유: [간단한 설명]"""

ANSWER_QUALITY_PROMPT = """당신은 ETF 투자 상담 품질을 평가하는 전문가입니다.

## 평가 기준
- 정확성: 제공된 ETF 정보와 일치하는지
- 완전성: 중요한 정보가 포함되어 있는지
- 명확성: 이해하기 쉽게 설명했는지
- 적절성: 사용자 상황에 맞게 추천했는지

## 사용자 질문
{question}

## AI 답변
{answer}

## 참고한 ETF 정보
{context}

## 평가
각 기준별로 1-5점을 부여하고 종합 평가를 해주세요.

응답 형식:
정확성: [1-5] - [이유]
완전성: [1-5] - [이유]
명확성: [1-5] - [이유]
적절성: [1-5] - [이유]
종합점수: [평균]
개선 제안: [선택사항]"""


class ETFEvaluator:
    """ETF 추천 시스템 품질 평가"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=0,
            openai_api_key=config.openai_api_key,
        )

    def evaluate_relevance(
        self, question: str, docs: List[Document]
    ) -> Dict[str, Any]:
        """검색된 ETF의 관련성 평가"""
        results = []
        for doc in docs:
            etf_name = doc.metadata.get("name", "Unknown")
            prompt = RELEVANCE_EVAL_PROMPT.format(
                question=question,
                etf_info=doc.page_content[:500],
            )

            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content

            score = 3
            for line in content.split("\n"):
                if line.startswith("점수:"):
                    try:
                        score = int(line.split(":")[1].strip()[0])
                    except (ValueError, IndexError):
                        pass

            results.append({
                "etf_name": etf_name,
                "ticker": doc.metadata.get("ticker", ""),
                "relevance_score": score,
                "evaluation": content,
            })

        avg_score = sum(r["relevance_score"] for r in results) / len(results) if results else 0
        return {
            "individual_scores": results,
            "average_relevance": round(avg_score, 2),
            "top_relevant": sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:3],
        }

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        source_docs: List[Document],
    ) -> Dict[str, Any]:
        """생성된 답변의 품질 평가"""
        context = "\n\n".join([doc.page_content[:300] for doc in source_docs[:3]])

        prompt = ANSWER_QUALITY_PROMPT.format(
            question=question,
            answer=answer,
            context=context,
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content

        scores = {}
        criteria = ["정확성", "완전성", "명확성", "적절성", "종합점수"]
        for criterion in criteria:
            for line in content.split("\n"):
                if line.startswith(f"{criterion}:"):
                    try:
                        score_part = line.split(":")[1].strip()
                        scores[criterion] = float(score_part.split()[0].replace("[", "").replace("]", ""))
                    except (ValueError, IndexError):
                        scores[criterion] = 3.0

        return {
            "scores": scores,
            "overall_score": scores.get("종합점수", 3.0),
            "full_evaluation": content,
        }

    def run_batch_evaluation(
        self, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """배치 평가 실행"""
        results = []
        total_relevance = 0
        total_quality = 0

        for i, case in enumerate(test_cases, 1):
            print(f"  평가 중 {i}/{len(test_cases)}: {case['question'][:30]}...")

            relevance = self.evaluate_relevance(
                case["question"], case.get("source_docs", [])
            )
            quality = self.evaluate_answer_quality(
                case["question"],
                case.get("answer", ""),
                case.get("source_docs", []),
            )

            total_relevance += relevance["average_relevance"]
            total_quality += quality["overall_score"]

            results.append({
                "question": case["question"],
                "relevance_eval": relevance,
                "quality_eval": quality,
            })

        n = len(test_cases)
        return {
            "individual_results": results,
            "summary": {
                "avg_relevance": round(total_relevance / n, 2) if n > 0 else 0,
                "avg_quality": round(total_quality / n, 2) if n > 0 else 0,
                "total_cases": n,
            },
        }

    def generate_evaluation_report(self, eval_results: Dict[str, Any]) -> str:
        """평가 결과 보고서 생성"""
        summary = eval_results.get("summary", {})
        report = f"""
# ETF 추천 시스템 평가 보고서

## 종합 성능 지표
- 평균 관련성 점수: {summary.get('avg_relevance', 0):.2f} / 5.0
- 평균 답변 품질: {summary.get('avg_quality', 0):.2f} / 5.0
- 평가 케이스 수: {summary.get('total_cases', 0)}개

## 세부 평가 결과
"""
        for i, result in enumerate(eval_results.get("individual_results", []), 1):
            report += f"""
### 케이스 {i}
- 질문: {result['question']}
- 관련성: {result['relevance_eval']['average_relevance']:.2f}/5.0
- 품질: {result['quality_eval']['overall_score']:.2f}/5.0
"""
        return report
