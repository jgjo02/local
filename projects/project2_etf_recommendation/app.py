"""Project 2: ETF 추천 RAG 시스템 - Gradio 웹 인터페이스"""

import os
import sys
import gradio as gr
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.config import Config
from src.etf_data_collector import ETFDataCollector
from src.etf_vector_store import ETFVectorStore
from src.etf_recommender import ETFRecommender

# ── 초기화 ──────────────────────────────────────────────────────────────────

config = Config()
collector = ETFDataCollector(config.etf_data_path)
vector_store = ETFVectorStore(config)

print("🚀 ETF 추천 시스템 초기화 중...")

# 벡터 스토어 로드 또는 구축
if not vector_store.load():
    print("📊 ETF 데이터 인덱싱 중...")
    collector.load_data()
    vector_store.build_from_collector(collector)
    vector_store.save()

recommender = ETFRecommender(config, vector_store)
print("✅ 초기화 완료!")

# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

RISK_OPTIONS = ["저위험", "중립", "고위험"]

EXAMPLE_QUESTIONS = [
    "노후 대비로 안전하게 투자하고 싶어요. 어떤 ETF가 좋을까요?",
    "AI와 반도체 테마에 집중 투자하고 싶습니다",
    "월 100만원씩 10년 장기투자할 ETF 포트폴리오 추천해주세요",
    "미국 주식 ETF 중 수수료가 저렴한 것을 알려주세요",
    "배당 수익을 원하는데 어떤 ETF가 적합할까요?",
    "2차전지와 전기차 관련 ETF를 비교해주세요",
]


def chat_with_bot(
    message: str,
    history: List[Tuple[str, str]],
    risk_preference: str,
    session_id: str,
) -> Tuple[List[Tuple[str, str]], str]:
    """챗봇 응답 생성"""
    if not message.strip():
        return history, ""

    if not config.openai_api_key:
        history.append((message, "⚠️ OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."))
        return history, ""

    risk = risk_preference if risk_preference != "선택 안함" else None

    history.append((message, ""))
    full_response = ""

    try:
        for token in recommender.stream_recommend(
            question=message,
            session_id=session_id,
            risk_preference=risk,
        ):
            full_response += token
            history[-1] = (message, full_response)
            yield history, ""
    except Exception as e:
        error_msg = f"⚠️ 오류가 발생했습니다: {str(e)}"
        history[-1] = (message, error_msg)
        yield history, ""


def get_portfolio_recommendation(
    investment_amount: int,
    risk_profile: str,
    investment_period: str,
    goals: List[str],
) -> str:
    """포트폴리오 추천"""
    if not config.openai_api_key:
        return "⚠️ OpenAI API 키가 설정되지 않았습니다."

    if not goals:
        return "⚠️ 투자 목표를 하나 이상 선택해주세요."

    try:
        result = recommender.get_portfolio_recommendation(
            investment_amount=investment_amount,
            risk_profile=risk_profile,
            investment_period=investment_period,
            goals=goals,
        )
        return result["portfolio_recommendation"]
    except Exception as e:
        return f"⚠️ 포트폴리오 추천 중 오류: {str(e)}"


def get_etf_stats() -> str:
    """ETF 데이터 통계"""
    try:
        collector.load_data()
        stats = collector.get_statistics()
        top5 = collector.get_top_performers("1y", 5)

        result = f"""## 📊 ETF 데이터 현황

**총 ETF 수**: {stats['total_count']}개
**총 운용규모**: {stats['total_aum_billion']:,}억원
**평균 운용보수**: {stats['avg_expense_ratio']:.3f}%

### 카테고리별 분포
"""
        for cat, count in stats["categories"].items():
            result += f"- {cat}: {count}개\n"

        result += "\n### 🏆 1년 수익률 TOP 5\n"
        for i, etf in enumerate(top5, 1):
            result += f"{i}. **{etf['name']}**: {etf['1y_return']:+.1f}%\n"

        return result
    except Exception as e:
        return f"⚠️ 통계 로드 오류: {str(e)}"


def reset_chat(session_id: str):
    """대화 초기화"""
    recommender.reset_session(session_id)
    return [], ""


# ── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="ETF 추천 AI 상담사",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .main-header { text-align: center; margin-bottom: 20px; }
    .disclaimer { font-size: 12px; color: #888; margin-top: 10px; }
    """,
) as demo:

    session_id = gr.State(value="user_session_001")

    gr.Markdown(
        """
        # 📈 ETF 추천 AI 상담사
        **하이브리드 RAG 기반** 개인화 ETF 추천 시스템입니다.
        투자 성향과 목표에 맞는 ETF를 추천해드립니다.
        """,
        elem_classes="main-header",
    )

    with gr.Tabs():

        # ── 탭 1: ETF 상담 챗봇 ──────────────────────────────────────────────
        with gr.TabItem("💬 ETF 상담 챗봇"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 설정")
                    risk_selector = gr.Radio(
                        choices=["선택 안함"] + RISK_OPTIONS,
                        value="선택 안함",
                        label="위험 선호도",
                        info="투자 성향에 맞게 선택하세요",
                    )
                    gr.Markdown("### 💡 추천 질문")
                    example_btns = []
                    for q in EXAMPLE_QUESTIONS:
                        btn = gr.Button(q, size="sm", variant="secondary")
                        example_btns.append(btn)

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        value=[],
                        label="ETF 상담 챗봇",
                        height=500,
                        show_label=True,
                        avatar_images=(None, "🤖"),
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="ETF 추천 또는 투자 관련 질문을 입력하세요...",
                            label="질문",
                            lines=2,
                            scale=4,
                        )
                        with gr.Column(scale=1):
                            send_btn = gr.Button("전송 ▶", variant="primary")
                            clear_btn = gr.Button("초기화 🔄", variant="secondary")

                    gr.Markdown(
                        "⚠️ 본 서비스는 교육 목적으로 제공되며, 실제 투자 결정에 활용 시 전문가 상담을 권장합니다.",
                        elem_classes="disclaimer",
                    )

            # 이벤트 핸들러
            send_btn.click(
                fn=chat_with_bot,
                inputs=[msg_input, chatbot, risk_selector, session_id],
                outputs=[chatbot, msg_input],
            )
            msg_input.submit(
                fn=chat_with_bot,
                inputs=[msg_input, chatbot, risk_selector, session_id],
                outputs=[chatbot, msg_input],
            )
            clear_btn.click(
                fn=reset_chat,
                inputs=[session_id],
                outputs=[chatbot, msg_input],
            )
            for btn, q in zip(example_btns, EXAMPLE_QUESTIONS):
                btn.click(fn=lambda x=q: x, outputs=msg_input)

        # ── 탭 2: 포트폴리오 추천 ────────────────────────────────────────────
        with gr.TabItem("📊 포트폴리오 추천"):
            gr.Markdown("### 맞춤형 ETF 포트폴리오 구성")
            with gr.Row():
                with gr.Column():
                    amount_slider = gr.Slider(
                        minimum=100,
                        maximum=10000,
                        value=1000,
                        step=100,
                        label="투자금액 (만원)",
                    )
                    risk_portfolio = gr.Radio(
                        choices=RISK_OPTIONS,
                        value="중립",
                        label="위험 성향",
                    )
                    period_selector = gr.Radio(
                        choices=["1년 이하", "1-3년", "3-5년", "5년 이상"],
                        value="3-5년",
                        label="투자 기간",
                    )
                    goals_checkboxes = gr.CheckboxGroup(
                        choices=["노후 준비", "자산 성장", "배당 수익", "인플레이션 헤지", "단기 수익"],
                        value=["자산 성장"],
                        label="투자 목표 (복수 선택 가능)",
                    )
                    portfolio_btn = gr.Button("포트폴리오 추천받기", variant="primary", size="lg")

                with gr.Column():
                    portfolio_output = gr.Markdown(label="추천 포트폴리오")

            portfolio_btn.click(
                fn=get_portfolio_recommendation,
                inputs=[amount_slider, risk_portfolio, period_selector, goals_checkboxes],
                outputs=portfolio_output,
            )

        # ── 탭 3: ETF 데이터 현황 ─────────────────────────────────────────────
        with gr.TabItem("📋 ETF 데이터"):
            gr.Markdown("### 현재 추천 가능한 ETF 현황")
            stats_btn = gr.Button("통계 불러오기", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(fn=get_etf_stats, outputs=stats_output)

    gr.Markdown(
        """
        ---
        **12주 AI 서비스 개발 과정 - Project 2: ETF 추천 RAG 시스템**
        LangChain + FAISS + BM25 하이브리드 검색 기반 ETF 추천
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_port=config.server_port,
        share=config.share,
        show_error=True,
    )
