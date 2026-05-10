import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

st.set_page_config(page_title="터틀 트레이딩 대시보드", layout="wide", page_icon="🐢")

KST = timezone(timedelta(hours=9))

# ── 설정 ────────────────────────────────────────────────────────────────
USD_TICKERS        = {"SNDK"}
ATR_PERIOD         = 20
MAX_UNITS_PER_MKT  = 4
MAX_UNITS_TOTAL    = 12

@st.cache_data(ttl=60)
def load_state():
    """저장된 상태 파일 읽기 (GitHub Actions에서 작성)"""
    state_file = Path("state.json")
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return data
        except Exception as e:
            st.error(f"상태 파일 읽기 오류: {e}")
            return None
    return None

@st.cache_data(ttl=300)
def fetch_chart_data(ticker):
    """차트용 데이터만 다운로드 (빠른 로딩)"""
    try:
        df = yf.download(ticker, period="4mo", progress=False)
        return df if not df.empty else None
    except Exception:
        return None

def signal_label(r):
    """신호 라벨"""
    if r.get("no_data"):
        return "⚫ 데이터없음"
    if r["exit_signal"]:
        return "🚨 청산"
    if r.get("n", 0) > 0 and r["current"] < r["stop_loss"] * 1.05:
        return "⚠️ 손절근접"
    if r["next_add"] is not None and r["current"] >= r["next_add"] and r["units"] < MAX_UNITS_PER_MKT:
        return "➕ 애드업가능"
    return "✅ 정상"

# ── 레이아웃 ────────────────────────────────────────────────────────────
st.title("🐢 터틀 트레이딩 대시보드")

# 상태 로드
state = load_state()

if state is None:
    st.warning("⚠️ 아직 데이터가 없습니다. GitHub Actions가 처음 실행될 때까지 기다려주세요.")
    st.info(
        "**자동 실행 시간:**\n"
        "- 매일 09:00~09:10 (한국 장 시작)\n"
        "- 매일 15:20~15:30 (한국 장 마감)"
    )
    st.stop()

update_time = state.get("timestamp", "")
usd_krw     = state.get("usd_krw", 1_380)
results     = state.get("results", [])

st.caption(
    f"마지막 업데이트: {update_time}  ·  "
    "System 2 (55일 신고가 진입 / 20일 신저가 청산)"
)

# ── 사이드바 ────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")
capital  = st.sidebar.number_input("총 자본 (원)", value=100_000_000, step=5_000_000, format="%d")
risk_pct = st.sidebar.slider("단위 리스크 (%)", 0.5, 2.0, 2.0, 0.25)
if st.sidebar.button("🔄 새로고침", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "**System 2 규칙 요약**\n"
    "- 진입: 55일 신고가 돌파\n"
    "- 청산: 20일 신저가 이탈\n"
    "- 손절: 진입가 − 2N  (ATR 20일)\n"
    "- 애드업: +0.5N / +1N / +1.5N\n"
    f"- 유닛 한도: 시장당 {MAX_UNITS_PER_MKT}, 총 {MAX_UNITS_TOTAL}"
)

st.sidebar.divider()
st.sidebar.subheader("📡 자동 실행")
st.sidebar.info(
    "**GitHub Actions 자동화**\n\n"
    "매일 **09:00~09:10** / **15:20~15:30** (KST)\n\n"
    "신호 발생 시:\n"
    "1️⃣ state.json 갱신\n"
    "2️⃣ Telegram 알림 전송\n"
    "3️⃣ 대시보드 자동 새로고침"
)

st.sidebar.metric("USD/KRW", f"{usd_krw:,.0f} 원")

# ── 포트폴리오 요약 ──────────────────────────────────────────────────────
total_value   = sum(r.get("value", 0) for r in results if not r.get("no_data"))
total_cost    = sum(r.get("cost", 0) for r in results if not r.get("no_data"))
total_pnl     = total_value - total_cost
total_pnl_pct = total_pnl / total_cost * 100 if total_cost > 0 else 0
total_units   = sum(r["units"] for r in results if not r.get("no_data"))
exit_count    = sum(1 for r in results if r["exit_signal"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 평가액",    f"{total_value / 1e8:.2f} 억원", f"{total_pnl / 1e4:+,.0f} 만원")
c2.metric("총 수익률",    f"{total_pnl_pct:+.2f}%")
c3.metric("보유 종목",    f"{len(results)}개")
c4.metric(
    "🚨 청산 신호",
    f"{exit_count}개",
    delta="신호 있음" if exit_count else "없음",
    delta_color="inverse" if exit_count else "off",
)
c5.metric(
    "📊 총 유닛",
    f"{total_units} / {MAX_UNITS_TOTAL}",
    delta="한도초과!" if total_units > MAX_UNITS_TOTAL else None,
    delta_color="inverse" if total_units > MAX_UNITS_TOTAL else "off",
)

st.divider()

# ── 신호 알림 ───────────────────────────────────────────────────────────
exits = [r for r in results if r["exit_signal"]]
if exits:
    for r in exits:
        st.error(
            f"🚨 **청산 신호 — {r['name']}** ({r['ticker']})  |  "
            f"현재가 {r['current']:,.0f}원  <  20일 저점 {r['low_20']:,.0f}원"
        )
else:
    st.success("✅ 청산 신호 없음")

near_stops = [
    r for r in results
    if not r.get("no_data") and not r["exit_signal"]
    and r.get("n", 0) > 0 and r["current"] < r["stop_loss"] * 1.05
]
for r in near_stops:
    st.warning(
        f"⚠️ 손절가 근접 — **{r['name']}**  |  "
        f"현재가 {r['current']:,.0f}원 / 손절가 {r['stop_loss']:,.0f}원"
    )

add_signals = [
    r for r in results
    if not r.get("no_data") and not r["exit_signal"]
    and r["next_add"] is not None and r["current"] >= r["next_add"]
    and r["units"] < MAX_UNITS_PER_MKT
]
for r in add_signals:
    st.info(
        f"➕ **애드업 신호 — {r['name']}**  |  "
        f"현재가 {r['current']:,.0f}원 ≥ 목표 {r['next_add']:,.0f}원"
    )

st.divider()

# ── 전체 포지션 테이블 ──────────────────────────────────────────────────
st.subheader("📋 전체 포지션")

rows = []
for r in results:
    nd = r.get("no_data")
    rows.append({
        "종목":           r["name"],
        "유닛":           "-" if nd else f"{r['units']}/{MAX_UNITS_PER_MKT}",
        "현재가(원)":     "데이터없음" if nd else f"{r['current']:,.0f}",
        "매수가(원)":     f"{r['avg']:,.0f}",
        "수익률":         "-" if nd else f"{r['pnl_pct']:+.1f}%",
        "N (ATR20)":      "-" if nd else f"{r['n']:,.0f}",
        "다음 애드업":    "-" if nd else ("최대" if r["next_add"] is None else f"{r['next_add']:,.0f}"),
        "20일저점":       "-" if nd else f"{r['low_20']:,.0f}",
        "손절가(-2N)":    "-" if nd else f"{r['stop_loss']:,.0f}",
        "상태":           signal_label(r),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── 종목별 상세 ─────────────────────────────────────────────────────────
st.subheader("🔍 종목별 상세")

for r in results:
    nd      = r.get("no_data")
    pnl_str = "데이터없음" if nd else f"{r['pnl_pct']:+.1f}%"
    lbl     = f"{r['name']} ({r['ticker']})  —  {pnl_str}  {signal_label(r)}"

    with st.expander(lbl, expanded=r["exit_signal"]):
        if nd:
            st.warning(f"시세 데이터를 가져올 수 없습니다. ({r['ticker']})")
            continue

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("현재가",       f"{r['current']:,.0f} 원", f"{r['pnl_pct']:+.1f}%")
        m2.metric("N (ATR 20일)", f"{r['n']:,.0f} 원")
        m3.metric("보유 유닛",    f"{r['units']} / {MAX_UNITS_PER_MKT}")
        m4.metric("애드업 목표",  "-" if r["next_add"] is None else f"{r['next_add']:,.0f} 원")
        m5.metric("손절가",       f"{r['stop_loss']:,.0f} 원")

        st.write("")

        # 애드업 가격 테이블
        detail = pd.DataFrame([
            {"구분": "➕ 애드업 3차 (+1.5N)", "가격(원)": f"{r['add3']:,.0f}"},
            {"구분": "➕ 애드업 2차 (+1.0N)", "가격(원)": f"{r['add2']:,.0f}"},
            {"구분": "➕ 애드업 1차 (+0.5N)", "가격(원)": f"{r['add1']:,.0f}"},
            {"구분": "📌 매수가",             "가격(원)": f"{r['avg']:,.0f}"},
            {"구분": "🔴 20일 저점 (청산)",   "가격(원)": f"{r['low_20']:,.0f}"},
            {"구분": "🛑 손절가 (−2N)",      "가격(원)": f"{r['stop_loss']:,.0f}"},
        ])
        st.table(detail)

        # 차트
        df_chart = fetch_chart_data(r["ticker"])
        if df_chart is not None and not df_chart.empty:
            fx   = usd_krw if r["is_usd"] else 1.0
            days = min(60, len(df_chart))
            chart_data = pd.DataFrame(
                {
                    "종가":           df_chart["Close"].values.flatten()[-days:] * fx,
                    "20일저점(청산)": [r["low_20"]]    * days,
                    "손절가(-2N)":    [r["stop_loss"]] * days,
                    "매수가":         [r["avg"]]        * days,
                    "다음애드업":     [r["next_add"] if r["next_add"] else r["add3"]] * days,
                },
                index=df_chart.index[-days:],
            )
            st.line_chart(chart_data)

st.divider()
st.caption(
    "🤖 이 대시보드는 GitHub Actions에서 **매일 09:00, 15:20**에 자동으로 갱신됩니다. "
    "신호 발생 시 텔레그램으로도 알림을 받습니다."
)
