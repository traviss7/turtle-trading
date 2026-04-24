import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="터틀 트레이딩 대시보드", layout="wide", page_icon="🐢")

# ── 포트폴리오 (매수가·총자본 모두 원화 기준) ──────────────────────────
PORTFOLIO = [
    {"name": "SK하이닉스",          "ticker": "000660.KS", "avg_krw": 691_909,   "shares": 44},
    {"name": "삼성중공업",          "ticker": "010140.KS", "avg_krw": 32_500,    "shares": 308},
    {"name": "한화에어로스페이스",  "ticker": "012450.KS", "avg_krw": 1_408_000, "shares": 7},
    {"name": "삼성전기",            "ticker": "009150.KS", "avg_krw": 388_539,   "shares": 25},
    {"name": "두산에너빌리티",      "ticker": "034020.KS", "avg_krw": 123_100,   "shares": 82},
    {"name": "한화오션",            "ticker": "042660.KS", "avg_krw": 135_000,   "shares": 44},
    {"name": "웨스턴디지털(WDC)",   "ticker": "WDC",       "avg_krw": 754_195,   "shares": 2},
]

USD_TICKERS = {"WDC"}


@st.cache_data(ttl=300)
def fetch_all():
    """전체 종목 + 환율 데이터 (5분 캐시)"""
    # USD/KRW 환율
    try:
        fx_df = yf.download("USDKRW=X", period="5d", progress=False)
        usd_krw = float(fx_df["Close"].values.flatten()[-1])
    except Exception:
        usd_krw = 1_380

    data = {}
    for s in PORTFOLIO:
        try:
            df = yf.download(s["ticker"], period="4mo", progress=False)
            data[s["ticker"]] = df if not df.empty else None
        except Exception:
            data[s["ticker"]] = None

    return data, usd_krw


def calc_atr(df, period=20):
    c = df["Close"].values.flatten()
    h = df["High"].values.flatten()
    lo = df["Low"].values.flatten()
    trs = [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(df))]
    return float(np.mean(trs[-period:])) if len(trs) >= period else float(np.mean(trs))


def analyze(stock, df, usd_krw):
    if df is None or len(df) < 22:
        return None

    is_usd = stock["ticker"] in USD_TICKERS
    fx = usd_krw if is_usd else 1.0

    c  = df["Close"].values.flatten()
    h  = df["High"].values.flatten()
    lo = df["Low"].values.flatten()

    current = float(c[-1]) * fx
    n       = calc_atr(df) * fx
    low_20  = float(np.min(lo[-20:])) * fx          # System 2 청산선
    high_55 = float(np.max(h[-min(55, len(h)):])) * fx  # 참고용

    avg    = stock["avg_krw"]
    shares = stock["shares"]
    pnl    = (current - avg) * shares
    pnl_pct = (current - avg) / avg * 100

    return {
        "name":        stock["name"],
        "ticker":      stock["ticker"],
        "shares":      shares,
        "avg":         avg,
        "current":     current,
        "n":           n,
        "low_20":      low_20,
        "high_55":     high_55,
        "pnl":         pnl,
        "pnl_pct":     pnl_pct,
        "exit_signal": current < low_20,
        "stop_loss":   avg - 2 * n,
        "add1":        avg + 0.5 * n,
        "add2":        avg + 1.0 * n,
        "add3":        avg + 1.5 * n,
        "dist_exit_pct": (current - low_20) / current * 100,
        "value":       current * shares,
        "cost":        avg * shares,
        "is_usd":      is_usd,
    }


# ── 레이아웃 ──────────────────────────────────────────────────────────
st.title("🐢 터틀 트레이딩 대시보드")
st.caption(
    f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
    "System 2 적용 (55일 신고가 진입 / 20일 신저가 청산)"
)

# 사이드바
st.sidebar.header("⚙️ 설정")
capital  = st.sidebar.number_input("총 자본 (원)", value=100_000_000, step=5_000_000, format="%d")
risk_pct = st.sidebar.slider("단위 리스크 (%)", 0.5, 2.0, 1.0, 0.25)
if st.sidebar.button("🔄 새로고침", type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(
    "**System 2 규칙 요약**\n"
    "- 진입: 55일 신고가 돌파\n"
    "- 청산: 20일 신저가 이탈\n"
    "- 손절: 진입가 − 2N\n"
    "- 애드업: +0.5N / +1N / +1.5N"
)

# 데이터 로드
with st.spinner("시세 데이터 불러오는 중..."):
    all_data, usd_krw = fetch_all()

st.sidebar.metric("USD/KRW", f"{usd_krw:,.0f} 원")

# 분석
results = [r for s in PORTFOLIO if (r := analyze(s, all_data.get(s["ticker"]), usd_krw))]

if not results:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 새로고침 해주세요.")
    st.stop()

# ── 포트폴리오 요약 ───────────────────────────────────────────────────
total_value   = sum(r["value"] for r in results)
total_cost    = sum(r["cost"]  for r in results)
total_pnl     = total_value - total_cost
total_pnl_pct = total_pnl / total_cost * 100
exit_count    = sum(1 for r in results if r["exit_signal"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 평가액",   f"{total_value / 1e8:.2f} 억원", f"{total_pnl / 1e4:+,.0f} 만원")
c2.metric("총 수익률",   f"{total_pnl_pct:+.2f}%")
c3.metric("보유 종목",   f"{len(results)}개")
c4.metric(
    "🚨 청산 신호",
    f"{exit_count}개",
    delta="신호 있음" if exit_count else "없음",
    delta_color="inverse" if exit_count else "off",
)

st.divider()

# ── 신호 알림 ─────────────────────────────────────────────────────────
exits = [r for r in results if r["exit_signal"]]
if exits:
    for r in exits:
        st.error(
            f"🚨 **청산 신호 — {r['name']}** ({r['ticker']})  |  "
            f"현재가 {r['current']:,.0f}원  <  20일 저점 {r['low_20']:,.0f}원  "
            f"(차이 {r['dist_exit_pct']:.1f}%)"
        )
else:
    st.success("✅ 청산 신호 없음 — 모든 포지션 정상")

# 손절가 근접 경고 (stop_loss의 5% 이내)
near_stops = [
    r for r in results
    if not r["exit_signal"] and r["n"] > 0 and r["current"] < r["stop_loss"] * 1.05
]
for r in near_stops:
    st.warning(
        f"⚠️ 손절가 근접 — **{r['name']}**  |  "
        f"현재가 {r['current']:,.0f}원 / 손절가(-2N) {r['stop_loss']:,.0f}원"
    )

st.divider()

# ── 전체 포지션 테이블 ────────────────────────────────────────────────
st.subheader("📋 전체 포지션")

def signal_label(r):
    if r["exit_signal"]:
        return "🚨 청산"
    if r["n"] > 0 and r["current"] < r["stop_loss"] * 1.05:
        return "⚠️ 손절근접"
    if r["current"] >= r["add1"]:
        return "➕ 애드업가능"
    return "✅ 정상"

rows = []
for r in results:
    rows.append({
        "종목":          r["name"],
        "현재가(원)":    f"{r['current']:,.0f}",
        "매수가(원)":    f"{r['avg']:,.0f}",
        "수익률":        f"{r['pnl_pct']:+.1f}%",
        "평가손익(만원)": f"{r['pnl'] / 1e4:+.1f}",
        "N (ATR20)":     f"{r['n']:,.0f}",
        "20일저점":      f"{r['low_20']:,.0f}",
        "청산까지":      f"{r['dist_exit_pct']:+.1f}%",
        "손절가(-2N)":   f"{r['stop_loss']:,.0f}",
        "상태":          signal_label(r),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── 종목별 상세 ───────────────────────────────────────────────────────
st.subheader("🔍 종목별 상세")

for r in results:
    label = f"{r['name']} ({r['ticker']})  —  {r['pnl_pct']:+.1f}%  {signal_label(r)}"
    with st.expander(label, expanded=r["exit_signal"]):

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("현재가",       f"{r['current']:,.0f} 원",   f"{r['pnl_pct']:+.1f}%")
        m2.metric("평가손익",     f"{r['pnl'] / 1e4:+.1f} 만원")
        m3.metric("N (ATR 20일)", f"{r['n']:,.0f} 원")
        m4.metric("보유 수량",    f"{r['shares']} 주")

        st.write("")

        unit_size = int((capital * risk_pct / 100) / (2 * r["n"])) if r["n"] > 0 else 0

        detail = pd.DataFrame([
            {"구분": "➕ 애드업 3차  (+1.5N)", "가격(원)": f"{r['add3']:,.0f}",    "설명": "4번째 유닛 진입가"},
            {"구분": "➕ 애드업 2차  (+1.0N)", "가격(원)": f"{r['add2']:,.0f}",    "설명": "3번째 유닛 진입가"},
            {"구분": "➕ 애드업 1차  (+0.5N)", "가격(원)": f"{r['add1']:,.0f}",    "설명": "2번째 유닛 진입가"},
            {"구분": "📌 평균 매수가",          "가격(원)": f"{r['avg']:,.0f}",     "설명": "현재 기준 진입가"},
            {"구분": "🔴 20일 저점 (청산선)",  "가격(원)": f"{r['low_20']:,.0f}",  "설명": "System 2 청산 트리거"},
            {"구분": "🛑 손절가  (−2N)",       "가격(원)": f"{r['stop_loss']:,.0f}", "설명": "하드 손절선"},
        ])
        st.table(detail)

        if unit_size > 0:
            st.info(
                f"💰 현재 설정 기준 추가 매수 단위: **{unit_size}주** "
                f"(총자본 {capital / 1e8:.1f}억 × 리스크 {risk_pct}% ÷ 2N)"
            )
        if r["is_usd"]:
            st.caption(f"USD 종목 — 적용 환율: {usd_krw:,.0f} 원/USD")

        # 차트 (최근 60 거래일)
        df = all_data.get(r["ticker"])
        if df is not None and not df.empty:
            fx   = usd_krw if r["is_usd"] else 1.0
            days = min(60, len(df))
            chart = pd.DataFrame(
                {
                    "종가":          df["Close"].values.flatten()[-days:] * fx,
                    "20일저점(청산)": [r["low_20"]]    * days,
                    "손절가(-2N)":   [r["stop_loss"]] * days,
                    "매수가":        [r["avg"]]        * days,
                },
                index=df.index[-days:],
            )
            st.line_chart(chart)
