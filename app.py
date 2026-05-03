import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="터틀 트레이딩 대시보드", layout="wide", page_icon="🐢")

KST = timezone(timedelta(hours=9))

# ── 포트폴리오 (units: 현재 보유 유닛 수) ────────────────────────────────
PORTFOLIO = [
    {"name": "SK하이닉스",          "ticker": "000660.KS", "avg_krw": 691_909,   "shares": 44,  "units": 1},
    {"name": "삼성중공업",          "ticker": "010140.KS", "avg_krw": 32_500,    "shares": 308, "units": 1},
    {"name": "한화에어로스페이스",  "ticker": "012450.KS", "avg_krw": 1_408_000, "shares": 7,   "units": 1},
    {"name": "삼성전기",            "ticker": "009150.KS", "avg_krw": 388_539,   "shares": 25,  "units": 1},
    {"name": "두산에너빌리티",      "ticker": "034020.KS", "avg_krw": 123_100,   "shares": 82,  "units": 1},
    {"name": "한화오션",            "ticker": "042660.KS", "avg_krw": 135_000,   "shares": 44,  "units": 1},
    {"name": "샌디스크(SNDK)",      "ticker": "SNDK",      "avg_krw": 754_195,   "shares": 2,   "units": 1},
]

USD_TICKERS        = {"SNDK"}
ATR_PERIOD         = 20
MAX_UNITS_PER_MKT  = 4
MAX_UNITS_TOTAL    = 12


# ── 텔레그램 ────────────────────────────────────────────────────────────

def _tg_send(token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=3600)
def _cached_alert(slot_key: str, token: str, chat_id: str, message: str) -> bool:
    """동일 slot_key는 TTL 내 1회만 전송 (전역 캐시)"""
    return _tg_send(token, chat_id, message)


def build_alert_message(results: list, title: str) -> str:
    now = datetime.now(KST)
    total_units = sum(r["units"] for r in results if not r.get("no_data"))
    lines = [f"{title}  {now.strftime('%Y-%m-%d %H:%M')} KST\n"]

    exits = [r for r in results if not r.get("no_data") and r["exit_signal"]]
    adds  = [
        r for r in results
        if not r.get("no_data") and not r["exit_signal"]
        and r["next_add"] is not None and r["current"] >= r["next_add"]
        and r["units"] < MAX_UNITS_PER_MKT and total_units < MAX_UNITS_TOTAL
    ]
    stops = [
        r for r in results
        if not r.get("no_data") and not r["exit_signal"]
        and r.get("n", 0) > 0 and r["current"] < r["stop_loss"] * 1.05
    ]

    if exits:
        lines.append("🚨 *청산 신호*")
        for r in exits:
            lines.append(f"  • {r['name']}: {r['current']:,.0f}원 < 20일저점 {r['low_20']:,.0f}원")
    if adds:
        lines.append("\n➕ *애드업 신호*")
        for r in adds:
            lines.append(
                f"  • {r['name']} (유닛 {r['units']}→{r['units']+1}): "
                f"{r['current']:,.0f}원 ≥ 목표 {r['next_add']:,.0f}원"
            )
    if stops:
        lines.append("\n⚠️ *손절가 근접*")
        for r in stops:
            lines.append(f"  • {r['name']}: {r['current']:,.0f}원  손절 {r['stop_loss']:,.0f}원")
    if not exits and not adds and not stops:
        lines.append("✅ 특이사항 없음")

    if total_units > 0:
        lines.append(f"\n📊 총 유닛: {total_units}/{MAX_UNITS_TOTAL}")
    return "\n".join(lines)


def maybe_send_daily_alerts(results: list, token: str, chat_id: str):
    if not token or not chat_id:
        return
    now = datetime.now(KST)
    h, m = now.hour, now.minute

    # 한국 장 시작 09:00 / 장 마감 15:20 — 한국 종목만
    in_kr_open  = (h == 9 and m < 10)
    in_kr_close = (h == 15 and 20 <= m < 30)
    # 미국 장 마감 종가 알림: 여름(EDT) 05:00, 겨울(EST) 06:00 — 미국 종목만
    in_us_close = (h == 5 and m < 10) or (h == 6 and m < 10)

    date_str = now.strftime("%Y-%m-%d")

    if in_kr_open or in_kr_close:
        kr_results = [r for r in results if not r.get("is_usd")]
        slot = f"{date_str}_{'kr_am' if in_kr_open else 'kr_pm'}"
        _cached_alert(slot, token, chat_id,
                      build_alert_message(kr_results, "🇰🇷 *한국주식 터틀 알람*"))

    if in_us_close:
        us_results = [r for r in results if r.get("is_usd")]
        slot = f"{date_str}_us"
        _cached_alert(slot, token, chat_id,
                      build_alert_message(us_results, "🇺🇸 *미국주식 종가 알람*"))


# ── 데이터 ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_all():
    try:
        fx_df   = yf.download("USDKRW=X", period="5d", progress=False)
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


def calc_atr(df, period=ATR_PERIOD):
    c  = df["Close"].values.flatten()
    h  = df["High"].values.flatten()
    lo = df["Low"].values.flatten()
    trs = [max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1])) for i in range(1, len(df))]
    return float(np.mean(trs[-period:])) if len(trs) >= period else float(np.mean(trs))


def analyze(stock, df, usd_krw):
    units = stock.get("units", 1)
    base = {
        "name": stock["name"], "ticker": stock["ticker"],
        "shares": stock["shares"], "avg": stock["avg_krw"], "units": units,
        "is_usd": stock["ticker"] in USD_TICKERS,
    }
    if df is None or len(df) < 5:
        return {
            **base,
            "current": None, "n": None, "low_20": None, "high_55": None,
            "pnl": None, "pnl_pct": None, "exit_signal": False,
            "stop_loss": None, "add1": None, "add2": None, "add3": None,
            "next_add": None, "dist_exit_pct": None,
            "value": 0, "cost": stock["avg_krw"] * stock["shares"], "no_data": True,
        }

    fx      = usd_krw if base["is_usd"] else 1.0
    c       = df["Close"].values.flatten()
    h       = df["High"].values.flatten()
    lo      = df["Low"].values.flatten()
    current = float(c[-1]) * fx
    n       = calc_atr(df) * fx
    low_20  = float(np.min(lo[-20:])) * fx
    high_55 = float(np.max(h[-min(55, len(h)):])) * fx
    avg     = stock["avg_krw"]
    shares  = stock["shares"]
    add1    = avg + 0.5 * n
    add2    = avg + 1.0 * n
    add3    = avg + 1.5 * n

    # 유닛별 다음 애드업 목표가 (units=1→add1, 2→add2, 3→add3, 4→없음)
    next_add = {1: add1, 2: add2, 3: add3}.get(units) if units < MAX_UNITS_PER_MKT else None

    return {
        **base,
        "current":     current,
        "n":           n,
        "low_20":      low_20,
        "high_55":     high_55,
        "pnl":         (current - avg) * shares,
        "pnl_pct":     (current - avg) / avg * 100,
        "exit_signal": current < low_20,
        "stop_loss":   avg - 2 * n,
        "add1":        add1,
        "add2":        add2,
        "add3":        add3,
        "next_add":    next_add,
        "dist_exit_pct": (current - low_20) / current * 100,
        "value":       current * shares,
        "cost":        avg * shares,
        "no_data":     False,
    }


def signal_label(r):
    if r.get("no_data"):
        return "⚫ 데이터없음"
    if r["exit_signal"]:
        return "🚨 청산"
    if r["n"] > 0 and r["current"] < r["stop_loss"] * 1.05:
        return "⚠️ 손절근접"
    if r["next_add"] is not None and r["current"] >= r["next_add"] and r["units"] < MAX_UNITS_PER_MKT:
        return "➕ 애드업가능"
    return "✅ 정상"


# ── 레이아웃 ────────────────────────────────────────────────────────────
st.title("🐢 터틀 트레이딩 대시보드")
st.caption(
    f"업데이트: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST  ·  "
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
st.sidebar.page_link("pages/1_백테스트.py", label="📈 백테스트로 전략 검증", icon="📈")

st.sidebar.divider()
st.sidebar.subheader("📡 텔레그램 알림")
try:
    _def_token   = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    _def_chat_id = st.secrets.get("TELEGRAM_CHAT_ID",   "")
except Exception:
    _def_token, _def_chat_id = "", ""

tg_token   = st.sidebar.text_input("Bot Token",  value=_def_token,   type="password", placeholder="123456:ABC...")
tg_chat_id = st.sidebar.text_input("Chat ID",    value=_def_chat_id, placeholder="-100xxxxx")
st.sidebar.caption("자동 알림: 장시작 09:00 / 장마감 15:20 (KST)")

# ── 데이터 로드 ─────────────────────────────────────────────────────────
with st.spinner("시세 데이터 불러오는 중..."):
    all_data, usd_krw = fetch_all()

st.sidebar.metric("USD/KRW", f"{usd_krw:,.0f} 원")

results     = [analyze(s, all_data.get(s["ticker"]), usd_krw) for s in PORTFOLIO]
total_units = sum(r["units"] for r in results if not r.get("no_data"))

if not results:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# 자동 텔레그램 알림 (하루 2회 시간대)
maybe_send_daily_alerts(results, tg_token, tg_chat_id)

# ── 포트폴리오 요약 ──────────────────────────────────────────────────────
total_value   = sum(r["value"] for r in results)
total_cost    = sum(r["cost"]  for r in results)
total_pnl     = total_value - total_cost
total_pnl_pct = total_pnl / total_cost * 100
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

# ── 유닛 한도 현황 ──────────────────────────────────────────────────────
unit_over = total_units > MAX_UNITS_TOTAL
with st.expander("📊 유닛 한도 현황", expanded=unit_over or total_units >= MAX_UNITS_TOTAL * 0.8):
    cols = st.columns(len(results))
    for i, r in enumerate(results):
        u     = r["units"]
        icon  = "🔴" if u >= MAX_UNITS_PER_MKT else "🟡" if u >= 3 else "🟢"
        with cols[i]:
            st.metric(r["name"], f"{u}/{MAX_UNITS_PER_MKT}", delta=f"{icon}")
    if unit_over:
        st.error(f"🚨 총 유닛 한도 초과: {total_units}/{MAX_UNITS_TOTAL} — 신규 진입 불가")
    elif total_units >= MAX_UNITS_TOTAL * 0.8:
        st.warning(f"⚠️ 총 유닛 한도 근접: {total_units}/{MAX_UNITS_TOTAL}")
    else:
        st.success(f"✅ 유닛 여유 있음: {total_units}/{MAX_UNITS_TOTAL}")

st.divider()

# ── 신호 알림 ───────────────────────────────────────────────────────────
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

# 손절가 근접 경고
near_stops = [
    r for r in results
    if not r.get("no_data") and not r["exit_signal"]
    and r["n"] > 0 and r["current"] < r["stop_loss"] * 1.05
]
for r in near_stops:
    st.warning(
        f"⚠️ 손절가 근접 — **{r['name']}**  |  "
        f"현재가 {r['current']:,.0f}원 / 손절가(-2N) {r['stop_loss']:,.0f}원"
    )

# 애드업 신호
add_signals = [
    r for r in results
    if not r.get("no_data") and not r["exit_signal"]
    and r["next_add"] is not None and r["current"] >= r["next_add"]
    and r["units"] < MAX_UNITS_PER_MKT and not unit_over
]
for r in add_signals:
    st.info(
        f"➕ **애드업 신호 — {r['name']}**  |  "
        f"현재가 {r['current']:,.0f}원 ≥ {r['units']+1}차 목표 {r['next_add']:,.0f}원  "
        f"(유닛 {r['units']} → {r['units']+1})"
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
        "평가손익(만원)": "-" if nd else f"{r['pnl'] / 1e4:+.1f}",
        "N (ATR20)":      "-" if nd else f"{r['n']:,.0f}",
        "다음 애드업":    "-" if nd else ("최대유닛" if r["next_add"] is None else f"{r['next_add']:,.0f}"),
        "20일저점":       "-" if nd else f"{r['low_20']:,.0f}",
        "청산까지":       "-" if nd else f"{r['dist_exit_pct']:+.1f}%",
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
            st.write(f"매수가: {r['avg']:,.0f}원  |  보유수량: {r['shares']}주")
            continue

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("현재가",       f"{r['current']:,.0f} 원", f"{r['pnl_pct']:+.1f}%")
        m2.metric("평가손익",     f"{r['pnl'] / 1e4:+.1f} 만원")
        m3.metric("N (ATR 20일)", f"{r['n']:,.0f} 원")
        m4.metric("보유 수량",    f"{r['shares']} 주")
        m5.metric("보유 유닛",    f"{r['units']} / {MAX_UNITS_PER_MKT}")

        st.write("")

        unit_size = int((capital * risk_pct / 100) / (2 * r["n"])) if r["n"] > 0 else 0

        detail = pd.DataFrame([
            {"구분": "➕ 애드업 3차 (+1.5N)", "가격(원)": f"{r['add3']:,.0f}", "설명": "4번째 유닛 진입가"},
            {"구분": "➕ 애드업 2차 (+1.0N)", "가격(원)": f"{r['add2']:,.0f}", "설명": "3번째 유닛 진입가"},
            {"구분": "➕ 애드업 1차 (+0.5N)", "가격(원)": f"{r['add1']:,.0f}", "설명": "2번째 유닛 진입가"},
            {"구분": "📌 평균 매수가",         "가격(원)": f"{r['avg']:,.0f}",  "설명": "현재 기준 진입가"},
            {"구분": "🔴 20일 저점 (청산선)",  "가격(원)": f"{r['low_20']:,.0f}", "설명": "System 2 청산 트리거"},
            {"구분": "🛑 손절가 (−2N)",        "가격(원)": f"{r['stop_loss']:,.0f}", "설명": "하드 손절선"},
        ])
        st.table(detail)

        # 애드업 조건 상태
        st.markdown("**애드업 조건 현황**")
        for idx, (price, add_lbl) in enumerate(
            [(r["add1"], "1차 +0.5N"), (r["add2"], "2차+1.0N"), (r["add3"], "3차 +1.5N")], 1
        ):
            reached   = r["current"] >= price
            can_enter = r["units"] == idx and reached and total_units < MAX_UNITS_TOTAL
            icon      = "✅" if reached else "⬜"
            note      = "  → **진입 가능!**" if can_enter else ""
            st.write(f"{icon} 애드업 {add_lbl}: {price:,.0f}원{note}")

        if unit_size > 0:
            st.info(
                f"💰 추가 매수 단위: **{unit_size}주** "
                f"(총자본 {capital/1e8:.1f}억 × 리스크 {risk_pct}% ÷ 2N)"
            )
        if r["is_usd"]:
            st.caption(f"USD 종목 — 적용 환율: {usd_krw:,.0f} 원/USD")

        df_chart = all_data.get(r["ticker"])
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

# ── 텔레그램 수동 테스트 ────────────────────────────────────────────────
st.divider()
with st.expander("📡 텔레그램 알림 테스트"):
    st.caption(
        "자동 발송 시간: **장 시작 09:00~09:10** / **장 마감 15:20~15:30** (KST)  \n"
        "청산·애드업·손절근접 신호가 있을 때만 내용에 포함됩니다."
    )
    if st.button("📨 지금 테스트 알림 보내기"):
        if tg_token and tg_chat_id:
            msg = build_alert_message(results, "🐢 *터틀 트레이딩 테스트 알람*")
            ok  = _tg_send(tg_token, tg_chat_id, msg)
            st.success("✅ 전송 성공") if ok else st.error("❌ 전송 실패 — Token/Chat ID 확인")
        else:
            st.warning("사이드바에서 Bot Token과 Chat ID를 입력해주세요.")
