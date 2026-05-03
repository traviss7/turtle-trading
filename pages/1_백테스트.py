import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

st.set_page_config(page_title="터틀 백테스트", layout="wide", page_icon="📈")

KST = timezone(timedelta(hours=9))

PORTFOLIO_TICKERS = [
    ("SK하이닉스",         "000660.KS"),
    ("삼성중공업",         "010140.KS"),
    ("한화에어로스페이스", "012450.KS"),
    ("삼성전기",           "009150.KS"),
    ("두산에너빌리티",     "034020.KS"),
    ("한화오션",           "042660.KS"),
    ("샌디스크(SNDK)",    "SNDK"),
    ("SPY (S&P500)",       "SPY"),
    ("QQQ (나스닥100)",    "QQQ"),
    ("GLD (금 ETF)",       "GLD"),
    ("직접 입력",          "__custom__"),
]


# ── 백테스트 엔진 ────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance MultiIndex 컬럼 처리"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def backtest_turtle(
    df: pd.DataFrame,
    capital: float,
    risk_pct: float,
    entry_period: int = 55,
    exit_period: int = 20,
    atr_period: int = 20,
    max_units: int = 4,
) -> dict:
    """
    Turtle System 2 백테스트.
    - 진입: entry_period일 신고가 돌파
    - 청산: exit_period일 신저가 이탈 또는 2N 손절
    - 애드업: +0.5N 간격으로 최대 max_units 유닛
    """
    df = _flatten(df).copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    dates = df.index

    # ATR (True Range의 rolling 평균)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    # 진입/청산 레벨 (look-ahead 방지: shift(1))
    high_entry = high.rolling(entry_period).max().shift(1)
    low_exit   = low.rolling(exit_period).min().shift(1)

    closes    = close.values
    atrs      = atr.values
    highs_ent = high_entry.values
    lows_ext  = low_exit.values

    in_trade          = False
    units             = []        # [{"price": float, "stop": float, "shares": int}]
    entry_date        = None
    first_entry_price = None

    trades     = []
    equity_pts = []
    cash       = float(capital)

    start_idx = max(entry_period, atr_period) + 1

    for i in range(start_idx, len(closes)):
        price = float(closes[i])
        N     = float(atrs[i])
        h_ent = float(highs_ent[i])
        l_ext = float(lows_ext[i])

        if np.isnan(N) or np.isnan(h_ent) or np.isnan(l_ext) or N <= 0:
            pos_val = sum(price * u["shares"] for u in units)
            equity_pts.append({"date": dates[i], "equity": cash + pos_val})
            continue

        if in_trade:
            min_stop     = min(u["stop"] for u in units)
            exit_trigger = price < l_ext or price < min_stop

            if exit_trigger:
                total_shares = sum(u["shares"] for u in units)
                pnl          = sum((price - u["price"]) * u["shares"] for u in units)
                cost         = sum(u["price"] * u["shares"] for u in units)
                cash        += price * total_shares

                holding_days = (dates[i] - entry_date).days if hasattr(dates[i], 'days') else 0
                try:
                    holding_days = (dates[i] - entry_date).days
                except Exception:
                    holding_days = 0

                trades.append({
                    "진입일":    str(entry_date)[:10],
                    "청산일":    str(dates[i])[:10],
                    "진입가":    round(first_entry_price, 2),
                    "청산가":    round(price, 2),
                    "유닛수":    len(units),
                    "수량":      total_shares,
                    "손익":      round(pnl, 0),
                    "손익률(%)": round(pnl / cost * 100, 2) if cost > 0 else 0,
                    "보유일수":  holding_days,
                    "청산사유":  "20일저점" if price < l_ext else "손절",
                })
                units             = []
                in_trade          = False
                first_entry_price = None
                entry_date        = None
            else:
                # 애드업 확인
                if len(units) < max_units:
                    next_add = first_entry_price + len(units) * 0.5 * N
                    if price >= next_add:
                        unit_size = int((capital * risk_pct / 100) / (2 * N))
                        cost      = price * unit_size
                        if unit_size > 0 and cash >= cost:
                            units.append({
                                "price":  price,
                                "stop":   price - 2 * N,
                                "shares": unit_size,
                            })
                            cash -= cost
        else:
            # 진입 확인
            if price > h_ent:
                unit_size = int((capital * risk_pct / 100) / (2 * N))
                cost      = price * unit_size
                if unit_size > 0 and cash >= cost:
                    units = [{"price": price, "stop": price - 2 * N, "shares": unit_size}]
                    cash -= cost
                    in_trade          = True
                    entry_date        = dates[i]
                    first_entry_price = price

        pos_val = sum(price * u["shares"] for u in units)
        equity_pts.append({"date": dates[i], "equity": cash + pos_val})

    # 미청산 포지션 종가 강제 청산
    if in_trade and units:
        price        = float(closes[-1])
        total_shares = sum(u["shares"] for u in units)
        pnl          = sum((price - u["price"]) * u["shares"] for u in units)
        cost         = sum(u["price"] * u["shares"] for u in units)
        cash        += price * total_shares
        try:
            holding_days = (dates[-1] - entry_date).days
        except Exception:
            holding_days = 0
        trades.append({
            "진입일":    str(entry_date)[:10],
            "청산일":    str(dates[-1])[:10],
            "진입가":    round(first_entry_price, 2),
            "청산가":    round(price, 2),
            "유닛수":    len(units),
            "수량":      total_shares,
            "손익":      round(pnl, 0),
            "손익률(%)": round(pnl / cost * 100, 2) if cost > 0 else 0,
            "보유일수":  holding_days,
            "청산사유":  "미청산(종료)",
        })

    eq_df      = pd.DataFrame(equity_pts).set_index("date")
    trades_df  = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 바이앤홀드 비교용 (동일 시작시점 매수)
    bnh_df = pd.DataFrame(index=eq_df.index)
    if len(eq_df) > 0:
        start_price = float(closes[start_idx])
        bnh_shares  = int(capital / start_price) if start_price > 0 else 0
        bnh_cash    = capital - bnh_shares * start_price
        bnh_vals    = [
            bnh_cash + bnh_shares * float(closes[i])
            for i in range(start_idx, len(closes))
        ]
        bnh_df["바이앤홀드"] = bnh_vals

    # 지표 계산
    metrics = _calc_metrics(eq_df, trades_df, capital)

    return {
        "trades":  trades_df,
        "equity":  eq_df,
        "bnh":     bnh_df,
        "metrics": metrics,
    }


def _calc_metrics(eq_df: pd.DataFrame, trades_df: pd.DataFrame, capital: float) -> dict:
    if eq_df.empty or trades_df.empty:
        return {"상태": "거래 신호 없음 — 기간을 늘리거나 파라미터를 조정해보세요."}

    final        = float(eq_df["equity"].iloc[-1])
    total_return = (final - capital) / capital * 100
    years        = max(len(eq_df) / 252, 0.01)
    cagr         = ((final / capital) ** (1 / years) - 1) * 100

    roll_max = eq_df["equity"].cummax()
    dd       = (eq_df["equity"] - roll_max) / roll_max * 100
    max_dd   = float(dd.min())
    mdd_date = str(dd.idxmin())[:10]

    daily_ret = eq_df["equity"].pct_change().dropna()
    sharpe    = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0 else 0.0
    )
    calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

    winners     = trades_df[trades_df["손익"] > 0]
    losers      = trades_df[trades_df["손익"] <= 0]
    win_rate    = len(winners) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_win     = float(winners["손익률(%)"].mean()) if len(winners) > 0 else 0.0
    avg_loss    = float(losers["손익률(%)"].mean())  if len(losers)  > 0 else 0.0
    gross_p     = float(winners["손익"].sum()) if len(winners) > 0 else 0.0
    gross_l     = abs(float(losers["손익"].sum())) if len(losers) > 0 else 0.0
    profit_fac  = gross_p / gross_l if gross_l > 0 else float("inf")
    avg_hold    = float(trades_df["보유일수"].mean()) if "보유일수" in trades_df else 0.0
    max_consec_loss = _max_consecutive_losses(trades_df)

    return {
        "total_return":  total_return,
        "cagr":          cagr,
        "max_dd":        max_dd,
        "mdd_date":      mdd_date,
        "sharpe":        sharpe,
        "calmar":        calmar,
        "win_rate":      win_rate,
        "num_trades":    len(trades_df),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "profit_factor": profit_factor if False else profit_fac,
        "final":         final,
        "avg_hold_days": avg_hold,
        "max_consec_loss": max_consec_loss,
    }


def _max_consecutive_losses(trades_df: pd.DataFrame) -> int:
    if trades_df.empty:
        return 0
    max_cl = cur = 0
    for v in trades_df["손익"].values:
        if v <= 0:
            cur  += 1
            max_cl = max(max_cl, cur)
        else:
            cur = 0
    return max_cl


# ── Streamlit UI ─────────────────────────────────────────────────────────

st.title("📈 터틀 트레이딩 백테스트")
st.caption(
    "System 2 · 진입: N일 신고가 돌파  ·  청산: M일 신저가 이탈 / 2N 손절  ·  애드업: +0.5N 간격 최대 4유닛"
)

# ── 사이드바 ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 백테스트 설정")

names  = [t[0] for t in PORTFOLIO_TICKERS]
chosen = st.sidebar.selectbox("종목 선택", names, index=0)
ticker_map = dict(PORTFOLIO_TICKERS)
ticker_val = ticker_map[chosen]

if ticker_val == "__custom__":
    ticker = st.sidebar.text_input("티커 직접 입력", placeholder="예: AAPL, 005930.KS")
else:
    ticker = ticker_val

period_opts = {"1년": "1y", "2년": "2y", "3년": "3y", "5년": "5y", "10년": "10y", "최대": "max"}
period_lbl  = st.sidebar.selectbox("백테스트 기간", list(period_opts.keys()), index=3)
period      = period_opts[period_lbl]

capital   = st.sidebar.number_input("초기 자본", value=100_000_000, step=5_000_000, format="%d",
                                     help="한국주식은 원(KRW), 미국주식은 달러(USD) 단위로 입력")
risk_pct  = st.sidebar.slider("단위 리스크 (%)", 0.25, 2.0, 1.0, 0.25,
                               help="1유닛 리스크 = 총자본 × 리스크% ÷ (2 × ATR)")

with st.sidebar.expander("고급 파라미터", expanded=False):
    entry_period = st.slider("진입 기간 (일)",  10, 100, 55, 5)
    exit_period  = st.slider("청산 기간 (일)",   5,  55, 20, 5)
    atr_period   = st.slider("ATR 기간 (일)",    5,  30, 20, 1)
    max_units    = st.slider("최대 유닛 수",      1,   6,  4, 1)

run_btn = st.sidebar.button("▶ 백테스트 실행", type="primary", use_container_width=True)

# ── 실행 ─────────────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.error("티커를 입력해주세요.")
        st.stop()

    with st.spinner(f"{ticker} 데이터 다운로드 중..."):
        try:
            raw_df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        except Exception as e:
            st.error(f"데이터 다운로드 실패: {e}")
            st.stop()

    if raw_df is None or raw_df.empty:
        st.error(f"'{ticker}' 데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
        st.stop()

    with st.spinner("백테스트 계산 중..."):
        result = backtest_turtle(
            raw_df, capital, risk_pct / 100,
            entry_period, exit_period, atr_period, max_units,
        )

    st.session_state.update({
        "bt_result":  result,
        "bt_ticker":  ticker,
        "bt_capital": capital,
        "bt_label":   chosen if ticker_val != "__custom__" else ticker,
        "bt_params":  {
            "entry": entry_period, "exit": exit_period,
            "atr": atr_period, "units": max_units,
            "risk": risk_pct, "period": period_lbl,
        },
    })

# ── 결과 표시 ─────────────────────────────────────────────────────────────
if "bt_result" not in st.session_state:
    st.info("왼쪽 사이드바에서 종목과 파라미터를 설정한 뒤 **▶ 백테스트 실행** 버튼을 눌러주세요.")

    st.markdown("""
    ### 백테스트 사용법

    1. **종목 선택** — 포트폴리오 종목 중 하나를 고르거나 직접 티커를 입력합니다.
    2. **기간 설정** — 과거 얼마나 긴 기간을 시뮬레이션할지 선택합니다 (기본: 5년).
    3. **자본·리스크** — 초기 자본과 1유닛당 허용 리스크 비율을 설정합니다.
    4. **실행** — 버튼을 누르면 터틀 System 2 전략이 해당 종목에서 얼마나 수익을 냈을지 계산됩니다.

    ### 전략 규칙 (System 2)

    | 항목 | 내용 |
    |------|------|
    | 진입 | `N`일 신고가 돌파 시 1유닛 매수 |
    | 애드업 | 진입 후 +0.5N / +1.0N / +1.5N 마다 1유닛 추가 (최대 4유닛) |
    | 청산 | `M`일 신저가 이탈 또는 진입가 − 2N 이탈 |
    | 포지션 크기 | 1유닛 = (초기자본 × 리스크%) ÷ (2 × ATR) |
    """)
    st.stop()

result    = st.session_state["bt_result"]
ticker_d  = st.session_state["bt_ticker"]
label_d   = st.session_state["bt_label"]
init_cap  = st.session_state["bt_capital"]
params    = st.session_state["bt_params"]
metrics   = result["metrics"]
eq_df     = result["equity"]
bnh_df    = result["bnh"]
trades_df = result["trades"]

st.subheader(f"📊 {label_d} ({ticker_d})  ·  {params['period']} 백테스트 결과")
st.caption(
    f"진입 {params['entry']}일  ·  청산 {params['exit']}일  ·  ATR {params['atr']}일  ·  "
    f"최대 {params['units']}유닛  ·  단위 리스크 {params['risk']}%  ·  "
    f"초기 자본 {init_cap:,.0f}"
)

# ── 핵심 지표가 없는 경우 ────────────────────────────────────────────────
if "상태" in metrics:
    st.warning(metrics["상태"])
    st.stop()

# ── 상단 지표 카드 ────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

ret_color = "normal" if metrics["total_return"] >= 0 else "inverse"
c1.metric("총 수익률",          f"{metrics['total_return']:+.2f}%",
          delta=f"CAGR {metrics['cagr']:+.2f}%", delta_color=ret_color)
c2.metric("최대 낙폭 (MDD)",    f"{metrics['max_dd']:.2f}%",
          delta=f"최저 {metrics['mdd_date']}", delta_color="inverse")
c3.metric("샤프 비율",          f"{metrics['sharpe']:.2f}")
c4.metric("승률",               f"{metrics['win_rate']:.1f}%",
          delta=f"총 {metrics['num_trades']}회")
c5.metric("손익비 (Profit Factor)", f"{metrics['profit_factor']:.2f}")
c6.metric("최종 자본",          f"{metrics['final']:,.0f}",
          delta=f"{metrics['final'] - init_cap:+,.0f}", delta_color=ret_color)

st.divider()

# ── 세부 지표 ─────────────────────────────────────────────────────────────
with st.expander("📐 세부 성과 지표", expanded=True):
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("평균 수익률 (승리 거래)",  f"{metrics['avg_win']:+.2f}%")
    d1.metric("평균 손실률 (패배 거래)",  f"{metrics['avg_loss']:+.2f}%")
    d2.metric("칼마 비율 (Calmar)",       f"{metrics['calmar']:.2f}")
    d2.metric("최대 연속 손실 횟수",      f"{metrics['max_consec_loss']}회")
    d3.metric("평균 보유 기간",           f"{metrics['avg_hold_days']:.0f}일")
    d3.metric("연환산 수익률 (CAGR)",     f"{metrics['cagr']:+.2f}%")
    d4.metric("총 거래 수",               f"{metrics['num_trades']}회")
    d4.metric("승리 거래 / 패배 거래",
              f"{int(metrics['win_rate'] * metrics['num_trades'] / 100)} / "
              f"{metrics['num_trades'] - int(metrics['win_rate'] * metrics['num_trades'] / 100)}")

st.divider()

# ── 자산 곡선 ─────────────────────────────────────────────────────────────
st.subheader("📈 자산 곡선")

if not eq_df.empty:
    chart_df = eq_df.rename(columns={"equity": "터틀 전략"})
    if not bnh_df.empty:
        chart_df = chart_df.join(bnh_df, how="left")
    st.line_chart(chart_df)

# ── 낙폭 차트 ─────────────────────────────────────────────────────────────
st.subheader("📉 낙폭 (Drawdown)")
if not eq_df.empty:
    roll_max = eq_df["equity"].cummax()
    dd_series = (eq_df["equity"] - roll_max) / roll_max * 100
    dd_df = pd.DataFrame({"낙폭(%)": dd_series}, index=eq_df.index)
    st.area_chart(dd_df)

st.divider()

# ── 거래 내역 ─────────────────────────────────────────────────────────────
st.subheader("🗒️ 거래 내역")

if trades_df.empty:
    st.info("거래 내역이 없습니다.")
else:
    total_trades = len(trades_df)
    winners_cnt  = int((trades_df["손익"] > 0).sum())
    losers_cnt   = total_trades - winners_cnt

    tab_all, tab_win, tab_lose = st.tabs([
        f"전체 ({total_trades})",
        f"✅ 수익 ({winners_cnt})",
        f"❌ 손실 ({losers_cnt})",
    ])

    def style_pnl(val):
        if isinstance(val, (int, float)):
            color = "#2ecc71" if val > 0 else "#e74c3c" if val < 0 else ""
            return f"color: {color}" if color else ""
        return ""

    display_cols = ["진입일", "청산일", "진입가", "청산가", "유닛수", "수량", "손익", "손익률(%)", "보유일수", "청산사유"]

    with tab_all:
        styled = trades_df[display_cols].style.applymap(
            style_pnl, subset=["손익", "손익률(%)"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab_win:
        win_df = trades_df[trades_df["손익"] > 0][display_cols]
        st.dataframe(
            win_df.style.applymap(style_pnl, subset=["손익", "손익률(%)"]),
            use_container_width=True, hide_index=True,
        )

    with tab_lose:
        lose_df = trades_df[trades_df["손익"] <= 0][display_cols]
        st.dataframe(
            lose_df.style.applymap(style_pnl, subset=["손익", "손익률(%)"]),
            use_container_width=True, hide_index=True,
        )

    # 월별 손익 히트맵 (연도 × 월)
    st.divider()
    st.subheader("📅 월별 손익 분포")
    try:
        tdf = trades_df.copy()
        tdf["청산월"] = pd.to_datetime(tdf["청산일"]).dt.to_period("M")
        monthly = tdf.groupby("청산월")["손익률(%)"].sum().reset_index()
        monthly["년도"] = monthly["청산월"].dt.year
        monthly["월"]   = monthly["청산월"].dt.month
        pivot = monthly.pivot(index="년도", columns="월", values="손익률(%)")
        pivot.columns = [f"{m}월" for m in pivot.columns]

        def color_cell(val):
            if pd.isna(val):
                return ""
            return "background-color: #1a6e3c; color: white" if val > 0 else \
                   "background-color: #8b0000; color: white"

        st.dataframe(
            pivot.style.applymap(color_cell).format("{:+.1f}%", na_rep="-"),
            use_container_width=True,
        )
    except Exception:
        pass

st.divider()
st.caption(
    f"백테스트 완료: {datetime.now(KST).strftime('%Y-%m-%d %H:%M')} KST  ·  "
    "과거 성과가 미래 수익을 보장하지 않습니다."
)
