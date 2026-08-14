"""터틀 트레이딩 백테스트 — 범용 스크립트.

사용법:
    python backtest.py                  # 기본값: SK하이닉스(000660), 최근 1년
    python backtest.py 000660           # 종목코드 지정
    python backtest.py 000660 005930    # 여러 종목 백테스트 + 비교 차트
    python backtest.py 000660 --system 1 --capital 200000000 --years 2

전략 로직은 turtle_core.py를 그대로 import해서 사용하고,
이 파일은 일자별 순회 시뮬레이션(체결/자본 관리)과 결과 집계만 담당한다.
데이터는 kis_data.fetch_daily()로 조회한다 (KIS API, 설정 없으면 yfinance 폴백).
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kis_data import fetch_daily
from turtle_core import (
    MAX_UNITS,
    Position,
    add_indicators,
    check_entry,
    check_exit,
    entry_channel,
    exit_channel,
    unit_shares,
)

WARMUP_CALENDAR_DAYS = 160  # 55일 채널 + ATR20 워밍업용 여유 (영업일 ~110일)


# ── 한글 폰트 (없으면 영문 라벨로 폴백) ─────────────────────────────────

def _setup_korean_font() -> bool:
    import matplotlib.font_manager as fm
    candidates = ["NanumGothic", "NanumBarunGothic", "Malgun Gothic",
                  "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_KR_FONT = _setup_korean_font()

L = {  # 차트 라벨 (한글 폰트 없으면 영문)
    "price": "주가" if HAS_KR_FONT else "Price",
    "buy": "매수" if HAS_KR_FONT else "Buy",
    "add": "피라미딩" if HAS_KR_FONT else "Add unit",
    "sell": "매도" if HAS_KR_FONT else "Sell",
    "equity": "자본금" if HAS_KR_FONT else "Equity",
    "bh": "단순보유" if HAS_KR_FONT else "Buy & Hold",
    "dd": "드로다운" if HAS_KR_FONT else "Drawdown",
    "krw": "원" if HAS_KR_FONT else "KRW",
}


# ── 백테스트 엔진 ───────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, capital: float, system: int = 2) -> dict:
    """지표가 붙은 일봉 df를 순회하며 터틀 규칙대로 매매를 시뮬레이션한다.

    반환: equity 곡선(DataFrame), 체결 내역(fills), 라운드트립(trades), 지표(metrics)
    """
    cash = capital
    pos = Position(system=system)
    fills = []          # 개별 체결 (매수/매도 마커용)
    trades = []         # 라운드트립 (진입~전량청산)
    open_trade = None
    equity_rows = []

    for date, row in df.iterrows():
        if pos.is_open:
            # 1) 손절: 저가가 손절가 이하로 내려오면 전량 청산
            #    (시가가 이미 손절가 아래면 시가 체결 — 갭하락 반영)
            if row["Low"] <= pos.stop_price:
                price = min(row["Open"], pos.stop_price)
                cash += _close_position(pos, date, price, "stop",
                                        fills, trades, open_trade)
                open_trade = None
            # 2) 채널 청산: 저가가 10/20일 신저가 아래로 이탈
            elif check_exit(row, system):
                price = min(row["Open"], exit_channel(row, system))
                cash += _close_position(pos, date, price, "channel",
                                        fills, trades, open_trade)
                open_trade = None
            else:
                # 3) 피라미딩: 고가가 '직전 진입가 + 0.5N' 도달 시 1유닛 추가
                while (pos.unit_count < MAX_UNITS
                       and not np.isnan(pos.next_add_price)
                       and row["High"] >= pos.next_add_price):
                    price = max(row["Open"], pos.next_add_price)
                    shares = unit_shares(capital, row["N"])
                    shares = min(shares, int(cash // price))
                    if shares <= 0:
                        break
                    pos.add_unit(date, price, shares, row["N"])
                    cash -= shares * price
                    fills.append({"date": date, "price": price, "shares": shares,
                                  "side": "buy", "kind": "add",
                                  "unit": pos.unit_count})
                    open_trade["entries"].append(
                        {"date": date, "price": price, "shares": shares})
        else:
            # 4) 신규 진입: 고가가 20/55일 신고가 돌파 시 1유닛 매수
            if check_entry(row, system):
                price = max(row["Open"], entry_channel(row, system))
                shares = unit_shares(capital, row["N"])
                shares = min(shares, int(cash // price))
                if shares > 0:
                    pos.add_unit(date, price, shares, row["N"])
                    cash -= shares * price
                    fills.append({"date": date, "price": price, "shares": shares,
                                  "side": "buy", "kind": "entry", "unit": 1})
                    open_trade = {
                        "entries": [{"date": date, "price": price,
                                     "shares": shares}]}

        equity_rows.append({
            "Date": date,
            "equity": cash + pos.total_shares * row["Close"],
            "cash": cash,
            "shares": pos.total_shares,
            "units": pos.unit_count,
            "stop": pos.stop_price if pos.is_open else np.nan,
        })

    # 기간 종료 시 미청산 포지션은 마지막 종가로 평가 청산
    if pos.is_open:
        last_date = df.index[-1]
        last_close = float(df["Close"].iloc[-1])
        cash += _close_position(pos, last_date, last_close, "eop",
                                fills, trades, open_trade)
        equity_rows[-1].update({"equity": cash, "cash": cash,
                                "shares": 0, "units": 0})

    equity = pd.DataFrame(equity_rows).set_index("Date")
    metrics = _compute_metrics(equity, trades, df, capital)
    return {"equity": equity, "fills": fills, "trades": trades,
            "metrics": metrics}


def _close_position(pos, date, price, reason, fills, trades, open_trade):
    """전량 청산 처리: 체결/라운드트립 기록 후 매도 대금을 반환."""
    shares = pos.total_shares
    avg = pos.avg_price
    closed_units = pos.close()
    fills.append({"date": date, "price": price, "shares": shares,
                  "side": "sell", "kind": reason, "unit": 0})
    pnl = (price - avg) * shares
    trades.append({
        "entry_date": closed_units[0].entry_date,
        "exit_date": date,
        "units": len(closed_units),
        "shares": shares,
        "avg_entry": avg,
        "exit_price": price,
        "pnl": pnl,
        "pnl_pct": (price - avg) / avg * 100,
        "reason": reason,
        "entries": open_trade["entries"] if open_trade else [],
    })
    return shares * price


def _compute_metrics(equity, trades, df, capital) -> dict:
    eq = equity["equity"]
    final = float(eq.iloc[-1])
    peak = eq.cummax()
    dd = (eq - peak) / peak
    wins = [t for t in trades if t["pnl"] > 0]
    bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100
    return {
        "initial": capital,
        "final": final,
        "return_pct": (final / capital - 1) * 100,
        "bh_return_pct": bh,
        "n_trades": len(trades),
        "n_wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "mdd_pct": float(dd.min()) * 100,
        "total_pnl": final - capital,
        "avg_pnl": np.mean([t["pnl"] for t in trades]) if trades else 0.0,
    }


# ── 결과 출력 ───────────────────────────────────────────────────────────

REASON_KO = {"stop": "손절(-2N)", "channel": "채널청산", "eop": "기간종료"}


def print_summary(code, name, system, res, start, end):
    m = res["metrics"]
    line = "─" * 62
    print(f"\n{line}")
    print(f"  터틀 트레이딩 백테스트 — {name} ({code})  System {system}")
    print(f"  기간: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}  |  초기자본 {m['initial']/1e8:.1f}억원")
    print(line)
    print(f"  최종 자본        : {m['final']:>15,.0f} 원")
    print(f"  총 손익          : {m['total_pnl']:>+15,.0f} 원")
    print(f"  수익률           : {m['return_pct']:>+14.2f} %   (단순보유 {m['bh_return_pct']:+.2f}%)")
    print(f"  총 거래(라운드트립): {m['n_trades']:>12d} 회")
    print(f"  승률             : {m['win_rate']:>14.1f} %   ({m['n_wins']}승 {m['n_trades']-m['n_wins']}패)")
    print(f"  최대 손실폭(MDD) : {m['mdd_pct']:>14.2f} %")
    print(line)
    if res["trades"]:
        print("  [거래 내역]")
        for i, t in enumerate(res["trades"], 1):
            print(f"   {i:2d}. {t['entry_date']:%Y-%m-%d} 진입({t['units']}유닛, "
                  f"평균 {t['avg_entry']:,.0f}) → {t['exit_date']:%Y-%m-%d} "
                  f"{REASON_KO.get(t['reason'], t['reason'])} "
                  f"{t['exit_price']:,.0f}  손익 {t['pnl']:+,.0f}원 "
                  f"({t['pnl_pct']:+.1f}%)")
    else:
        print("  [거래 내역] 진입 신호 없음")
    print(line)


def plot_result(code, name, system, res, df, outpath: Path):
    """주가(매매 시점 표시) + 자본 곡선 2단 차트 저장."""
    equity = res["equity"]
    fills = res["fills"]
    view = df.loc[equity.index[0]:]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]})
    title_name = name if HAS_KR_FONT else code
    fig.suptitle(f"Turtle Backtest — {title_name} ({code})  System {system}",
                 fontsize=14, fontweight="bold")

    # 상단: 주가 + 채널 + 매매 마커
    ax1.plot(view.index, view["Close"], color="#1f77b4", lw=1.2,
             label=L["price"])
    ent_col = "high_20" if system == 1 else "high_55"
    exi_col = "low_10" if system == 1 else "low_20"
    ax1.plot(view.index, view[ent_col], color="#2ca02c", lw=0.8, ls="--",
             alpha=0.6, label=f"Entry ch. ({ent_col})")
    ax1.plot(view.index, view[exi_col], color="#d62728", lw=0.8, ls="--",
             alpha=0.6, label=f"Exit ch. ({exi_col})")

    buys = [f for f in fills if f["side"] == "buy" and f["kind"] == "entry"]
    adds = [f for f in fills if f["side"] == "buy" and f["kind"] == "add"]
    sells = [f for f in fills if f["side"] == "sell"]
    if buys:
        ax1.scatter([f["date"] for f in buys], [f["price"] for f in buys],
                    marker="^", s=110, color="#2ca02c", zorder=5,
                    edgecolors="k", linewidths=0.5, label=L["buy"])
    if adds:
        ax1.scatter([f["date"] for f in adds], [f["price"] for f in adds],
                    marker="^", s=60, color="#98df8a", zorder=5,
                    edgecolors="k", linewidths=0.5, label=L["add"])
    if sells:
        ax1.scatter([f["date"] for f in sells], [f["price"] for f in sells],
                    marker="v", s=110, color="#d62728", zorder=5,
                    edgecolors="k", linewidths=0.5, label=L["sell"])
    ax1.set_ylabel(f"{L['price']} ({L['krw']})")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # 하단: 자본 곡선 + 단순보유 비교 + 드로다운 음영
    m = res["metrics"]
    bh_curve = m["initial"] * view["Close"] / float(view["Close"].iloc[0])
    ax2.plot(equity.index, equity["equity"], color="#ff7f0e", lw=1.5,
             label=f"{L['equity']} ({m['return_pct']:+.1f}%)")
    ax2.plot(bh_curve.index, bh_curve, color="#7f7f7f", lw=1.0, ls=":",
             label=f"{L['bh']} ({m['bh_return_pct']:+.1f}%)")
    peak = equity["equity"].cummax()
    ax2.fill_between(equity.index, equity["equity"], peak,
                     color="#d62728", alpha=0.15,
                     label=f"{L['dd']} (MDD {m['mdd_pct']:.1f}%)")
    ax2.axhline(m["initial"], color="k", lw=0.6, alpha=0.5)
    ax2.set_ylabel(f"{L['equity']} ({L['krw']})")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e8:.2f}" + ("억" if HAS_KR_FONT else "e8")))

    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"  차트 저장: {outpath}")


def plot_comparison(all_results: dict, capital: float, outpath: Path):
    """여러 종목 자본 곡선 비교 차트."""
    fig, ax = plt.subplots(figsize=(13, 6))
    for code, (name, res) in all_results.items():
        eq = res["equity"]["equity"]
        label_name = name if HAS_KR_FONT else code
        ax.plot(eq.index, eq / capital * 100 - 100,
                lw=1.5, label=f"{label_name} ({res['metrics']['return_pct']:+.1f}%)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_ylabel("Return (%)")
    ax.set_title("Turtle Backtest — Comparison", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110)
    plt.close(fig)
    print(f"\n비교 차트 저장: {outpath}")


# ── 메인 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="터틀 트레이딩 백테스트 (국내주식, KIS API/yfinance)")
    parser.add_argument("codes", nargs="*", default=["000660"],
                        help="종목코드 (기본: 000660 SK하이닉스). 여러 개 입력 시 비교")
    parser.add_argument("--system", type=int, choices=[1, 2], default=2,
                        help="1: 20일 진입/10일 청산, 2: 55일 진입/20일 청산 (기본 2)")
    parser.add_argument("--capital", type=float, default=100_000_000,
                        help="초기 자본금 (기본 1억원)")
    parser.add_argument("--years", type=float, default=1.0,
                        help="백테스트 기간(년, 기본 1)")
    parser.add_argument("--source", choices=["auto", "kis", "yf"],
                        default="auto", help="데이터 소스 (기본 auto)")
    parser.add_argument("--outdir", default="results", help="차트 저장 폴더")
    args = parser.parse_args()

    end = datetime.now()
    start = end - timedelta(days=int(365 * args.years))
    fetch_start = start - timedelta(days=WARMUP_CALENDAR_DAYS)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    all_results = {}
    for code in args.codes:
        print(f"\n[{code}] 일봉 데이터 조회 중...")
        try:
            df, name = fetch_daily(code, fetch_start, end, source=args.source)
        except Exception as e:
            print(f"  [오류] {code} 데이터 조회 실패: {e}")
            continue
        df = add_indicators(df)
        bt_df = df.loc[df.index >= pd.Timestamp(start.date())]
        if len(bt_df) < 30:
            print(f"  [오류] {code}: 백테스트 구간 데이터 부족 ({len(bt_df)}일)")
            continue

        res = run_backtest(bt_df, args.capital, system=args.system)
        print_summary(code, name, args.system, res,
                      bt_df.index[0], bt_df.index[-1])
        plot_result(code, name, args.system, res, df,
                    outdir / f"backtest_{code}.png")
        all_results[code] = (name, res)

    if len(all_results) > 1:
        plot_comparison(all_results, args.capital, outdir / "comparison.png")
        print("\n[종목 비교]")
        print(f"  {'종목':<20}{'수익률':>10}{'거래':>6}{'승률':>8}{'MDD':>9}")
        for code, (name, res) in all_results.items():
            m = res["metrics"]
            print(f"  {name+'('+code+')':<20}{m['return_pct']:>+9.2f}%"
                  f"{m['n_trades']:>6d}{m['win_rate']:>7.1f}%"
                  f"{m['mdd_pct']:>8.2f}%")

    if not all_results:
        print("\n백테스트를 수행한 종목이 없습니다.")
        sys.exit(1)

    # 대화형 환경이면 차트 표시 시도 (헤드리스면 자동 스킵)
    import os
    if os.environ.get("DISPLAY"):
        try:
            import subprocess
            for code in all_results:
                subprocess.Popen(["xdg-open", str(outdir / f"backtest_{code}.png")])
        except Exception:
            pass


if __name__ == "__main__":
    main()
