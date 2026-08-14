"""터틀 트레이딩 원본 규칙 — 순수 함수/클래스 모음.

백테스트 엔진(backtest.py)과 실시간 대시보드가 공유하는 핵심 로직.
데이터 조회나 시각화는 여기서 하지 않는다.

규칙 요약 (롱 온리):
- N        : ATR 20일 (Wilder 방식 지수평활)
- 진입     : System 1 = 20일 신고가 돌파 / System 2 = 55일 신고가 돌파
- 청산     : System 1 = 10일 신저가 이탈 / System 2 = 20일 신저가 이탈
- 손절     : 마지막 진입가 - 2N (피라미딩 시 전체 유닛의 손절가를 함께 끌어올림)
- 피라미딩 : 직전 진입가 + 0.5N 마다 1유닛 추가, 종목당 최대 4유닛
- 유닛 크기: 계좌자본 x 1% / N  (N 1단위 변동 = 자본의 1%)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ATR_PERIOD = 20
MAX_UNITS = 4          # 종목(시장)당 최대 유닛
RISK_PER_UNIT = 0.01   # 1유닛 리스크 = 자본의 1%
STOP_N = 2.0           # 손절: 진입가 - 2N
PYRAMID_STEP_N = 0.5   # 피라미딩 간격: +0.5N

ENTRY_WINDOW = {1: 20, 2: 55}   # 시스템별 진입 채널
EXIT_WINDOW = {1: 10, 2: 20}    # 시스템별 청산 채널


# ── 지표 ────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, atr_period: int = ATR_PERIOD) -> pd.DataFrame:
    """OHLCV DataFrame에 터틀 지표 컬럼을 추가해 반환한다.

    추가 컬럼:
      N        : ATR(atr_period), Wilder 평활
      high_20  : 전일까지의 20일 최고가 (당일 제외 → 당일 돌파 판정용)
      high_55  : 전일까지의 55일 최고가
      low_10   : 전일까지의 10일 최저가
      low_20   : 전일까지의 20일 최저가
    """
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder 평활: N = (19 * N_prev + TR) / 20
    df["N"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # 채널은 "전일까지"의 극값 — 당일 고가/저가와 비교해 돌파를 판정한다
    df["high_20"] = high.rolling(ENTRY_WINDOW[1]).max().shift(1)
    df["high_55"] = high.rolling(ENTRY_WINDOW[2]).max().shift(1)
    df["low_10"] = low.rolling(EXIT_WINDOW[1]).min().shift(1)
    df["low_20"] = low.rolling(EXIT_WINDOW[2]).min().shift(1)
    return df


# ── 신호 판정 (하루치 row 기준) ─────────────────────────────────────────

def check_entry(row: pd.Series, system: int = 2) -> bool:
    """당일 고가가 진입 채널(전일까지의 20/55일 최고가)을 돌파했는가."""
    channel = row["high_20"] if system == 1 else row["high_55"]
    if pd.isna(channel) or pd.isna(row["N"]):
        return False
    return row["High"] > channel


def check_exit(row: pd.Series, system: int = 2) -> bool:
    """당일 저가가 청산 채널(전일까지의 10/20일 최저가)을 이탈했는가."""
    channel = row["low_10"] if system == 1 else row["low_20"]
    if pd.isna(channel):
        return False
    return row["Low"] < channel


def entry_channel(row: pd.Series, system: int = 2) -> float:
    return float(row["high_20"] if system == 1 else row["high_55"])


def exit_channel(row: pd.Series, system: int = 2) -> float:
    return float(row["low_10"] if system == 1 else row["low_20"])


# ── 포지션 사이징 ───────────────────────────────────────────────────────

def unit_shares(capital: float, n: float, risk_pct: float = RISK_PER_UNIT) -> int:
    """1유닛 주식 수 = (계좌자본 x risk_pct) / N. 정수 내림."""
    if n is None or n <= 0 or np.isnan(n):
        return 0
    return int((capital * risk_pct) / n)


# ── 포지션 상태 ─────────────────────────────────────────────────────────

@dataclass
class Unit:
    """피라미딩 1유닛."""
    entry_date: object
    entry_price: float
    shares: int
    n: float  # 진입 시점의 N


@dataclass
class Position:
    """한 종목의 롱 포지션 (최대 MAX_UNITS 유닛)."""
    system: int = 2
    units: list = field(default_factory=list)
    stop_price: float = float("nan")

    @property
    def is_open(self) -> bool:
        return len(self.units) > 0

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def total_shares(self) -> int:
        return sum(u.shares for u in self.units)

    @property
    def avg_price(self) -> float:
        shares = self.total_shares
        if shares == 0:
            return float("nan")
        return sum(u.entry_price * u.shares for u in self.units) / shares

    @property
    def last_entry_price(self) -> float:
        return self.units[-1].entry_price if self.units else float("nan")

    @property
    def next_add_price(self) -> float:
        """다음 피라미딩 목표가 = 마지막 진입가 + 0.5N. 최대 유닛이면 nan."""
        if not self.units or self.unit_count >= MAX_UNITS:
            return float("nan")
        last = self.units[-1]
        return last.entry_price + PYRAMID_STEP_N * last.n

    def add_unit(self, entry_date, entry_price: float, shares: int, n: float):
        """유닛 추가. 손절가는 전체 유닛에 대해 '마지막 진입가 - 2N'으로 갱신."""
        if self.unit_count >= MAX_UNITS:
            raise ValueError("최대 유닛 초과")
        self.units.append(Unit(entry_date, entry_price, shares, n))
        self.stop_price = entry_price - STOP_N * n

    def close(self):
        closed = list(self.units)
        self.units = []
        self.stop_price = float("nan")
        return closed
