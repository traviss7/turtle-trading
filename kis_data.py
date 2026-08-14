"""한국투자증권(KIS) Open API — 일봉 데이터 조회 모듈.

설정 로드 우선순위:
  1. 환경변수 / .env 파일: KIS_APP_KEY, KIS_APP_SECRET, (선택) KIS_ACCOUNT_NO
  2. kis_config.py: APP_KEY, APP_SECRET, (선택) ACCOUNT_NO

설정이 없으면 get_credentials()가 None을 반환하고,
backtest.py는 자동으로 yfinance 폴백을 사용한다.

토큰은 .kis_token_cache.json에 캐시한다 (KIS 토큰은 24시간 유효,
발급 API에 분당 1회 제한이 있어 매번 새로 받으면 안 된다).
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_CACHE = Path(__file__).parent / ".kis_token_cache.json"

_ENV_LOADED = False


def _load_dotenv():
    """python-dotenv 없이도 동작하도록 .env를 직접 파싱한다."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_credentials():
    """(appkey, appsecret) 튜플 또는 None."""
    _load_dotenv()
    appkey = os.environ.get("KIS_APP_KEY")
    appsecret = os.environ.get("KIS_APP_SECRET")
    if not (appkey and appsecret):
        try:
            import kis_config
            appkey = getattr(kis_config, "APP_KEY", None)
            appsecret = getattr(kis_config, "APP_SECRET", None)
        except ImportError:
            pass
    if appkey and appsecret and "여기에" not in appkey:
        return appkey, appsecret
    return None


def get_access_token(appkey: str, appsecret: str) -> str:
    """접근 토큰 발급 (kis_get_token.py 방식). 23시간 파일 캐시."""
    if TOKEN_CACHE.exists():
        try:
            cache = json.loads(TOKEN_CACHE.read_text())
            if time.time() - cache.get("issued_at", 0) < 23 * 3600:
                return cache["access_token"]
        except (json.JSONDecodeError, KeyError):
            pass

    resp = requests.post(
        f"{BASE_URL}/oauth2/tokenP",
        headers={"content-type": "application/json"},
        json={"grant_type": "client_credentials",
              "appkey": appkey, "appsecret": appsecret},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    TOKEN_CACHE.write_text(json.dumps(
        {"access_token": token, "issued_at": time.time()}))
    return token


def fetch_daily_kis(code: str, start: datetime, end: datetime,
                    appkey: str, appsecret: str) -> tuple:
    """국내주식기간별시세 API로 일봉 조회. (DataFrame, 종목명) 반환.

    endpoint: /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice
    tr_id   : FHKST03010100 — 1회 최대 100건이므로 기간을 나눠 반복 조회.
    """
    token = get_access_token(appkey, appsecret)
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": "FHKST03010100",
        "custtype": "P",
    }

    rows, name = [], code
    chunk_end = end
    while chunk_end >= start:
        chunk_start = max(start, chunk_end - timedelta(days=140))  # 100 영업일 여유
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",           # 주식/ETF/ETN
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": chunk_start.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": chunk_end.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",              # 일봉
            "FID_ORG_ADJ_PRC": "0",                  # 0: 수정주가
        }
        resp = requests.get(
            f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=headers, params=params, timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("rt_cd") != "0":
            raise RuntimeError(f"KIS API 오류 [{code}]: {body.get('msg1')}")
        name = body.get("output1", {}).get("hts_kor_isnm") or name
        for item in body.get("output2", []):
            if item.get("stck_bsop_date"):
                rows.append(item)
        chunk_end = chunk_start - timedelta(days=1)
        time.sleep(0.25)  # 초당 호출 제한(20건/초) 여유

    if not rows:
        raise RuntimeError(f"KIS API에서 {code} 데이터를 받지 못했습니다.")

    df = pd.DataFrame([{
        "Date": pd.to_datetime(r["stck_bsop_date"]),
        "Open": float(r["stck_oprc"]),
        "High": float(r["stck_hgpr"]),
        "Low": float(r["stck_lwpr"]),
        "Close": float(r["stck_clpr"]),
        "Volume": float(r["acml_vol"]),
    } for r in rows])
    df = (df.drop_duplicates("Date").set_index("Date")
            .sort_index().loc[str(start.date()):str(end.date())])
    return df, name


def fetch_daily_yf(code: str, start: datetime, end: datetime) -> tuple:
    """yfinance 폴백 — .KS(코스피) 우선, 실패 시 .KQ(코스닥)."""
    import yfinance as yf

    for suffix in (".KS", ".KQ"):
        ticker = code + suffix
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True, threads=False)
        except Exception:
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"
            try:
                name = yf.Ticker(ticker).info.get("shortName") or code
            except Exception:
                name = code
            return df, name
    raise RuntimeError(f"yfinance에서 {code}(.KS/.KQ) 데이터를 받지 못했습니다.")


def fetch_daily(code: str, start: datetime, end: datetime,
                source: str = "auto") -> tuple:
    """일봉 OHLCV 조회. (DataFrame[Open,High,Low,Close,Volume], 종목명) 반환.

    source: "auto" | "kis" | "yf"
      auto: KIS 설정이 있으면 KIS, 없으면 yfinance 폴백
    """
    creds = get_credentials()
    if source == "kis" and creds is None:
        raise RuntimeError(
            "KIS API 설정이 없습니다. .env 또는 kis_config.py에 "
            "KIS_APP_KEY/KIS_APP_SECRET을 채워주세요. "
            "(.env.example / kis_config.py.example 참고)")

    if source in ("auto", "kis") and creds is not None:
        try:
            return fetch_daily_kis(code, start, end, *creds)
        except Exception as e:
            if source == "kis":
                raise
            print(f"  [경고] KIS API 조회 실패({e}) → yfinance 폴백")
    elif source in ("auto", "kis") and creds is None:
        print("  [안내] KIS API 설정이 없어 yfinance 데이터를 사용합니다. "
              "(.env.example 참고하여 설정하면 KIS API 사용)")
    return fetch_daily_yf(code, start, end)
