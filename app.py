import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

st.set_page_config(page_title="🐢 터틀 트레이딩", layout="wide")
st.title("🐢 터틀 트레이딩 대시보드")

# 사이드바 입력
st.sidebar.header("📊 설정")
symbol = st.sidebar.text_input("종목 코드", value="AAPL", help="미국주식: AAPL, 한국주식: 005930.KS")
capital = st.sidebar.number_input("총 자본 (원/달러)", value=10000000, step=100000)
risk_pct = st.sidebar.slider("리스크 (%)", 0.5, 3.0, 1.0, 0.5)
atr_period = st.sidebar.selectbox("ATR 기간", [14, 20], index=1)
entry_period = st.sidebar.selectbox("진입 기준", [20, 55], index=0)

if st.sidebar.button("🔍 조회", type="primary"):
    with st.spinner("데이터 조회 중..."):
        try:
            # 데이터 가져오기
            df = yf.download(symbol, period="6mo", progress=False)
            
            if df.empty:
                st.error("종목을 찾을 수 없습니다.")
            else:
                df = df.copy()
                close = df['Close'].values.flatten()
                high = df['High'].values.flatten()
                low = df['Low'].values.flatten()
                
                current_price = float(close[-1])
                
                # ATR 계산
                tr_list = []
                for i in range(1, len(df)):
                    tr = max(
                        high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])
                    )
                    tr_list.append(tr)
                
                atr = np.mean(tr_list[-atr_period:])
                n = atr  # 터틀에서 N = ATR
                
                # 돌파 기준
                high_20 = np.max(close[-20:])
                high_55 = np.max(close[-55:]) if len(close) >= 55 else np.max(close)
                
                entry_high = high_20 if entry_period == 20 else high_55
                breakout = current_price > entry_high
                
                # 포지션 사이징
                dollar_risk = capital * (risk_pct / 100)
                unit_size = int(dollar_risk / (2 * n))
                
                # 진입가 & 손절가
                entry_price = entry_high
                stop_loss = entry_price - (2 * n)
                
                # 애드업 가격
                add1 = entry_price + (0.5 * n)
                add2 = entry_price + (1.0 * n)
                add3 = entry_price + (1.5 * n)
                
                # 결과 표시
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("현재가", f"{current_price:,.2f}")
                with col2:
                    st.metric(f"{entry_period}일 고점", f"{entry_high:,.2f}")
                with col3:
                    if breakout:
                        st.success("✅ 돌파 (매수 신호)")
                    else:
                        st.warning("❌ 미돌파 (대기)")
                
                st.divider()
                
                # ATR 정보
                st.subheader(f"📈 ATR({atr_period}) = N")
                st.info(f"**N = {n:,.2f}** (변동성 단위)")
                
                st.divider()
                
                # 포지션 계산
                st.subheader("💰 포지션 사이징")
                st.write(f"- 총 자본: **{capital:,}**")
                st.write(f"- 리스크: **{risk_pct}%** = **{dollar_risk:,.0f}**")
                st.write(f"- 2N = **{2*n:,.2f}**")
                st.write(f"- **1 Unit = {unit_size}주**")
                
                st.divider()
                
                # 진입/손절/애드업
                st.subheader("🎯 매매 포인트")
                
                points = pd.DataFrame({
                    "구분": ["진입가", "손절가 (-2N)", "애드업 1차 (+0.5N)", "애드업 2차 (+1N)", "애드업 3차 (+1.5N)"],
                    "가격": [f"{entry_price:,.2f}", f"{stop_loss:,.2f}", f"{add1:,.2f}", f"{add2:,.2f}", f"{add3:,.2f}"],
                    "수량": [f"{unit_size}주", "-", f"{unit_size}주", f"{unit_size}주", f"{unit_size}주"]
                })
                st.table(points)
                
                # 차트
                st.subheader("📊 가격 차트")
                chart_data = pd.DataFrame({
                    "종가": close[-60:],
                    f"{entry_period}일 고점": [entry_high] * min(60, len(close)),
                    "손절가": [stop_loss] * min(60, len(close))
                })
                st.line_chart(chart_data)
                
        except Exception as e:
            st.error(f"오류 발생: {e}")

else:
    st.info("👈 왼쪽에서 종목 코드 입력 후 '조회' 버튼을 누르세요")
    st.write("")
    st.write("**예시 종목 코드:**")
    st.write("- 미국주식: AAPL, TSLA, NVDA, SPY")
    st.write("- 한국주식: 005930.KS (삼성전자), 000660.KS (SK하이닉스)")
