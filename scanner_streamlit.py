#!/usr/bin/env python3
"""
==========================================================================
  Veteran v4.0 — Streamlit Scanner + Backtest
==========================================================================
  Scanner: same logic as daily_scanner.py (Core + Score 2/3)
  Backtest: same logic as backtest_options.py (Veteran backtest engine)
  Run:   streamlit run scanner_streamlit.py
  Deps:  pip install streamlit yfinance pandas ta
==========================================================================
"""

import time
import pandas as pd
import streamlit as st

# Reuse scanner logic from the existing CLI script
from daily_scanner import (
    analyze_stock,
    get_tickers,
    DEFAULT_TICKERS,
    HK_TICKERS,
    US_TICKERS,
)

st.set_page_config(
    page_title="Veteran v4.0 Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Veteran v4.0 Scanner & Backtest")
st.caption("Core: Close>SMA20, 20<ADX<50, PDI>MDI  |  Score 2/3: RSI>50, MFI>55, RVOL≥1.0")

tab_scan, tab_backtest = st.tabs(["Scanner", "Backtest"])

# ========== TAB 1: Scanner ==========
with tab_scan:
    st.sidebar.header("Ticker source")
    source = st.sidebar.radio(
        "Choose list",
        ["HK stocks", "US stocks", "Default list", "Custom (type below)"],
        index=0,
    )

    tickers = []
    if source == "HK stocks":
        tickers = HK_TICKERS.copy()
        st.sidebar.info(f"Using {len(tickers)} HK tickers.")
    elif source == "US stocks":
        tickers = US_TICKERS.copy()
        st.sidebar.info(f"Using {len(tickers)} US tickers.")
    elif source == "Default list":
        tickers = DEFAULT_TICKERS.copy()
        st.sidebar.info(f"Using default list ({len(tickers)} tickers).")
    else:
        custom = st.sidebar.text_area(
            "Enter tickers (one per line or comma-separated)",
            placeholder="0700.HK\n9988.HK\nNVDA",
            height=120,
        )
        if custom:
            raw = custom.replace(",", " ").split()
            tickers = [t.strip().upper() for t in raw if t.strip()]
        if not tickers and source == "Custom (type below)":
            st.sidebar.warning("Enter at least one ticker.")

    if tickers:
        if st.button("Run scan", type="primary", key="scan_btn"):
            progress = st.progress(0, text="Scanning...")
            results = []
            n = len(tickers)
            for i, t in enumerate(tickers):
                res = analyze_stock(t)
                if res:
                    results.append(res)
                progress.progress((i + 1) / n, text=f"Scanning {t}...")
                time.sleep(0.2)
            progress.progress(1.0, text="Done.")
            time.sleep(0.3)
            progress.empty()

            if results:
                st.success(f"Found **{len(results)}** actionable signal(s).")
                df = pd.DataFrame(results)
                cols = ["Ticker", "Price", "Signal", "Why", "ADX", "RSI", "RVOL"]
                df = df[[c for c in cols if c in df.columns]]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No actionable signals today. Stay cash.")

        st.sidebar.divider()
        st.sidebar.metric("Tickers to scan", len(tickers))
    else:
        st.info("Select a ticker source in the sidebar and (for Custom) enter at least one ticker.")

# ========== TAB 2: Backtest ==========
with tab_backtest:
    st.subheader("Veteran backtest (single symbol)")
    st.caption("Uses same Core + Score 2/3 logic as the scanner. Data: yfinance (1y default).")

    # Backtest uses backtest_options (fetch_data, add_indicators, run_veteran_backtest)
    try:
        from backtest_options import (
            fetch_data_yfinance,
            add_indicators,
            run_veteran_backtest,
        )
    except ImportError as e:
        st.error(f"Backtest requires backtest_options.py: {e}")
        st.stop()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.text_input("Ticker", value="9988.HK", key="bt_symbol", placeholder="0700.HK or NVDA")
    with col2:
        period = st.selectbox("Period", ["1y", "6mo", "2y"], index=0, key="bt_period")
    with col3:
        use_smart_exit = st.checkbox("Smart exit (trail + profit take)", value=True, key="bt_smart")

    if st.button("Run backtest", type="primary", key="backtest_btn"):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            st.warning("Enter a ticker.")
        else:
            with st.spinner(f"Fetching {symbol} and running backtest..."):
                try:
                    df = fetch_data_yfinance(symbol, period=period)
                    df = add_indicators(df)
                    required = ["SMA20", "RSI14", "ADX", "ADX_prev", "PDI", "MDI", "MFI14", "RVOL", "ATR14"]
                    valid = df.dropna(subset=required)
                    if len(valid) < 10:
                        st.warning(f"Not enough valid bars after warm-up ({len(valid)}). Try a longer period.")
                    else:
                        trades = run_veteran_backtest(valid, verbose=False, use_smart_exit=use_smart_exit)

                        if not trades:
                            st.info(f"No trades triggered for **{symbol}** in this period. Data: {valid.index[0].date()} → {valid.index[-1].date()} ({len(valid)} bars).")
                        else:
                            tdf = pd.DataFrame(trades)
                            wins = tdf[tdf["Result"] == "WIN"]
                            losses = tdf[tdf["Result"] == "LOSS"]
                            n = len(tdf)
                            total_pnl = tdf["PnL"].sum()
                            win_rate = len(wins) / n * 100
                            avg_win = wins["PnL%"].mean() if len(wins) > 0 else 0
                            avg_loss = losses["PnL%"].mean() if len(losses) > 0 else 0

                            st.success(f"**{symbol}** — {n} trades | Win rate {win_rate:.1f}% | Total P&L **HK$ {total_pnl:+.2f}**")
                            st.metric("Total P&L (HK$)", f"{total_pnl:+.2f}", None)
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Trades", n, None)
                            c2.metric("Wins / Losses", f"{len(wins)} / {len(losses)}", None)
                            c3.metric("Avg Win %", f"{avg_win:+.2f}%", None)
                            c4.metric("Avg Loss %", f"{avg_loss:+.2f}%", None)

                            log_cols = ["Entry_Date", "Entry_Price", "E_ADX", "Exit_Date", "Exit_Price", "Hold_Days", "PnL", "PnL%", "Result", "Exit_Reason"]
                            log_cols = [c for c in log_cols if c in tdf.columns]
                            st.dataframe(tdf[log_cols], use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
