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

import sys
from pathlib import Path

# Ensure the app directory is on Python path (so backtest_options + daily_scanner are found)
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

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

    # Load backtest_options from the same directory as this script (works from any cwd / Streamlit Cloud)
    import importlib.util
    _bt_path = _here / "backtest_options.py"
    if not _bt_path.exists():
        st.error("Backtest requires **backtest_options.py** in the same folder as this app. File not found.")
        st.stop()
    try:
        spec = importlib.util.spec_from_file_location("backtest_options", _bt_path)
        _bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bt)
        fetch_data_yfinance = _bt.fetch_data_yfinance
        add_indicators = _bt.add_indicators
        run_veteran_backtest = _bt.run_veteran_backtest
    except Exception as e:
        st.error(f"Could not load backtest_options.py: {e}")
        st.stop()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.text_input("Ticker", value="9988.HK", key="bt_symbol", placeholder="0700.HK or NVDA")
    with col2:
        period = st.selectbox("Period", ["1y", "6mo", "2y"], index=0, key="bt_period")
    with col3:
        use_smart_exit = st.checkbox("Smart exit (trail + profit take)", value=True, key="bt_smart")

    with st.expander("Adjust criteria (Core & Scorecard)", expanded=False):
        st.caption("Core: turn conditions on/off or change thresholds. Scorecard: RSI, MFI, RVOL; trigger = min points out of 3.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Core (BUY)**")
            core_require_trend = st.checkbox("Require Close > SMA20", value=True, key="core_trend")
            core_require_pdi_mdi = st.checkbox("Require PDI > MDI", value=True, key="core_pdi_mdi")
            adx_min = st.number_input("ADX min (trend floor)", min_value=10, max_value=40, value=20, step=1, key="adx_min")
            adx_max = st.number_input("ADX max (not overheated)", min_value=30, max_value=70, value=50, step=1, key="adx_max")
        with c2:
            st.markdown("**Scorecard (1 pt each)**")
            rsi_entry = st.number_input("RSI > (momentum)", min_value=30, max_value=70, value=50, step=1, key="rsi_entry")
            mfi_entry = st.number_input("MFI > (money flow)", min_value=30, max_value=70, value=55, step=1, key="mfi_entry")
            rvol_min = st.number_input("RVOL ≥ (volume)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, format="%.1f", key="rvol_min")
        with c3:
            st.markdown("**Trigger**")
            scorecard_min = st.slider("Score ≥ (out of 3)", min_value=1, max_value=3, value=2, key="scorecard_min")
            st.caption("Need at least this many of RSI/MFI/RVOL to trigger BUY (with Core true).")

    with st.expander("Adjust sell rules", expanded=False):
        st.caption("Turn exit conditions on/off and edit parameters. Order: SMA20 → Trailing → Profit take → PDI<MDI → Stop loss → Month-end.")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Exit toggles**")
            sell_use_sma20 = st.checkbox("Sell when Close < SMA20", value=True, key="sell_sma20")
            sell_use_pdi_mdi = st.checkbox("Sell when PDI < MDI", value=True, key="sell_pdi_mdi")
            sell_use_stop_loss = st.checkbox("Sell at stop loss %", value=True, key="sell_stop")
            sell_use_trailing = st.checkbox("Sell on trailing stop (Smart Exit)", value=True, key="sell_trail")
            sell_use_profit_take = st.checkbox("Sell 50% on RSI climax (Smart Exit)", value=True, key="sell_pt")
            sell_use_month_end = st.checkbox("Force close at month-end", value=True, key="sell_me")
        with s2:
            st.markdown("**Exit parameters**")
            stop_loss_pct = st.number_input("Stop loss %", min_value=1, max_value=20, value=8, step=1, key="sl_pct") / 100.0
            atr_trail_mult = st.number_input("Trailing stop (× ATR)", min_value=1.0, max_value=6.0, value=3.0, step=0.5, format="%.1f", key="atr_mult")
            rsi_profit_taking = st.number_input("Profit take when RSI >", min_value=65, max_value=85, value=75, step=1, key="rsi_pt")

    if st.button("Run backtest", type="primary", key="backtest_btn"):
        symbol = (symbol or "").strip().upper()
        if not symbol:
            st.warning("Enter a ticker.")
        else:
            with st.spinner(f"Fetching {symbol} and running backtest..."):
                try:
                    # Apply user criteria to the backtest module before running
                    _bt.CORE_REQUIRE_TREND = core_require_trend
                    _bt.CORE_REQUIRE_PDI_MDI = core_require_pdi_mdi
                    _bt.ADX_MIN = int(adx_min)
                    _bt.ADX_MAX = int(adx_max)
                    _bt.RSI_ENTRY = int(rsi_entry)
                    _bt.MFI_ENTRY = int(mfi_entry)
                    _bt.RVOL_MIN = float(rvol_min)
                    _bt.SCORECARD_MIN = int(scorecard_min)
                    _bt.SELL_USE_SMA20 = sell_use_sma20
                    _bt.SELL_USE_PDI_MDI = sell_use_pdi_mdi
                    _bt.SELL_USE_STOP_LOSS = sell_use_stop_loss
                    _bt.SELL_USE_TRAILING = sell_use_trailing
                    _bt.SELL_USE_PROFIT_TAKE = sell_use_profit_take
                    _bt.SELL_USE_MONTH_END = sell_use_month_end
                    _bt.STOP_LOSS_PCT = stop_loss_pct
                    _bt.ATR_TRAIL_MULT = atr_trail_mult
                    _bt.RSI_PROFIT_TAKING = rsi_profit_taking

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
