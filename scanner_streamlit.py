#!/usr/bin/env python3
"""
==========================================================================
  Veteran v4.0 — Streamlit Scanner (HK & US)
==========================================================================
  Same logic as daily_scanner.py: Core + Score 2/3 (RSI>50, MFI>55, RVOL>=1.0)
  Run:   streamlit run scanner_streamlit.py
  Deps:  pip install streamlit yfinance pandas ta
==========================================================================
"""

import time
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

st.title("📊 Veteran v4.0 Scanner")
st.caption("Core: Close>SMA20, 20<ADX<50, PDI>MDI  |  Score 2/3: RSI>50, MFI>55, RVOL≥1.0")

# --- Sidebar: ticker source ---
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

# --- Main: Run scan ---
if tickers:
    if st.button("Run scan", type="primary"):
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
            import pandas as pd
            df = pd.DataFrame(results)
            # Reorder and show Signal first
            cols = ["Ticker", "Price", "Signal", "Why", "ADX", "RSI", "RVOL"]
            df = df[[c for c in cols if c in df.columns]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No actionable signals today. Stay cash.")

    st.sidebar.divider()
    st.sidebar.metric("Tickers to scan", len(tickers))
else:
    st.info("Select a ticker source in the sidebar and (for Custom) enter at least one ticker.")
