#!/usr/bin/env python3
"""
==========================================================================
  Veteran v4.0 — Automated Daily Scanner (HK & US)
==========================================================================
  Logic: Core (Close>SMA20, 20<ADX<50, PDI>MDI) + Score 2/3 (RSI>50, MFI>55, RVOL>=1.0)
  Run:   python daily_scanner.py                    # start scheduler (HK 17:00, US 08:30)
         python daily_scanner.py HK                 # run HK list once
         python daily_scanner.py US                 # run US list once
         python daily_scanner.py 0700.HK 9988.HK    # scan these tickers only (custom)
  Deps:  pip install yfinance pandas ta schedule
==========================================================================
"""

import sys
import time
from datetime import datetime

try:
    import yfinance as yf
    import pandas as pd
    from ta.momentum import RSIIndicator
    from ta.trend import ADXIndicator, SMAIndicator
    from ta.volume import MFIIndicator
    import schedule
except ImportError as e:
    print(f"[ERROR] Missing module: {e}")
    print("Install with: pip install yfinance pandas ta schedule")
    print("Or use a venv: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
    sys.exit(1)

# --- 1. TICKER LISTS ---
# Default list when no args (used if you run with custom logic elsewhere)
DEFAULT_TICKERS = [
    "9988.HK", "0700.HK", "0883.HK", "0005.HK", "9992.HK", "9626.HK",
    "9999.HK", "0027.HK", "1772.HK", "9888.HK", "1810.HK", "1211.HK",
]

HK_TICKERS = [
    "9988.HK", "0700.HK", "1211.HK", "0005.HK", "9992.HK", "9626.HK",
    "9999.HK", "0027.HK", "1772.HK", "9888.HK", "1810.HK", "3750.HK",
    "2318.HK", "0388.HK", "0941.HK", "0001.HK", "0016.HK", "0823.HK",
    "3416.HK", "1299.HK", "1024.HK", "0003.HK", "1928.HK", "2020.HK",
    "0939.HK", "1398.HK", "6618.HK", "0001.HK", "0016.HK", "3690.HK",
]

US_TICKERS = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD",
    "PLTR", "COIN", "MSTR", "SMCI", "AVGO", "COST", "NFLX", "JPM",
    "INTC", "WMT", "HOOD", "APP", "SNDK", "LITE", "CRWV", "BKNG",
    "MNDY", "BIDU", "BABA", "FUTU", "INTU", "SHOP", "PEP", "TXN",
    "GS", "IBM", "JNJ", "V", "JPM", "KO", "MRK", "NKE",
]


def get_tickers(market: str) -> list:
    """Return ticker list for market (HK or US)."""
    if market == "HK":
        print(f" Loaded {len(HK_TICKERS)} HK tickers.")
        return HK_TICKERS.copy()
    if market == "US":
        print(f" Loaded {len(US_TICKERS)} US tickers.")
        return US_TICKERS.copy()
    return []


# --- 2. VETERAN v4.0 SIGNAL (Core + Score 2/3) ---
def _fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    """Fetch OHLCV using same method as backtest (Ticker.history) to avoid yfinance download quirks."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False)
        if df is None or len(df) < 50:
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        need = ["Open", "High", "Low", "Close", "Volume"]
        for col in need:
            if col not in df.columns:
                return None
        return df[need].astype(float).sort_index()
    except Exception:
        return None


def analyze_stock(ticker: str) -> dict | None:
    """
    One stock: download 6mo, compute indicators with ta library, apply Veteran v4.0.
    Returns a result dict if there is an actionable signal, else None.
    """
    try:
        df = _fetch_ohlcv(ticker)
        if df is None or len(df) < 50:
            return None

        h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]

        # Indicators using ta library (no pandas_ta / numba)
        df["SMA20"] = SMAIndicator(close=c, window=20).sma_indicator()
        df["RSI"] = RSIIndicator(close=c, window=14).rsi()
        df["MFI"] = MFIIndicator(high=h, low=l, close=c, volume=v, window=14).money_flow_index()
        adx_ind = ADXIndicator(high=h, low=l, close=c, window=14)
        df["ADX"] = adx_ind.adx()
        df["PDI"] = adx_ind.adx_pos()   # +DI
        df["MDI"] = adx_ind.adx_neg()   # -DI
        vol_sma = v.rolling(window=20, min_periods=1).mean().replace(0, float("nan"))
        df["RVOL"] = v / vol_sma

        curr = df.iloc[-1]
        adx = curr["ADX"]
        pdi = curr["PDI"]
        mdi = curr["MDI"]
        if pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi):
            return None

        # CORE (all required)
        core_pass = (
            curr["Close"] > curr["SMA20"]
            and 20 < adx < 50
            and pdi > mdi
        )

        # SCORECARD (2 of 3)
        score = 0
        details = []
        if curr["RSI"] > 50:
            score += 1
            details.append("RSI")
        if pd.notna(curr.get("MFI")) and curr["MFI"] > 55:
            score += 1
            details.append("MFI")
        if pd.notna(curr.get("RVOL")) and curr["RVOL"] >= 1.0:
            score += 1
            details.append("VOL")

        signal = None
        if core_pass and score >= 2:
            signal = f"BUY ({score}/3)"
        elif curr["Close"] > curr["SMA20"] and curr["RSI"] > 75:
            signal = "PROFIT TAKE"
            details = ["Overheated"]

        if signal:
            rvol_str = f"{curr['RVOL']:.2f}" if pd.notna(curr.get("RVOL")) else "—"
            return {
                "Ticker": ticker,
                "Price": f"{curr['Close']:.2f}",
                "Signal": signal,
                "Why": ",".join(details),
                "ADX": f"{adx:.1f}",
                "RSI": f"{curr['RSI']:.1f}",
                "RVOL": rvol_str,
            }
    except Exception:
        return None
    return None


# --- 3. SCANNER ENGINE ---
def _run_scan_with_tickers(tickers: list, label: str) -> None:
    """Run scan over a given list of tickers and print results."""
    print(f"\n  Veteran v4.0 — {label} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 85)
    if not tickers:
        print(" No tickers to scan.")
        print("=" * 85)
        return
    print(" Scanning...", end="", flush=True)
    results = []
    for i, t in enumerate(tickers):
        res = analyze_stock(t)
        if res:
            results.append(res)
        print(".", end="", flush=True)
        if i < len(tickers) - 1:
            time.sleep(0.25)
    print(" done.\n" + "=" * 85)
    if results:
        print(f" {'Ticker':<10} {'Price':<10} {'Signal':<20} {'Why':<15} {'ADX':<6} {'RSI':<6} {'RVOL':<6}")
        print("-" * 85)
        for r in results:
            icon = "🟢" if "BUY" in r["Signal"] else "🟠"
            print(f"{icon} {r['Ticker']:<8} {r['Price']:<10} {r['Signal']:<20} {r['Why']:<15} {r['ADX']:<6} {r['RSI']:<6} {r['RVOL']:<6}")
    else:
        print(" No actionable signals today. Stay cash.")
    print("=" * 85)


def run_scan(market: str) -> None:
    """Run full scan for a market (HK or US) and print results."""
    print(f" Fetching {market} tickers...", end="", flush=True)
    tickers = get_tickers(market)
    _run_scan_with_tickers(tickers, market)


# --- 4. SCHEDULER ---
def job_hk() -> None:
    print(" Waking for HK scan...")
    run_scan("HK")


def job_us() -> None:
    print(" Waking for US scan...")
    run_scan("US")


def main() -> None:
    # sys.argv[0] = script name; [1:] = user args
    if len(sys.argv) > 1:
        args = [a.strip() for a in sys.argv[1:] if a.strip()]
        # Single arg "HK" or "US" → use that market's list
        if len(args) == 1 and args[0].upper() == "HK":
            run_scan("HK")
            return
        if len(args) == 1 and args[0].upper() == "US":
            run_scan("US")
            return
        # Otherwise treat all args as custom tickers
        tickers = args
        print(f" Custom scan requested: {tickers}")
        _run_scan_with_tickers(tickers, "Custom")
        return

    # No args: use default list and run once
    print(f" Default scan mode: {len(DEFAULT_TICKERS)} stocks")
    _run_scan_with_tickers(DEFAULT_TICKERS.copy(), "Default")
    return

    # Uncomment below to run scheduler when no args (and comment out the two lines above)
    # schedule.every().day.at("17:00").do(job_hk)
    # schedule.every().day.at("08:30").do(job_us)
    # print(" Veteran Scanner ONLINE. Waiting for schedule (17:00 HK, 08:30 US). Ctrl+C to stop.")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)


if __name__ == "__main__":
    main()
