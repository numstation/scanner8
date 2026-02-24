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
except ImportError as e:
    print(f"[ERROR] Missing module: {e}")
    print("Install with: pip install yfinance pandas ta")
    sys.exit(1)

try:
    import schedule
except ImportError:
    schedule = None  # Optional: only needed for scheduler mode

# --- 1. TICKER LISTS ---
# Normalize: stocklist.txt uses 5-digit codes (e.g. 09988.HK); we use 4-digit (9988.HK)
def _norm_code(raw: str) -> str:
    """Strip leading 0 from XXXXX.HK -> XXXX.HK"""
    s = raw.strip().upper()
    if not s.endswith(".HK"):
        return s
    prefix = s[:-3]
    if len(prefix) == 5 and prefix.startswith("0"):
        return prefix[1:] + ".HK"
    return s

# Tech stocks (from stocklist.txt)
TECH_TICKERS = [
    _norm_code(c) for c in
    ["00020.HK", "00241.HK", "00268.HK", "00285.HK", "00300.HK", "00700.HK", "00780.HK",
     "00981.HK", "00992.HK", "01024.HK", "01211.HK", "01347.HK", "01698.HK", "01810.HK",
     "02015.HK", "02382.HK", "03690.HK", "03888.HK", "06618.HK", "06690.HK", "09618.HK",
     "09626.HK", "09660.HK", "09863.HK", "09866.HK", "09868.HK", "09888.HK", "09961.HK",
     "09988.HK", "09999.HK"]
]

# HSI (Hang Seng Index) constituents
HSI_TICKERS = [
    _norm_code(c) for c in
    ["00001.HK", "00002.HK", "00003.HK", "00005.HK", "00006.HK", "00012.HK", "00016.HK",
     "00027.HK", "00066.HK", "00101.HK", "00241.HK", "00285.HK", "00288.HK", "00300.HK",
     "00316.HK", "00322.HK", "00388.HK", "00669.HK", "00728.HK", "00823.HK", "00836.HK",
     "00868.HK", "00881.HK", "00960.HK", "00968.HK", "01038.HK", "01044.HK", "01099.HK",
     "01113.HK", "01177.HK", "01209.HK", "01299.HK", "01876.HK", "01928.HK", "01929.HK",
     "01997.HK", "02269.HK", "02331.HK", "02359.HK", "02388.HK", "02618.HK", "02688.HK",
     "03692.HK", "06862.HK", "09901.HK"]
]

# HKCEI (Hang Seng China Enterprises Index)
HKCEI_TICKERS = [
    _norm_code(c) for c in
    ["00175.HK", "00267.HK", "00291.HK", "00386.HK", "00688.HK", "00700.HK", "00762.HK",
     "00857.HK", "00883.HK", "00939.HK", "00941.HK", "00981.HK", "00992.HK", "01024.HK",
     "01088.HK", "01093.HK", "01109.HK", "01211.HK", "01288.HK", "01378.HK", "01398.HK",
     "01658.HK", "01801.HK", "01810.HK", "02015.HK", "02020.HK", "02057.HK", "02313.HK",
     "02318.HK", "02319.HK", "02328.HK", "02382.HK", "02628.HK", "02899.HK", "03328.HK",
     "03690.HK", "03968.HK", "03988.HK", "06160.HK", "06618.HK", "06690.HK", "09618.HK",
     "09633.HK", "09868.HK", "09888.HK", "09961.HK", "09987.HK", "09988.HK", "09992.HK",
     "09999.HK"]
]

# Default: Tech list when no args
DEFAULT_TICKERS = TECH_TICKERS.copy()

# HK = all three lists combined, deduplicated (for CLI: python daily_scanner.py HK)
HK_TICKERS = list(dict.fromkeys(TECH_TICKERS + HSI_TICKERS + HKCEI_TICKERS))

US_TICKERS = [
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD",
    "PLTR", "COIN", "MSTR", "SMCI", "AVGO", "COST", "NFLX", "JPM",
    "INTC", "WMT", "HOOD", "APP", "SNDK", "LITE", "CRWV", "BKNG",
    "MNDY", "BIDU", "BABA", "FUTU", "INTU", "SHOP", "PEP", "TXN",
    "GS", "IBM", "JNJ", "V", "JPM", "KO", "MRK", "NKE",
]


def get_tickers(market: str) -> list:
    """Return ticker list for market: Tech, HSI, HKCEI, HK (all), or US."""
    if market == "TECH":
        print(f" Loaded {len(TECH_TICKERS)} Tech tickers.")
        return TECH_TICKERS.copy()
    if market == "HSI":
        print(f" Loaded {len(HSI_TICKERS)} HSI tickers.")
        return HSI_TICKERS.copy()
    if market == "HKCEI":
        print(f" Loaded {len(HKCEI_TICKERS)} HKCEI tickers.")
        return HKCEI_TICKERS.copy()
    if market == "HK":
        print(f" Loaded {len(HK_TICKERS)} HK tickers (Tech+HSI+HKCEI).")
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
        close_curr = curr["Close"]
        open_curr = curr["Open"]
        sma20_curr = curr["SMA20"]

        if core_pass and score >= 2:
            signal = f"BUY ({score}/3)"
        elif close_curr > sma20_curr and curr["RSI"] > 75 and close_curr < open_curr:
            signal = "PROFIT TAKE"
            details = ["RSI>75", "Bearish"]
        elif close_curr < sma20_curr:
            signal = "SELL (Trend Break)"
            details = ["Close<SMA20"]
        elif pdi < mdi:
            signal = "SELL (Momentum Flip)"
            details = ["PDI<MDI"]

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
            sig = r["Signal"]
            icon = "🟢" if "BUY" in sig else ("🔴" if "SELL" in sig else "🟠")
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
        # Single arg: use that list
        if len(args) == 1:
            a = args[0].upper()
            if a in ("HK", "TECH", "HSI", "HKCEI", "US"):
                run_scan(a)
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
