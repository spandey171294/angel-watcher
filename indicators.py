"""
indicators.py — TradeswithLines Pine Script → Python port.

Faithfully reproduces every signal-generating component from the
TradeswithLines TradingView indicator (Pine Script v6):

  1. Supertrend         → st_signal       (BUY / SELL / HOLD)
     • Wilder ATR, band-clamping exactly as in Pine
  2. Diamond Signals    → diamond_signal  (BUY / SELL / HOLD)
     • Ichimoku lead-line crossover (enrev block in Pine)
  3. QQE Reversal       → qqe_signal      (BUY / SELL / HOLD)
     • Triangle up/down signals (enrevser block in Pine)
  4. Pivot S/R          → 8 price levels
     • left=50, right=25, quick_right=5
  5. ATR-based TP / SL  → sl, tp1, tp2, tp3
     • Matches Pine TP & SL block (atrRisk=4, r1=0.7, r2=1.2, r3=1.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA (= EMA with alpha=1/period). Matches Pine ta.rma / ta.atr."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EMA. Matches Pine ta.ema."""
    return series.ewm(span=span, adjust=False).mean()


def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR via Wilder smoothing — matches Pine ta.atr(period)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low,
         (high - close.shift(1)).abs(),
         (low  - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return _rma(tr, period)


# ═══════════════════════════════════════════════════════════════════════
# 1. SUPERTREND  (Pine Script: supertrend(_src, factor, atrLen))
# ═══════════════════════════════════════════════════════════════════════

def compute_supertrend(
    df: pd.DataFrame,
    factor: float = 2.0,
    atr_len: int = 11,
) -> pd.DataFrame:
    """
    Ports the Pine Script supertrend() function including:
      • Band clamping (lowerBand never drops, upperBand never rises)
      • Exact direction flip logic

    Adds columns
    ───────────
    supertrend   : float — the supertrend line value
    st_direction : int   — -1 = bullish (close above ST), 1 = bearish
    st_bull      : bool  — crossover  (bearish → bullish flip)
    st_bear      : bool  — crossunder (bullish → bearish flip)
    """
    atr  = _wilder_atr(df, atr_len)
    hl2  = (df["high"] + df["low"]) / 2.0

    upper_raw = (hl2 + factor * atr).values.astype(float)
    lower_raw = (hl2 - factor * atr).values.astype(float)
    close_arr = df["close"].values.astype(float)
    n = len(df)

    upper     = upper_raw.copy()
    lower     = lower_raw.copy()
    direction = np.ones(n, dtype=int)          # 1 = bearish, -1 = bullish
    st        = np.full(n, np.nan, dtype=float)

    for i in range(1, n):
        # ── handle NaN from early ATR bars ──
        if np.isnan(upper[i]):
            upper[i] = upper[i - 1] if not np.isnan(upper[i - 1]) else upper_raw[i]
        if np.isnan(lower[i]):
            lower[i] = lower[i - 1] if not np.isnan(lower[i - 1]) else lower_raw[i]

        # ── Pine band-clamping ──
        lower[i] = (lower[i]
                    if (lower[i] > lower[i - 1] or close_arr[i - 1] < lower[i - 1])
                    else lower[i - 1])
        upper[i] = (upper[i]
                    if (upper[i] < upper[i - 1] or close_arr[i - 1] > upper[i - 1])
                    else upper[i - 1])

        # ── direction logic (mirrors Pine else-if chain) ──
        prev_st = st[i - 1]
        if np.isnan(prev_st):
            direction[i] = 1
        elif prev_st == upper[i - 1]:           # was on upper → bearish
            direction[i] = -1 if close_arr[i] > upper[i] else 1
        else:                                   # was on lower → bullish
            direction[i] = 1 if close_arr[i] < lower[i] else -1

        st[i] = lower[i] if direction[i] == -1 else upper[i]

    df = df.copy()
    df["supertrend"]   = st
    df["st_direction"] = direction

    dir_s       = pd.Series(direction, index=df.index)
    df["st_bull"] = (dir_s == -1) & (dir_s.shift(1) == 1)   # bearish→bullish
    df["st_bear"] = (dir_s == 1)  & (dir_s.shift(1) == -1)  # bullish→bearish
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. DIAMOND SIGNALS  (Pine: enrev block — Ichimoku lead-line breakout)
# ═══════════════════════════════════════════════════════════════════════

def _donchian(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    return (high.rolling(length).max() + low.rolling(length).min()) / 2.0


def compute_diamond_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ports the Pine Script Diamond Signals section.

    Pine parameters used:
      conversionPeriods=5, basePeriods=2, laggingSpan2Periods=5, displacement=6

    Logic:
      lead1  = avg(donchian(5), donchian(2))  shifted 5 bars back
      lead2  = donchian(5)                    shifted 5 bars back
      breakup  when lead2>lead1, green candle, close crossover lead2
      breakdn  when lead2<lead1, red   candle, close crossunder lead2

    Adds columns: diamond_buy, diamond_sell (bool)
    """
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    disp = 6                              # displacement - 1 = 5 bars shift

    conv  = _donchian(high, low, 5)
    base  = _donchian(high, low, 2)
    lead1 = ((conv + base) / 2.0).shift(disp - 1)
    lead2 = _donchian(high, low, 5).shift(disp - 1)

    cross_up = (close > lead2) & (close.shift(1) <= lead2.shift(1))
    cross_dn = (close < lead2) & (close.shift(1) >= lead2.shift(1))

    df = df.copy()
    df["diamond_buy"]  = (lead2 > lead1) & (close > open_) & cross_up
    df["diamond_sell"] = (lead2 < lead1) & (close < open_) & cross_dn
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. QQE REVERSAL SIGNALS  (Pine: enrevser block — triangle shapes)
# ═══════════════════════════════════════════════════════════════════════

def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI using Wilder's smoothing (matches Pine ta.rsi)."""
    delta    = close.diff()
    avg_gain = _rma(delta.clip(lower=0), period)
    avg_loss = _rma((-delta).clip(lower=0), period)
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_qqe_signals(
    df: pd.DataFrame,
    rsi_period: int  = 14,
    sf: int          = 5,
    kqe: float       = 4.238,
) -> pd.DataFrame:
    """
    Quantitative Qualitative Estimation (QQE).
    Ports the Pine Script reversal-signals section exactly.

    Pine defaults: RSI_Period=14, SF=5, KQE=4.238

    Adds columns: qqe_long, qqe_short (bool — triangle buy/sell signals)
    """
    wilders  = rsi_period * 2 - 1
    rsi      = _wilder_rsi(df["close"], rsi_period)
    rsi_ma   = _ema(rsi, sf)                      # EMA of RSI

    atr_rsi    = rsi_ma.diff().abs()
    ma_atr_rsi = _rma(atr_rsi, wilders)
    dar        = _rma(ma_atr_rsi, wilders) * kqe  # dynamic ATR range

    n       = len(df)
    rma_arr = rsi_ma.values.astype(float)
    dar_arr = dar.values.astype(float)

    longband  = np.zeros(n, dtype=float)
    shortband = np.zeros(n, dtype=float)
    trend     = np.ones(n, dtype=int)
    fast_tl   = np.zeros(n, dtype=float)

    for i in range(1, n):
        if np.isnan(dar_arr[i]) or np.isnan(rma_arr[i]):
            longband[i]  = longband[i - 1]
            shortband[i] = shortband[i - 1]
            trend[i]     = trend[i - 1]
            fast_tl[i]   = fast_tl[i - 1]
            continue

        nl = rma_arr[i] - dar_arr[i]
        ns = rma_arr[i] + dar_arr[i]

        # Pine: clamp longband upward only when RSI stays above it
        longband[i] = (max(longband[i - 1], nl)
                       if (rma_arr[i - 1] > longband[i - 1]
                           and rma_arr[i] > longband[i - 1])
                       else nl)
        # Pine: clamp shortband downward only when RSI stays below it
        shortband[i] = (min(shortband[i - 1], ns)
                        if (rma_arr[i - 1] < shortband[i - 1]
                            and rma_arr[i] < shortband[i - 1])
                        else ns)

        # Pine: ta.cross(RSIndex, shortband[1]) → trend = 1 (bullish)
        #       ta.cross(longband[1], RSIndex)  → trend = -1 (bearish)
        if rma_arr[i] > shortband[i - 1]:
            trend[i] = 1
        elif rma_arr[i] < longband[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        fast_tl[i] = longband[i] if trend[i] == 1 else shortband[i]

    # Count consecutive bars: fast_tl below rsi_ma → bullish run (Exlong)
    exlong  = np.zeros(n, dtype=int)
    exshort = np.zeros(n, dtype=int)
    for i in range(1, n):
        exlong[i]  = exlong[i - 1]  + 1 if fast_tl[i] < rma_arr[i] else 0
        exshort[i] = exshort[i - 1] + 1 if fast_tl[i] > rma_arr[i] else 0

    # Signal fires on the FIRST bar of the new run (== 1)
    df = df.copy()
    df["qqe_long"]  = pd.Series(exlong,  index=df.index) == 1
    df["qqe_short"] = pd.Series(exshort, index=df.index) == 1
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. PIVOT-BASED SUPPORT & RESISTANCE  (8 levels, Pine S&R section)
# ═══════════════════════════════════════════════════════════════════════

def compute_pivot_sr(
    df: pd.DataFrame,
    left: int        = 50,
    right: int       = 25,
    quick_right: int = 5,
) -> dict:
    """
    Ports Pine Script S/R pivot logic (left=50, right=25, quick_right=5).

    Returns dict with keys level1–level8:
      level1/2  = most recent quick-pivot high/low
      level3/4  = most recent standard pivot high/low
      level5/6  = 2nd most recent standard pivot high/low
      level7/8  = 3rd most recent standard pivot high/low
    """
    n    = len(df)
    high = df["high"].values.astype(float)
    low  = df["low"].values.astype(float)

    ph_list, pl_list   = [], []   # standard pivot highs/lows
    qph_list, qpl_list = [], []   # quick pivot highs/lows

    # Standard pivots: need left bars before AND right bars after
    for i in range(left, n - right):
        h_win = high[i - left: i + right + 1]
        l_win = low[i  - left: i + right + 1]
        if high[i] >= h_win.max():
            ph_list.append(high[i])
        if low[i] <= l_win.min():
            pl_list.append(low[i])

    # Quick pivots
    for i in range(left, n - quick_right):
        h_win = high[i - left: i + quick_right + 1]
        l_win = low[i  - left: i + quick_right + 1]
        if high[i] >= h_win.max():
            qph_list.append(high[i])
        if low[i] <= l_win.min():
            qpl_list.append(low[i])

    def _get(lst: list, occ: int):
        return round(float(lst[-(occ + 1)]), 2) if len(lst) > occ else None

    return {
        "level1": _get(qph_list, 0),   # most-recent quick pivot high
        "level2": _get(qpl_list, 0),   # most-recent quick pivot low
        "level3": _get(ph_list,  0),   # most-recent standard pivot high
        "level4": _get(pl_list,  0),   # most-recent standard pivot low
        "level5": _get(ph_list,  1),   # 2nd standard pivot high
        "level6": _get(pl_list,  1),   # 2nd standard pivot low
        "level7": _get(ph_list,  2),   # 3rd standard pivot high
        "level8": _get(pl_list,  2),   # 3rd standard pivot low
    }


def nearest_sr(levels: dict, price: float) -> tuple:
    """Return (nearest_support_below, nearest_resistance_above) price."""
    vals  = [v for v in levels.values() if v is not None]
    above = [v for v in vals if v > price]
    below = [v for v in vals if v <= price]
    return (
        round(max(below), 2) if below else None,
        round(min(above), 2) if above else None,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. ATR-BASED TP / SL  (Pine: TP & SL block)
# ═══════════════════════════════════════════════════════════════════════

def compute_tp_sl(
    df: pd.DataFrame,
    signal: str,          # "BUY" or "SELL"
    entry: float,
    atr_len: int   = 14,
    atr_risk: int  = 4,
    r1: float      = 0.7,
    r2: float      = 1.2,
    r3: float      = 1.5,
) -> dict | None:
    """
    Computes stop-loss and three take-profit levels.
    Matches Pine Script TP & SL block exactly.

    BUY  → SL = low  of signal bar - ATR(14)*4
    SELL → SL = high of signal bar + ATR(14)*4
    TP1..3 = entry ± (entry-SL) * r1/r2/r3
    """
    if signal not in ("BUY", "SELL"):
        return None

    atr_val = float(_wilder_atr(df, atr_len).iloc[-1])
    last    = df.iloc[-1]

    if signal == "BUY":
        sl   = float(last["low"])  - atr_val * atr_risk
        diff = entry - sl
        tp1  = entry + diff * r1
        tp2  = entry + diff * r2
        tp3  = entry + diff * r3
    else:   # SELL
        sl   = float(last["high"]) + atr_val * atr_risk
        diff = sl - entry
        tp1  = entry - diff * r1
        tp2  = entry - diff * r2
        tp3  = entry - diff * r3

    return {
        "sl":  round(sl,  2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# MASTER SIGNAL GENERATOR  (runs all indicators, returns one clean dict)
# ═══════════════════════════════════════════════════════════════════════

def generate_all_signals(df: pd.DataFrame, sens: float = 2.0) -> dict:
    """
    Run all TradeswithLines indicators on a candle DataFrame.

    Parameters
    ----------
    df   : Must have columns: time, open, high, low, close, volume
    sens : Supertrend factor / sensitivity (default 2.0 matches Pine)

    Returns
    -------
    dict with keys:
      price, bar_time,
      st_signal, diamond_signal, qqe_signal,    ← "BUY" / "SELL" / "HOLD"
      support, resistance,                        ← nearest S/R levels
      all_levels,                                 ← all 8 pivot levels
      tp_sl,                                      ← dict or None
      st_direction                                ← -1 bull / 1 bear
    """
    if len(df) < 80:
        raise ValueError(f"Need ≥80 bars for reliable signals, got {len(df)}")

    df = compute_supertrend(df, factor=sens, atr_len=11)
    df = compute_diamond_signals(df)
    df = compute_qqe_signals(df)

    last  = df.iloc[-1]
    price = round(float(last["close"]), 2)
    bt    = str(last.get("time", ""))

    st_sig = "BUY"  if bool(last["st_bull"])     else ("SELL" if bool(last["st_bear"])    else "HOLD")
    dm_sig = "BUY"  if bool(last["diamond_buy"]) else ("SELL" if bool(last["diamond_sell"]) else "HOLD")
    qq_sig = "BUY"  if bool(last["qqe_long"])    else ("SELL" if bool(last["qqe_short"])  else "HOLD")

    # Dominant signal for TP/SL (first non-HOLD, or HOLD)
    dominant = next((s for s in (st_sig, dm_sig, qq_sig) if s != "HOLD"), "HOLD")
    tp_sl    = compute_tp_sl(df, dominant, price)

    levels          = compute_pivot_sr(df)
    support, resist = nearest_sr(levels, price)

    return {
        "price":          price,
        "bar_time":       bt,
        "st_signal":      st_sig,
        "diamond_signal": dm_sig,
        "qqe_signal":     qq_sig,
        "support":        support,
        "resistance":     resist,
        "all_levels":     levels,
        "tp_sl":          tp_sl,
        "st_direction":   int(last["st_direction"]),
    }
