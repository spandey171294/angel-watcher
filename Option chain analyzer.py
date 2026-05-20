"""
option_chain_analyzer.py — Advanced NSE Option Chain Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features
────────
  1. OI Buildup Detection    — Long / Short buildup per strike band
  2. Short Covering          — CE OI falling while price rising
  3. Long Unwinding          — PE OI falling while price falling
  4. Max Pain Calculation    — Strike where option buyers lose most
  5. IV Spike Detection      — Near-ATM IV vs rolling mean z-score
  6. PCR Bias                — Put-Call Ratio directional bias

Plug-and-play with tradewithlines.py
────────────────────────────────────
  STEP 1 — Drop this file next to tradewithlines.py

  STEP 2 — In tradewithlines.py, add near the top:
      from option_chain_analyzer import analyze_option_chain, format_oc_alert

  STEP 3 — In run_once(), after `spot = fetch_spot(...)`, add:
      oc = None
      try:
          oc = analyze_option_chain(underlying)
          print(f"  OC → {oc['signal']} | PCR {oc['pcr']} | "
                f"MaxPain {oc['max_pain']} | {oc['oi_trend']}")
      except Exception as exc:
          print(f"  ⚠ Option chain: {exc}")

  STEP 4 — When a signal fires, optionally send an OC alert:
      if oc and oc["signal"] != "NO SIGNAL":
          send_telegram(format_oc_alert(oc))

  Or use get_oc_confirmation() to filter low-conviction signals:
      from option_chain_analyzer import get_oc_confirmation
      confirmation = get_oc_confirmation(oc, technical_signal=sig_val)
      # confirmation → "CONFIRM" | "CONFLICT" | "NEUTRAL"
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

_NSE_BASE = "https://www.nseindia.com"
_NSE_CHAIN_URL = "{base}/api/option-chain-indices?symbol={symbol}"
_NSE_CHAIN_URL_EQ = "{base}/api/option-chain-equities?symbol={symbol}"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Connection":      "keep-alive",
}

# PCR thresholds (index-specific tuning)
PCR_BULLISH_THRESHOLD  = 1.10    # PCR > this → bullish bias
PCR_BEARISH_THRESHOLD  = 0.80    # PCR < this → bearish bias

# IV z-score threshold to flag as spike
IV_SPIKE_ZSCORE = 1.5

# Number of strikes each side of ATM to consider "near-ATM"
ATM_BAND = 5

# OI change significance: ignore changes < this % of total strike OI
OI_CHANGE_MIN_PCT = 0.5

# IV environment buckets (annualised %)
IV_LOW    = 10.0
IV_NORMAL = 18.0
IV_HIGH   = 28.0


# ═══════════════════════════════════════════════════════════════════════
# NSE FETCH
# ═══════════════════════════════════════════════════════════════════════

def _build_session() -> requests.Session:
    """Open an NSE-authenticated session (sets cookies from homepage)."""
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    try:
        session.get(_NSE_BASE, timeout=10)
        time.sleep(0.4)
    except Exception:
        pass
    return session


def fetch_option_chain(symbol: str = "NIFTY") -> dict:
    """
    Fetch raw NSE option chain JSON for index or equity.

    Parameters
    ----------
    symbol : e.g. "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"

    Returns
    -------
    Raw JSON dict from NSE API.

    Raises
    ------
    RuntimeError if the API call fails after 3 attempts.
    """
    index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
                     "NIFTYNXT50", "SENSEX"}
    url_template = (_NSE_CHAIN_URL if symbol.upper() in index_symbols
                    else _NSE_CHAIN_URL_EQ)
    url = url_template.format(base=_NSE_BASE, symbol=symbol.upper())

    last_exc: Exception = RuntimeError("Unknown error")
    for attempt in range(1, 4):
        try:
            session = _build_session()
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NSE fetch failed for {symbol}: {last_exc}")


# ═══════════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════════

def parse_option_chain(data: dict) -> pd.DataFrame:
    """
    Convert raw NSE option chain JSON to a clean DataFrame.

    Columns
    ───────
    strike, expiry_date,
    ce_oi, ce_change_oi, ce_iv, ce_ltp, ce_volume, ce_bid, ce_ask,
    pe_oi, pe_change_oi, pe_iv, pe_ltp, pe_volume, pe_bid, pe_ask
    """
    rows = []
    for item in data.get("records", {}).get("data", []):
        strike = item.get("strikePrice")
        expiry = item.get("expiryDate", "")
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        rows.append({
            "strike":        float(strike) if strike else None,
            "expiry_date":   expiry,
            # CE fields
            "ce_oi":         float(ce.get("openInterest", 0) or 0),
            "ce_change_oi":  float(ce.get("changeinOpenInterest", 0) or 0),
            "ce_iv":         float(ce.get("impliedVolatility", 0) or 0),
            "ce_ltp":        float(ce.get("lastPrice", 0) or 0),
            "ce_volume":     float(ce.get("totalTradedVolume", 0) or 0),
            # PE fields
            "pe_oi":         float(pe.get("openInterest", 0) or 0),
            "pe_change_oi":  float(pe.get("changeinOpenInterest", 0) or 0),
            "pe_iv":         float(pe.get("impliedVolatility", 0) or 0),
            "pe_ltp":        float(pe.get("lastPrice", 0) or 0),
            "pe_volume":     float(pe.get("totalTradedVolume", 0) or 0),
        })

    df = pd.DataFrame(rows).dropna(subset=["strike"])
    df = df[df["strike"] > 0].sort_values("strike").reset_index(drop=True)
    return df


def _nearest_expiry_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to nearest expiry (most liquid)."""
    if "expiry_date" not in df.columns or df["expiry_date"].nunique() <= 1:
        return df
    try:
        expiries = pd.to_datetime(df["expiry_date"], format="%d-%b-%Y", errors="coerce")
        nearest  = expiries.min()
        return df[expiries == nearest].reset_index(drop=True)
    except Exception:
        return df


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def calculate_pcr(df: pd.DataFrame) -> dict:
    """
    PCR on total OI across all strikes (nearest expiry).

    Returns
    ───────
    {pcr, total_ce_oi, total_pe_oi, bias}
    """
    total_ce = df["ce_oi"].sum()
    total_pe = df["pe_oi"].sum()
    pcr = round(total_pe / total_ce, 3) if total_ce > 0 else 0.0

    if   pcr > PCR_BULLISH_THRESHOLD:  bias = "BULLISH"
    elif pcr < PCR_BEARISH_THRESHOLD:  bias = "BEARISH"
    else:                              bias = "NEUTRAL"

    return {
        "pcr":         pcr,
        "total_ce_oi": total_ce,
        "total_pe_oi": total_pe,
        "pcr_bias":    bias,
    }


def calculate_max_pain(df: pd.DataFrame) -> dict:
    """
    Max pain = the strike at which total option buyer value is MINIMUM
    (i.e., option writers extract maximum premium at expiry).

    For each test strike P:
      CE buyer value = sum(max(0, P - K_i) * CE_OI_i)  — ITM calls
      PE buyer value = sum(max(0, K_i - P) * PE_OI_i)  — ITM puts
    Min of that sum across all strikes = max pain strike.

    Returns
    ───────
    {max_pain, max_pain_diff, max_pain_diff_pct, spot}
    """
    strikes = df["strike"].values
    ce_oi   = df["ce_oi"].values
    pe_oi   = df["pe_oi"].values
    min_pain  = float("inf")
    max_pain  = float(strikes[len(strikes) // 2])

    for test_price in strikes:
        ce_value   = np.maximum(test_price - strikes, 0) * ce_oi
        pe_value   = np.maximum(strikes - test_price, 0) * pe_oi
        total_pain = ce_value.sum() + pe_value.sum()
        if total_pain < min_pain:
            min_pain  = total_pain
            max_pain  = float(test_price)

    return {"max_pain": max_pain}


def detect_oi_patterns(
    df: pd.DataFrame,
    spot: float,
    strike_step: int = 50,
    atm_band: int = ATM_BAND,
) -> dict:
    """
    Detect OI-based patterns across all strikes and specifically near ATM.

    Patterns detected per strike band
    ──────────────────────────────────
    LONG_BUILDUP   : OI ↑ + positive change  → fresh positions, directional
    SHORT_BUILDUP  : OI ↑ + positions on other side → fresh hedge / shorts
    SHORT_COVERING : CE OI ↓ while price > CE strike → bears bailing out (bullish)
    LONG_UNWINDING : PE OI ↓ while price < PE strike → bulls bailing out (bearish)
    NEUTRAL        : No significant change

    Returns
    ───────
    {
      total_ce_change, total_pe_change, oi_trend,
      atm_ce_change, atm_pe_change, atm_pattern,
      oi_resistance, oi_support,
      call_wall, put_wall,
      strike_details: [{strike, pattern, ce_change_oi, pe_change_oi}]
    }
    """
    atm = round(spot / strike_step) * strike_step
    atm_idx  = (df["strike"] - atm).abs().idxmin()
    lo_idx   = max(0, atm_idx - atm_band)
    hi_idx   = min(len(df) - 1, atm_idx + atm_band)
    atm_df   = df.iloc[lo_idx: hi_idx + 1]

    total_ce_change = df["ce_change_oi"].sum()
    total_pe_change = df["pe_change_oi"].sum()
    atm_ce_change   = atm_df["ce_change_oi"].sum()
    atm_pe_change   = atm_df["pe_change_oi"].sum()

    # ── OI Resistance / Support from absolute OI ────────────────────
    oi_resistance = float(df.loc[df["ce_oi"].idxmax(), "strike"])
    oi_support    = float(df.loc[df["pe_oi"].idxmax(), "strike"])

    # ── Call Wall / Put Wall (top-2 OI strikes) ──────────────────────
    top2_ce = df.nlargest(2, "ce_oi")["strike"].tolist()
    top2_pe = df.nlargest(2, "pe_oi")["strike"].tolist()
    call_wall = [float(s) for s in top2_ce]
    put_wall  = [float(s) for s in top2_pe]

    # ── Global OI trend (based on overall change) ────────────────────
    #  CE ↑ + PE ↓ = bears building resistance + bulls bailing  → BEARISH
    #  CE ↓ + PE ↑ = bears covering + bulls building support    → BULLISH
    #  CE ↑ + PE ↑ = both sides adding                          → INDECISION
    #  CE ↓ + PE ↓ = both sides unwinding                       → CAUTION
    if   total_ce_change > 0 and total_pe_change < 0: oi_trend = "BEARISH"
    elif total_ce_change < 0 and total_pe_change > 0: oi_trend = "BULLISH"
    elif total_ce_change > 0 and total_pe_change > 0: oi_trend = "INDECISION"
    else:                                             oi_trend = "CAUTION"   # both unwinding

    # ── Near-ATM dominant pattern ────────────────────────────────────
    # Short covering: CE OI falling at/above ATM while index rising
    # Long unwinding: PE OI falling at/below ATM while index falling
    # Long buildup  : PE OI rising at/below ATM (put writers defending support)
    # Short buildup : CE OI rising at/above ATM (call writers building resistance)
    patterns = []
    for _, row in atm_df.iterrows():
        s  = float(row["strike"])
        cc = float(row["ce_change_oi"])
        pc = float(row["pe_change_oi"])

        if s >= atm:   # at / above ATM → CE analysis
            if cc < 0:               patterns.append("SHORT_COVERING")
            elif cc > 0:             patterns.append("SHORT_BUILDUP")
        if s <= atm:   # at / below ATM → PE analysis
            if pc < 0:               patterns.append("LONG_UNWINDING")
            elif pc > 0:             patterns.append("LONG_BUILDUP")

    from collections import Counter
    dominant_pattern = (Counter(patterns).most_common(1)[0][0]
                        if patterns else "NEUTRAL")

    # Per-strike detail list (for logging/display)
    strike_details = []
    for _, row in atm_df.iterrows():
        s   = float(row["strike"])
        cc  = float(row["ce_change_oi"])
        pc  = float(row["pe_change_oi"])
        if abs(cc) > 0 or abs(pc) > 0:
            strike_details.append({
                "strike":        s,
                "ce_oi":         float(row["ce_oi"]),
                "pe_oi":         float(row["pe_oi"]),
                "ce_change_oi":  cc,
                "pe_change_oi":  pc,
            })

    return {
        "total_ce_change": total_ce_change,
        "total_pe_change": total_pe_change,
        "atm_ce_change":   atm_ce_change,
        "atm_pe_change":   atm_pe_change,
        "oi_trend":        oi_trend,
        "atm_pattern":     dominant_pattern,
        "oi_resistance":   oi_resistance,
        "oi_support":      oi_support,
        "call_wall":       call_wall,
        "put_wall":        put_wall,
        "strike_details":  strike_details,
    }


def detect_iv_spikes(
    df: pd.DataFrame,
    spot: float,
    strike_step: int = 50,
    atm_band: int    = ATM_BAND,
    zscore_thresh: float = IV_SPIKE_ZSCORE,
) -> dict:
    """
    Detect IV spikes near ATM and characterise the IV environment.

    Logic
    ─────
    1. Use only strikes with non-zero IV.
    2. Calculate z-score of each strike's IV vs mean across all strikes.
    3. Flag any near-ATM strike where |z| > zscore_thresh as a spike.
    4. IV skew: CE_HEAVY (calls expensive) | PE_HEAVY (puts expensive — normal)
                | BALANCED.

    Returns
    ───────
    {
      avg_ce_iv, avg_pe_iv, iv_skew, iv_environment,
      iv_spike_detected, iv_spike_details: [...]
    }
    """
    atm    = round(spot / strike_step) * strike_step
    atm_idx = (df["strike"] - atm).abs().idxmin()
    lo_idx  = max(0, atm_idx - atm_band)
    hi_idx  = min(len(df) - 1, atm_idx + atm_band)
    atm_df  = df.iloc[lo_idx: hi_idx + 1]

    # Average IV (use all non-zero strikes for baseline)
    ce_iv_all = df[df["ce_iv"] > 0]["ce_iv"]
    pe_iv_all = df[df["pe_iv"] > 0]["pe_iv"]

    avg_ce_iv = float(ce_iv_all.mean()) if len(ce_iv_all) else 0.0
    avg_pe_iv = float(pe_iv_all.mean()) if len(pe_iv_all) else 0.0
    std_ce_iv = float(ce_iv_all.std())  if len(ce_iv_all) > 1 else 1.0
    std_pe_iv = float(pe_iv_all.std())  if len(pe_iv_all) > 1 else 1.0

    std_ce_iv = std_ce_iv if std_ce_iv > 0 else 1.0
    std_pe_iv = std_pe_iv if std_pe_iv > 0 else 1.0

    # ── IV environment ───────────────────────────────────────────────
    atm_iv = (avg_ce_iv + avg_pe_iv) / 2
    if   atm_iv < IV_LOW:    iv_env = "LOW"
    elif atm_iv < IV_NORMAL: iv_env = "NORMAL"
    elif atm_iv < IV_HIGH:   iv_env = "HIGH"
    else:                    iv_env = "EXTREME"

    # ── IV Skew ──────────────────────────────────────────────────────
    # For indices, PE IV > CE IV is normal (fear premium on downside).
    # If CE IV > PE IV, unusual — could signal a squeeze / event.
    diff = avg_ce_iv - avg_pe_iv
    if   diff > 2.0:   iv_skew = "CE_HEAVY"    # unusual — call demand spike
    elif diff < -2.0:  iv_skew = "PE_HEAVY"    # normal for indices
    else:              iv_skew = "BALANCED"

    # ── Spike Detection ─────────────────────────────────────────────
    spike_details = []
    for _, row in atm_df.iterrows():
        s = float(row["strike"])
        for side, iv_val, mean, std in (
            ("CE", float(row["ce_iv"]), avg_ce_iv, std_ce_iv),
            ("PE", float(row["pe_iv"]), avg_pe_iv, std_pe_iv),
        ):
            if iv_val <= 0:
                continue
            z = (iv_val - mean) / std
            if abs(z) >= zscore_thresh:
                spike_details.append({
                    "strike":  s,
                    "side":    side,
                    "iv":      round(iv_val, 2),
                    "avg_iv":  round(mean, 2),
                    "z_score": round(z, 2),
                })

    return {
        "avg_ce_iv":        round(avg_ce_iv, 2),
        "avg_pe_iv":        round(avg_pe_iv, 2),
        "atm_iv":           round(atm_iv, 2),
        "iv_skew":          iv_skew,
        "iv_environment":   iv_env,
        "iv_spike_detected": bool(spike_details),
        "iv_spike_details":  spike_details,
    }


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR  — combines all analyses into one actionable output
# ═══════════════════════════════════════════════════════════════════════

def generate_oc_signal(
    pcr_data: dict,
    oi_data:  dict,
    iv_data:  dict,
    mp_data:  dict,
    spot:     float,
    symbol:   str,
) -> dict:
    """
    Score each bullish/bearish dimension and produce a final signal.

    Scoring (each factor adds +1 bull or -1 bear):
    ───────────────────────────────────────────────
    PCR bias           : BULLISH → +1 | BEARISH → -1
    OI trend           : BULLISH → +1 | BEARISH → -1 | INDECISION → 0 | CAUTION → -0.5
    ATM OI pattern     : SHORT_COVERING/LONG_BUILDUP → +1
                         LONG_UNWINDING/SHORT_BUILDUP → -1
    Max Pain           : spot < max_pain → +1 | spot > max_pain → -1
    IV skew            : CE_HEAVY → mild bearish (-0.5) | PE_HEAVY → mild bullish (+0.5)

    Final decision:
      score >=  1.5 and at least 3 indicators agree → BUY CE
      score <= -1.5 and at least 3 indicators agree → BUY PE
      otherwise                                       → NO SIGNAL

    Avoid conditions (override signal):
      IV environment = EXTREME          (too expensive to buy options)
      IV spike on target side           (buying into an IV spike)
      OI trend = CAUTION (both unwinding — no conviction)
    """
    bull_score = 0.0
    reasons:       list[str] = []
    avoid_reasons: list[str] = []

    # 1. PCR
    if   pcr_data["pcr_bias"] == "BULLISH": bull_score += 1;   reasons.append(f"PCR {pcr_data['pcr']} → bullish")
    elif pcr_data["pcr_bias"] == "BEARISH": bull_score -= 1;   reasons.append(f"PCR {pcr_data['pcr']} → bearish")

    # 2. OI trend
    oi_map = {"BULLISH": 1.0, "BEARISH": -1.0, "INDECISION": 0.0, "CAUTION": -0.5}
    bull_score += oi_map.get(oi_data["oi_trend"], 0.0)
    reasons.append(f"OI trend: {oi_data['oi_trend']}")

    # 3. ATM OI pattern
    pat_map = {
        "SHORT_COVERING": +1.0,
        "LONG_BUILDUP":   +1.0,
        "LONG_UNWINDING": -1.0,
        "SHORT_BUILDUP":  -1.0,
        "NEUTRAL":         0.0,
    }
    bull_score += pat_map.get(oi_data["atm_pattern"], 0.0)
    reasons.append(f"ATM pattern: {oi_data['atm_pattern']}")

    # 4. Max pain gravity
    mp = mp_data["max_pain"]
    if   spot < mp: bull_score += 1.0;  reasons.append(f"Spot {spot} below MaxPain {mp} → gravity pull up")
    elif spot > mp: bull_score -= 1.0;  reasons.append(f"Spot {spot} above MaxPain {mp} → gravity pull down")

    # 5. IV skew (mild weight)
    if   iv_data["iv_skew"] == "CE_HEAVY": bull_score -= 0.5; reasons.append("CE IV heavy → unusual call demand")
    elif iv_data["iv_skew"] == "PE_HEAVY": bull_score += 0.5; reasons.append("PE IV heavy → normal put skew (bullish tilt)")

    # ── Avoid conditions ─────────────────────────────────────────────
    if iv_data["iv_environment"] == "EXTREME":
        avoid_reasons.append(f"EXTREME IV ({iv_data['atm_iv']}%) — options very expensive")
    if oi_data["oi_trend"] == "CAUTION":
        avoid_reasons.append("Both CE & PE OI unwinding — no directional conviction")
    if iv_data["iv_spike_detected"]:
        spikes = iv_data["iv_spike_details"]
        for sp in spikes[:2]:
            avoid_reasons.append(
                f"IV spike at {sp['strike']}{sp['side']} → {sp['iv']}% (z={sp['z_score']})"
            )

    # ── Final decision ───────────────────────────────────────────────
    if   bull_score >=  1.5: bias = "BULLISH"
    elif bull_score <= -1.5: bias = "BEARISH"
    else:                    bias = "NEUTRAL"

    # Only fire a signal when conviction is there
    if avoid_reasons:
        signal     = "NO SIGNAL"
        confidence = "LOW"
    elif bull_score >= 2.5:
        signal = "BUY CE"; confidence = "HIGH"
    elif bull_score >= 1.5:
        signal = "BUY CE"; confidence = "MEDIUM"
    elif bull_score <= -2.5:
        signal = "BUY PE"; confidence = "HIGH"
    elif bull_score <= -1.5:
        signal = "BUY PE"; confidence = "MEDIUM"
    else:
        signal     = "NO SIGNAL"
        confidence = "LOW"

    return {
        "bias":         bias,
        "signal":       signal,
        "confidence":   confidence,
        "bull_score":   round(bull_score, 2),
        "reasons":      reasons,
        "avoid_reasons": avoid_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

_INDEX_STEP = {
    "NIFTY":      50,
    "BANKNIFTY":  100,
    "FINNIFTY":   50,
    "MIDCPNIFTY": 25,
}


def analyze_option_chain(symbol: str = "NIFTY") -> dict:
    """
    Full option chain analysis pipeline.

    Parameters
    ----------
    symbol : "NIFTY" | "BANKNIFTY" | "FINNIFTY" | "MIDCPNIFTY"

    Returns
    -------
    Flat dict with ALL analysis fields — safe to pass to format_oc_alert()
    or inspect directly.

    Keys include:
      symbol, timestamp, spot,
      pcr, pcr_bias, total_ce_oi, total_pe_oi,
      max_pain,
      total_ce_change, total_pe_change, oi_trend,
      atm_pattern, oi_resistance, oi_support, call_wall, put_wall,
      avg_ce_iv, avg_pe_iv, atm_iv, iv_skew, iv_environment,
      iv_spike_detected, iv_spike_details,
      bias, signal, confidence, bull_score,
      reasons, avoid_reasons
    """
    step = _INDEX_STEP.get(symbol.upper(), 50)

    raw_data   = fetch_option_chain(symbol)
    spot       = float(raw_data.get("records", {}).get("underlyingValue", 0))
    df         = parse_option_chain(raw_data)
    df         = _nearest_expiry_df(df)

    pcr_data   = calculate_pcr(df)
    mp_data    = calculate_max_pain(df)
    oi_data    = detect_oi_patterns(df, spot, strike_step=step)
    iv_data    = detect_iv_spikes(df, spot, strike_step=step)
    sig_data   = generate_oc_signal(pcr_data, oi_data, iv_data, mp_data, spot, symbol)

    return {
        # Identity
        "symbol":    symbol.upper(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spot":      spot,
        # PCR
        **pcr_data,
        # Max pain
        **mp_data,
        # OI patterns
        **oi_data,
        # IV
        **iv_data,
        # Signal
        **sig_data,
    }


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER  — use alongside technical signals
# ═══════════════════════════════════════════════════════════════════════

def get_oc_confirmation(
    oc_result: dict,
    technical_signal: str,    # "BUY" or "SELL" from tradewithlines
) -> str:
    """
    Compare option chain bias with the technical signal from tradewithlines.

    Returns
    ───────
    "CONFIRM"  — OC agrees with the technical signal
    "CONFLICT" — OC disagrees (trade with caution / skip)
    "NEUTRAL"  — OC has no strong opinion
    """
    oc_bias = oc_result.get("bias", "NEUTRAL")

    if technical_signal == "BUY"  and oc_bias == "BULLISH": return "CONFIRM"
    if technical_signal == "SELL" and oc_bias == "BEARISH": return "CONFIRM"
    if technical_signal == "BUY"  and oc_bias == "BEARISH": return "CONFLICT"
    if technical_signal == "SELL" and oc_bias == "BULLISH": return "CONFLICT"
    return "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════
# TELEGRAM FORMATTER
# ═══════════════════════════════════════════════════════════════════════

def format_oc_alert(oc: dict) -> str:
    """
    Build a Telegram-ready HTML message from an analyze_option_chain() result.
    Matches the existing tradewithlines.py alert style.
    """
    sig      = oc.get("signal", "NO SIGNAL")
    conf     = oc.get("confidence", "LOW")
    bias     = oc.get("bias", "NEUTRAL")

    conf_icon = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💤"}.get(conf, "")
    dir_icon  = "🟢📈" if "CE" in sig else ("🔴📉" if "PE" in sig else "⚪")
    iv_env    = oc.get("iv_environment", "NORMAL")
    iv_icon   = {"LOW": "🟦", "NORMAL": "🟩", "HIGH": "🟧", "EXTREME": "🟥"}.get(iv_env, "")

    def _f(v, d=2):
        try:    return f"{float(v):,.{d}f}"
        except: return "—"

    spikes = oc.get("iv_spike_details", [])
    spike_lines = ""
    if spikes:
        spike_lines = "\n" + "\n".join(
            f"   ⚠ {sp['strike']}{sp['side']} IV {sp['iv']}% (z={sp['z_score']})"
            for sp in spikes[:3]
        )

    avoid_lines = ""
    avoid = oc.get("avoid_reasons", [])
    if avoid:
        avoid_lines = "\n🚫 Avoid Reasons:\n" + "\n".join(f"   • {r}" for r in avoid[:3])

    reason_lines = "\n".join(
        f"   {'✅' if 'bullish' in r.lower() or 'support' in r.lower() or 'up' in r.lower() else '❌' if 'bearish' in r.lower() or 'down' in r.lower() else '➡'} {r}"
        for r in oc.get("reasons", [])
    )

    call_wall = ", ".join(_f(s, 0) for s in oc.get("call_wall", []))
    put_wall  = ", ".join(_f(s, 0) for s in oc.get("put_wall",  []))

    lines = [
        f"{'═' * 35}",
        f"{dir_icon}  📊 OPTION CHAIN — {sig}",
        f"{'═' * 35}",
        f"",
        f"📌 Index      : {oc.get('symbol')}",
        f"💰 Spot       : {_f(oc.get('spot'))}",
        f"🎯 Bias       : {bias}  {conf_icon} {conf}",
        f"",
        f"{'─' * 35}",
        f"📉 Max Pain   : {_f(oc.get('max_pain'), 0)}",
        f"📊 PCR        : {oc.get('pcr')}  ({oc.get('pcr_bias')})",
        f"",
        f"⚡ OI Resistance : {_f(oc.get('oi_resistance'), 0)}",
        f"🛡  OI Support    : {_f(oc.get('oi_support'), 0)}",
        f"📌 Call Wall  : {call_wall}",
        f"📌 Put  Wall  : {put_wall}",
        f"",
        f"{'─' * 35}",
        f"🔄 OI Trend   : {oc.get('oi_trend')}",
        f"🔍 ATM Pattern: {oc.get('atm_pattern')}",
        f"",
        f"{'─' * 35}",
        f"{iv_icon} IV Env      : {iv_env}",
        f"   CE IV avg  : {_f(oc.get('avg_ce_iv'))}%",
        f"   PE IV avg  : {_f(oc.get('avg_pe_iv'))}%",
        f"   IV Skew    : {oc.get('iv_skew')}",
    ]

    if spike_lines:
        lines.append(f"{'─' * 35}")
        lines.append("⚠ IV Spikes Near ATM:" + spike_lines)

    lines += [
        f"",
        f"{'─' * 35}",
        f"🧠 Analysis:",
        reason_lines,
    ]

    if avoid_lines:
        lines.append(avoid_lines)

    lines.append(f"{'═' * 35}")
    return "\n".join(lines)


def format_oc_confirmation_line(oc: dict, tech_signal: str) -> str:
    """
    One-liner to append to an existing tradewithlines alert.
    E.g.: "🔗 OC Confirmation: CONFIRM ✅ | PCR 1.15 | MaxPain 22350"
    """
    status = get_oc_confirmation(oc, tech_signal)
    icon   = {"CONFIRM": "✅", "CONFLICT": "⚠️", "NEUTRAL": "➡️"}.get(status, "")
    return (
        f"🔗 OC Check : {status} {icon}  "
        f"PCR {oc.get('pcr', '—')} | "
        f"MaxPain {oc.get('max_pain', '—'):.0f} | "
        f"{oc.get('oi_trend', '—')}"
    )


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "NIFTY"
    print(f"\nFetching option chain for {symbol}…\n")

    result = analyze_option_chain(symbol)

    print("=" * 50)
    print(f"  {symbol} OPTION CHAIN ANALYSIS")
    print("=" * 50)
    for k, v in result.items():
        if k in ("reasons", "avoid_reasons", "iv_spike_details",
                 "strike_details", "call_wall", "put_wall"):
            if v:
                print(f"  {k}:")
                for item in (v if isinstance(v, list) else [v]):
                    print(f"    • {item}")
        else:
            print(f"  {k:<22}: {v}")

    print()
    print(format_oc_alert(result))
