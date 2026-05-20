"""
tradewithlines.py — Angel One SmartAPI multi-timeframe index scanner.

Scans NSE index candles across multiple timeframes and fires Telegram alerts
for all three signal types from the TradeswithLines TradingView indicator:

  • Supertrend  (BUY / SELL crossover)
  • Diamond     (Ichimoku breakout — ◆ diamond shapes)
  • QQE Reversal(RSI-based QQE — ▲ triangle shapes)

On a BUY signal  → recommends buying the ATM CE of that index.
On a SELL signal → recommends buying the ATM PE of that index.

Includes nearest Support/Resistance and TP/SL levels in each alert.

Option Chain Confirmation (via option_chain_analyzer.py)
─────────────────────────────────────────────────────────
Each scan cycle also fetches the live NSE option chain and checks:
  • PCR bias            — Put/Call ratio direction
  • OI trend            — CE/PE change OI balance
  • ATM OI pattern      — Short covering / Long buildup / etc.
  • Max Pain            — Gravitational pull on index
  • IV spikes           — Unusual IV at near-ATM strikes
  • IV skew             — CE vs PE implied vol difference

The result is appended to every technical alert as a confirmation line.
When --oc-alert is passed, a full standalone OC alert is also sent when
the option chain has a HIGH/MEDIUM confidence signal of its own.
When --oc-confirm-only is passed, technical signals that CONFLICT with
the option chain are suppressed (not sent to Telegram).

Usage
─────
python tradewithlines.py \
    --api-key      <KEY> \
    --client-code  <CLIENT> \
    --mpin         <MPIN> \
    --totp-secret  <TOTP_SECRET> \
    --underlyings  NIFTY,BANKNIFTY \
    --intervals    ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE \
    --polling-seconds 60 \
    --oc-alert \
    --oc-confirm-only

Environment variables required for Telegram:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pyotp
import requests
from SmartApi import SmartConnect

from indicators import generate_all_signals
from telegram_notifier import send_telegram
from option_chain_analyzer import (
    analyze_option_chain,
    format_oc_alert,
    format_oc_confirmation_line,
    get_oc_confirmation,
)


# ═══════════════════════════════════════════════════════════════════════
# INDEX CONFIG
# ═══════════════════════════════════════════════════════════════════════

INDEX_MAP: dict[str, dict] = {
    "NIFTY": {
        "exchange":      "NSE",
        "tradingsymbol": "Nifty 50",
        "symboltoken":   "99926000",
        "strike_step":   50,
    },
    "BANKNIFTY": {
        "exchange":      "NSE",
        "tradingsymbol": "Nifty Bank",
        "symboltoken":   "99926009",
        "strike_step":   100,
    },
    "FINNIFTY": {
        "exchange":      "NSE",
        "tradingsymbol": "Nifty Fin Service",
        "symboltoken":   "99926037",
        "strike_step":   50,
    },
    "MIDCPNIFTY": {
        "exchange":      "NSE",
        "tradingsymbol": "NIFTY MID SELECT",
        "symboltoken":   "99926074",
        "strike_step":   25,
    },
}

# How many calendar days of history to request per interval.
# More days → more candles → better pivot S/R detection.
INTERVAL_DAYS: dict[str, int] = {
    "ONE_MINUTE":     5,
    "THREE_MINUTE":   5,
    "FIVE_MINUTE":    7,
    "TEN_MINUTE":    10,
    "FIFTEEN_MINUTE":15,
    "THIRTY_MINUTE": 20,
    "ONE_HOUR":      30,
    "ONE_DAY":      365,
}

VALID_UNDERLYINGS = list(INDEX_MAP.keys())
VALID_INTERVALS   = list(INTERVAL_DAYS.keys())


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    api_key:                  str
    client_code:              str
    mpin:                     str
    totp_secret:              str
    underlyings:              list[str]
    intervals:                list[str]
    option_strikes_each_side: int
    polling_seconds:          int
    expiry:                   Optional[str]
    sens:                     float = 2.0       # Supertrend factor
    skip_mkt_hours:           bool  = False
    oc_alert:                 bool  = False     # send standalone OC alert when OC fires HIGH/MEDIUM signal
    oc_confirm_only:          bool  = False     # suppress technical signals that CONFLICT with OC


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL CACHE  — prevents duplicate alerts for the same bar
# ═══════════════════════════════════════════════════════════════════════

# key = "UNDERLYING|interval|SignalName"  →  (last_signal_value, bar_time)
LAST_SIGNALS: dict[str, tuple[str, str]] = {}


# ═══════════════════════════════════════════════════════════════════════
# ANGEL ONE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def login(cfg: Config) -> SmartConnect:
    client = SmartConnect(api_key=cfg.api_key)
    totp   = pyotp.TOTP(cfg.totp_secret).now()
    sess   = client.generateSession(cfg.client_code, cfg.mpin, totp)
    if not sess.get("status"):
        raise RuntimeError(f"Login failed: {sess}")
    return client


def load_master() -> list:
    url = ("https://margincalculator.angelbroking.com/"
           "OpenAPI_File/files/OpenAPIScripMaster.json")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_spot(client: SmartConnect, underlying: str) -> float:
    idx = INDEX_MAP[underlying]
    ltp = client.ltpData(idx["exchange"], idx["tradingsymbol"], idx["symboltoken"])
    if not ltp.get("status"):
        raise RuntimeError(f"LTP fetch failed: {ltp}")
    return float(ltp["data"]["ltp"])


def get_nearest_expiry(master: list, underlying: str) -> str:
    expiries = {
        row["expiry"]
        for row in master
        if (row.get("exch_seg") == "NFO"
            and row.get("instrumenttype") == "OPTIDX"
            and row.get("name") == underlying
            and row.get("expiry"))
    }
    if not expiries:
        raise RuntimeError(f"No expiry found for {underlying}")
    return sorted(expiries, key=lambda x: datetime.strptime(x, "%d%b%Y"))[0]


def get_atm_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def get_option_info(
    master: list,
    underlying: str,
    expiry: str,
    strike: int,
    opt_type: str,    # "CE" or "PE"
) -> tuple[Optional[str], Optional[str]]:
    """Return (token, symbol) for the specified option, or (None, None)."""
    for row in master:
        if (row.get("exch_seg") == "NFO"
                and row.get("instrumenttype") == "OPTIDX"
                and row.get("name") == underlying
                and row.get("expiry") == expiry
                and abs(float(row.get("strike", 0)) / 100 - strike) < 0.1
                and row["symbol"].endswith(opt_type)):
            return row["token"], row["symbol"]
    return None, None


def fetch_option_ltp(
    client: SmartConnect,
    token: Optional[str],
    symbol: Optional[str],
) -> Optional[float]:
    if not token or not symbol:
        return None
    try:
        ltp = client.ltpData("NFO", symbol, token)
        if ltp.get("status"):
            return round(float(ltp["data"]["ltp"]), 2)
    except Exception:
        pass
    return None


def fetch_candles(
    client: SmartConnect,
    exchange: str,
    token: str,
    interval: str,
    days: int = 5,
) -> Optional[pd.DataFrame]:
    to_dt   = dt.datetime.now()
    from_dt = to_dt - dt.timedelta(days=days)
    params  = {
        "exchange":    exchange,
        "symboltoken": token,
        "interval":    interval,
        "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
    }
    res = client.getCandleData(params)
    if not res.get("status") or not res.get("data"):
        return None
    df = pd.DataFrame(
        res["data"],
        columns=["time", "open", "high", "low", "close", "volume"],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# TELEGRAM ALERT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _fmt(value, decimals=2) -> str:
    return f"{value:,.{decimals}f}" if value is not None else "—"


def build_alert(
    underlying: str,
    interval: str,
    sig_name: str,
    direction: str,
    price: float,
    atm: int,
    opt_type: str,
    opt_symbol: Optional[str],
    opt_price: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
    tp_sl: Optional[dict],
    oc_line: Optional[str] = None,    # one-line OC confirmation appended at bottom
) -> str:
    bull   = direction == "BUY"
    arrow  = "🟢📈" if bull else "🔴📉"
    act    = f"BUY {atm}{opt_type} ({underlying}{''.join(opt_symbol[-6:] if opt_symbol else '')})"
    tf_str = interval.replace("_", " ")

    sig_icons = {
        "Supertrend":   "📊",
        "Diamond":      "◆",
        "QQE Reversal": "🔺" if bull else "🔻",
    }
    icon = sig_icons.get(sig_name, "⚡")

    lines = [
        f"{'═'*35}",
        f"{arrow}  {icon} {sig_name.upper()} — {direction}",
        f"{'═'*35}",
        f"",
        f"📌 Index      : {underlying}",
        f"⏱  Timeframe  : {tf_str}",
        f"💰 Index LTP  : {_fmt(price)}",
    ]

    if support or resistance:
        lines.append("")
        if resistance:
            lines.append(f"⚡ Resistance : {_fmt(resistance)}")
        if support:
            lines.append(f"🛡  Support    : {_fmt(support)}")

    if tp_sl:
        lines += [
            "",
            f"🎯 Entry      : {_fmt(price)}",
            f"🛑 Stop Loss  : {_fmt(tp_sl['sl'])}",
            f"✅ TP 1       : {_fmt(tp_sl['tp1'])}",
            f"✅ TP 2       : {_fmt(tp_sl['tp2'])}",
            f"✅ TP 3       : {_fmt(tp_sl['tp3'])}",
        ]

    lines += [
        "",
        f"{'─'*35}",
        f"📢 Option Suggestion",
        f"   Action  : {act}",
        f"   Strike  : {atm}  |  Type: {opt_type}",
    ]
    if opt_price:
        lines.append(f"   Option LTP : {_fmt(opt_price)}")

    # ── Option chain confirmation line (if available) ────────────────
    if oc_line:
        lines += ["", f"{'─'*35}", oc_line]

    lines.append(f"{'═'*35}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# MARKET HOURS CHECK
# ═══════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """True during NSE cash-market hours: Mon–Fri 09:15–15:30 IST."""
    try:
        import pytz
        now     = datetime.now(pytz.timezone("Asia/Kolkata"))
        if now.weekday() >= 5:
            return False
        open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return open_t <= now <= close_t
    except ImportError:
        return True   # pytz not installed — assume open


# ═══════════════════════════════════════════════════════════════════════
# OPTION CHAIN CACHE  — re-use OC result across all intervals per cycle
# ═══════════════════════════════════════════════════════════════════════

# key = underlying  →  OC result dict from last successful fetch
OC_CACHE: dict[str, dict] = {}

# key = underlying  →  last bar_time when standalone OC alert was sent
OC_ALERT_SENT: dict[str, str] = {}


# ═══════════════════════════════════════════════════════════════════════
# MAIN SCAN LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_once(
    cfg: Config,
    client: SmartConnect,
    master: list,
    expiry_cache: dict,
) -> None:

    for underlying in cfg.underlyings:
        idx  = INDEX_MAP[underlying]
        step = idx["strike_step"]

        try:
            spot = fetch_spot(client, underlying)
        except Exception as exc:
            print(f"  ⚠  Could not fetch spot for {underlying}: {exc}")
            continue

        atm = get_atm_strike(spot, step)

        # Cache nearest expiry (re-use within the session)
        if underlying not in expiry_cache:
            expiry_cache[underlying] = (
                cfg.expiry if cfg.expiry
                else get_nearest_expiry(master, underlying)
            )
        expiry = expiry_cache[underlying]

        # ── Option chain analysis (once per underlying per cycle) ────
        oc = None
        try:
            oc = analyze_option_chain(underlying)
            OC_CACHE[underlying] = oc
            oc_summary = (
                f"OC→ {oc['signal']:<8} conf={oc['confidence']:<6} "
                f"PCR={oc['pcr']:<5} MaxPain={oc['max_pain']:.0f} "
                f"OI={oc['oi_trend']}"
            )
            iv_warn = "  ⚠ IV SPIKE" if oc.get("iv_spike_detected") else ""
            print(f"  📊 {oc_summary}{iv_warn}")
        except Exception as exc:
            # Non-fatal: fall back to last cached result if available
            oc = OC_CACHE.get(underlying)
            print(f"  ⚠  OC fetch failed ({exc})"
                  + ("  — using cached" if oc else "  — no cache"))

        # If OC has its own standalone signal (HIGH/MEDIUM confidence)
        # and --oc-alert was passed, send a dedicated OC alert once per signal.
        if cfg.oc_alert and oc and oc.get("signal") not in ("NO SIGNAL", None):
            oc_key = f"{underlying}|OC|{oc['signal']}|{oc.get('oi_trend')}"
            if OC_ALERT_SENT.get(underlying) != oc_key:
                OC_ALERT_SENT[underlying] = oc_key
                send_telegram(format_oc_alert(oc))
                print(f"  📤  OC STANDALONE ALERT → {oc['signal']} ({oc['confidence']})")

        print(f"\n{'═'*72}")
        print(f"  {underlying:<12}  SPOT {spot:>10,.2f}  |  ATM {atm}  |  EXPIRY {expiry}")
        print(f"{'═'*72}")
        print(f"  {'Timeframe':<18}  {'Price':<10}  {'ST':<6}  {'DM':<6}  {'QQE':<6}  "
              f"{'Support':<10}  {'Resist':<10}  {'OC':<10}")
        print(f"  {'─'*78}")

        for interval in cfg.intervals:
            days = INTERVAL_DAYS.get(interval, 10)
            df   = fetch_candles(client, idx["exchange"], idx["symboltoken"], interval, days)
            time.sleep(0.4)     # gentle rate-limit between calls

            if df is None:
                print(f"  {interval:<18}  No data")
                continue
            if len(df) < 80:
                print(f"  {interval:<18}  Only {len(df)} bars (need ≥80)")
                continue

            try:
                result = generate_all_signals(df, sens=cfg.sens)
            except ValueError as exc:
                print(f"  {interval:<18}  {exc}")
                continue
            except Exception as exc:
                print(f"  {interval:<18}  Signal error: {exc}")
                continue

            price = result["price"]
            sup   = result["support"]
            res   = result["resistance"]
            bt    = result["bar_time"]

            oc_bias_str = oc.get("bias", "—") if oc else "—"
            print(
                f"  {interval:<18}  {price:<10,.2f}  "
                f"{result['st_signal']:<6}  {result['diamond_signal']:<6}  {result['qqe_signal']:<6}  "
                f"{_fmt(sup):<10}  {_fmt(res):<10}  {oc_bias_str:<10}"
            )

            # ─── check all three signal types ───────────────────────────
            for sig_name, sig_val in (
                ("Supertrend",   result["st_signal"]),
                ("Diamond",      result["diamond_signal"]),
                ("QQE Reversal", result["qqe_signal"]),
            ):
                if sig_val not in ("BUY", "SELL"):
                    continue

                cache_key = f"{underlying}|{interval}|{sig_name}"
                last_val, last_bt = LAST_SIGNALS.get(cache_key, (None, None))

                # Skip if we already alerted for this exact bar
                if last_val == sig_val and last_bt == bt:
                    continue

                LAST_SIGNALS[cache_key] = (sig_val, bt)

                # ── OC confirmation check ────────────────────────────
                oc_status = "NEUTRAL"
                oc_line   = None
                if oc:
                    oc_status = get_oc_confirmation(oc, sig_val)
                    oc_line   = format_oc_confirmation_line(oc, sig_val)

                    if cfg.oc_confirm_only and oc_status == "CONFLICT":
                        print(
                            f"    ⛔  SUPPRESSED ({sig_name} {sig_val}) — "
                            f"OC CONFLICT (bias={oc.get('bias')}, "
                            f"avoid={oc.get('avoid_reasons', [])})"
                        )
                        continue   # skip this signal — OC disagrees

                # Fetch option details
                opt_type              = "CE" if sig_val == "BUY" else "PE"
                opt_token, opt_symbol = get_option_info(
                    master, underlying, expiry, atm, opt_type
                )
                opt_price = fetch_option_ltp(client, opt_token, opt_symbol)

                msg = build_alert(
                    underlying=underlying,
                    interval=interval,
                    sig_name=sig_name,
                    direction=sig_val,
                    price=price,
                    atm=atm,
                    opt_type=opt_type,
                    opt_symbol=opt_symbol,
                    opt_price=opt_price,
                    support=sup,
                    resistance=res,
                    tp_sl=result.get("tp_sl"),
                    oc_line=oc_line,
                )

                send_telegram(msg)
                conf_tag = f"OC:{oc_status}" if oc else "OC:N/A"
                print(
                    f"    🚨  ALERT SENT → {sig_name} {sig_val}  "
                    f"{underlying}{atm}{opt_type}  OPT LTP={opt_price}  [{conf_tag}]"
                )


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradeswithLines — Angel One multi-timeframe index scanner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--api-key",      required=True, help="Angel One API key")
    parser.add_argument("--client-code",  required=True, help="Angel One client code")
    parser.add_argument("--mpin",         required=True, help="Angel One M-PIN")
    parser.add_argument("--totp-secret",  required=True, help="TOTP secret (base32)")

    parser.add_argument(
        "--underlyings",
        default="NIFTY",
        help=(
            "Comma-separated list of indices to scan.\n"
            f"Choices: {', '.join(VALID_UNDERLYINGS)}\n"
            "Default: NIFTY"
        ),
    )
    parser.add_argument(
        "--intervals",
        default="FIVE_MINUTE",
        help=(
            "Comma-separated list of candle intervals.\n"
            f"Choices: {', '.join(VALID_INTERVALS)}\n"
            "Default: FIVE_MINUTE\n"
            "Example: ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE"
        ),
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help="Override expiry date (format: 24APR2025). Defaults to nearest weekly.",
    )
    parser.add_argument(
        "--option-strikes-each-side",
        type=int, default=1,
        help="How many strikes each side of ATM to consider (currently informational).",
    )
    parser.add_argument(
        "--polling-seconds",
        type=int, default=60,
        help="Seconds between scans. Default: 60.",
    )
    parser.add_argument(
        "--sens",
        type=float, default=2.0,
        help="Supertrend sensitivity (factor). Default: 2.0 (matches Pine Script).",
    )
    parser.add_argument(
        "--skip-market-hours-check",
        action="store_true",
        help="Run even outside 09:15–15:30 IST. Useful for back-testing/testing.",
    )
    parser.add_argument(
        "--oc-alert",
        action="store_true",
        help=(
            "Send a standalone Option Chain alert to Telegram when the OC\n"
            "analysis has a HIGH or MEDIUM confidence signal of its own.\n"
            "These are independent of the technical (Supertrend/Diamond/QQE) alerts."
        ),
    )
    parser.add_argument(
        "--oc-confirm-only",
        action="store_true",
        help=(
            "Suppress any technical signal that CONFLICTS with the option chain bias.\n"
            "Example: if OC says BEARISH but Supertrend fires BUY, the BUY is skipped.\n"
            "Reduces noise at the cost of some true signals. Recommended with --oc-alert."
        ),
    )

    args = parser.parse_args()

    underlyings = [u.strip().upper() for u in args.underlyings.split(",")]
    intervals   = [i.strip().upper() for i in args.intervals.split(",")]

    for u in underlyings:
        if u not in VALID_UNDERLYINGS:
            parser.error(f"Unknown underlying '{u}'. Choose from: {VALID_UNDERLYINGS}")
    for i in intervals:
        if i not in VALID_INTERVALS:
            parser.error(f"Unknown interval '{i}'. Choose from: {VALID_INTERVALS}")

    cfg = Config(
        api_key=args.api_key,
        client_code=args.client_code,
        mpin=args.mpin,
        totp_secret=args.totp_secret,
        underlyings=underlyings,
        intervals=intervals,
        option_strikes_each_side=args.option_strikes_each_side,
        polling_seconds=args.polling_seconds,
        expiry=args.expiry,
        sens=args.sens,
        skip_mkt_hours=args.skip_market_hours_check,
        oc_alert=args.oc_alert,
        oc_confirm_only=args.oc_confirm_only,
    )

    oc_mode = (
        "confirm+alert" if (cfg.oc_confirm_only and cfg.oc_alert) else
        "confirm-only"  if cfg.oc_confirm_only else
        "alert-only"    if cfg.oc_alert        else
        "passive (append only)"
    )
    print("╔" + "═" * 58 + "╗")
    print("║       TradeswithLines — Angel One Index Scanner          ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Underlyings  : {', '.join(cfg.underlyings):<40} ║")
    print(f"║  Intervals    : {', '.join(cfg.intervals[:3]):<40} ║")
    if len(cfg.intervals) > 3:
        print(f"║               : {', '.join(cfg.intervals[3:]):<40} ║")
    print(f"║  Polling      : every {cfg.polling_seconds}s{'':<33} ║")
    print(f"║  Sensitivity  : {cfg.sens:<41} ║")
    print(f"║  OC Mode      : {oc_mode:<40} ║")
    print("╚" + "═" * 58 + "╝")

    print("\nLogging in to Angel One…")
    client = login(cfg)
    print("✅ Login successful")

    print("Loading master contract file…")
    master = load_master()
    print(f"✅ Master loaded ({len(master):,} records)")

    expiry_cache: dict = {}
    cycle = 0

    while True:
        cycle += 1
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Cycle {cycle}  {now_str}]")

        if not cfg.skip_mkt_hours and not is_market_open():
            print("⏸  Market closed — waiting for next cycle…")
        else:
            try:
                run_once(cfg, client, master, expiry_cache)
            except Exception as exc:
                print(f"\n⚠  ERROR: {exc}")
                # Re-login on session expiry
                try:
                    client = login(cfg)
                    print("✅ Re-login successful")
                except Exception as re_exc:
                    print(f"❌ Re-login failed: {re_exc}")

        print(f"\n⏳ Sleeping {cfg.polling_seconds}s…")
        time.sleep(cfg.polling_seconds)


if __name__ == "__main__":
    main()
