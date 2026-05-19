# TradeswithLines — Angel One Automated Scanner

Automated Python port of the **TradeswithLines** TradingView indicator.
Scans NSE index option chains across multiple timeframes and sends
**Telegram alerts** whenever a signal fires.

---

## Signals ported from Pine Script

| Signal | Pine Script section | What it detects |
|---|---|---|
| **Supertrend** | `supertrend()` function | Price crossover/under of the ATR-based trend band |
| **Diamond ◆** | `enrev` / Diamond Signals block | Ichimoku lead-line breakout on green/red candles |
| **QQE Reversal ▲▼** | `enrevser` / Reversal Signals block | QQE (RSI-based) trend exhaustion flip |

Every alert also includes:
- Nearest **Support & Resistance** (8 pivot levels, same as chart)
- **Stop Loss + 3 Take-Profit levels** (ATR × Risk-Reward, same as chart)
- The **ATM CE or PE** symbol + live LTP to act on immediately

---

## Files

```
tradewithlines.py      ← Main scanner (Angel One API + alerts)
indicators.py          ← All indicator math (pure Python, no API)
telegram_notifier.py   ← Telegram sender
requirements.txt       ← pip dependencies
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Telegram credentials

Create a bot via [@BotFather](https://t.me/BotFather), then export:

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdef..."
export TELEGRAM_CHAT_ID="987654321"          # your chat or channel ID
```

To make these permanent, add the two lines to `~/.bashrc` or `~/.profile`.

On Windows:
```cmd
set TELEGRAM_BOT_TOKEN=123456789:ABCdef...
set TELEGRAM_CHAT_ID=987654321
```

### 3. Get your Angel One TOTP secret

- Log in to Angel One SmartAPI portal → My Profile → Enable TOTP
- Save the **base-32 secret key** (not the OTP itself)

---

## Running

### Basic — NIFTY on 5-minute candles

```bash
python tradewithlines.py \
  --api-key      YOUR_API_KEY \
  --client-code  YOUR_CLIENT_CODE \
  --mpin         YOUR_MPIN \
  --totp-secret  YOUR_TOTP_SECRET_BASE32
```

### Scan multiple indices and timeframes

```bash
python tradewithlines.py \
  --api-key      YOUR_API_KEY \
  --client-code  YOUR_CLIENT_CODE \
  --mpin         YOUR_MPIN \
  --totp-secret  YOUR_TOTP_SECRET_BASE32 \
  --underlyings  NIFTY,BANKNIFTY,FINNIFTY \
  --intervals    ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE \
  --polling-seconds 60
```

### All available options

| Argument | Default | Description |
|---|---|---|
| `--underlyings` | `NIFTY` | Comma-separated: `NIFTY`, `BANKNIFTY`, `FINNIFTY`, `MIDCPNIFTY` |
| `--intervals` | `FIVE_MINUTE` | Comma-separated — see table below |
| `--polling-seconds` | `60` | Seconds between scan cycles |
| `--sens` | `2.0` | Supertrend factor (matches Pine Script default) |
| `--expiry` | auto | Override option expiry e.g. `24APR2025` |
| `--skip-market-hours-check` | off | Run outside 09:15–15:30 IST (testing) |

#### Valid interval strings

```
ONE_MINUTE   THREE_MINUTE   FIVE_MINUTE   TEN_MINUTE
FIFTEEN_MINUTE   THIRTY_MINUTE   ONE_HOUR   ONE_DAY
```

---

## Telegram alert format

```
═══════════════════════════════════
🟢📈  📊 SUPERTREND — BUY
═══════════════════════════════════

📌 Index      : NIFTY
⏱  Timeframe  : FIVE MINUTE
💰 Index LTP  : 22,345.50

⚡ Resistance : 22,400.00
🛡  Support    : 22,250.00

🎯 Entry      : 22,345.50
🛑 Stop Loss  : 22,100.00
✅ TP 1       : 22,517.00
✅ TP 2       : 22,639.00
✅ TP 3       : 22,712.00

─────────────────────────────────
📢 Option Suggestion
   Action  : BUY 22350CE (NIFTY)
   Strike  : 22350  |  Type: CE
   Option LTP : 145.50
═══════════════════════════════════
```

---

## Running as a background service (Linux / VPS)

### Option A — systemd service

Create `/etc/systemd/system/tradewithlines.service`:

```ini
[Unit]
Description=TradeswithLines Angel One Scanner
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/tradewithlines
Environment="TELEGRAM_BOT_TOKEN=YOUR_TOKEN"
Environment="TELEGRAM_CHAT_ID=YOUR_CHAT_ID"
ExecStart=/usr/bin/python3 tradewithlines.py \
    --api-key      YOUR_API_KEY \
    --client-code  YOUR_CLIENT \
    --mpin         YOUR_MPIN \
    --totp-secret  YOUR_TOTP \
    --underlyings  NIFTY,BANKNIFTY \
    --intervals    ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE \
    --polling-seconds 60
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tradewithlines
sudo systemctl start  tradewithlines
sudo journalctl -u tradewithlines -f   # tail logs
```

### Option B — screen / tmux (simpler)

```bash
screen -S scanner
python tradewithlines.py --api-key ... --client-code ... --mpin ... --totp-secret ...
# Ctrl+A then D to detach
screen -r scanner    # re-attach
```

---

## How signals map to actions

| Index signal | Recommended option action |
|---|---|
| BUY  (any signal type) | **Buy ATM CE** of that index |
| SELL (any signal type) | **Buy ATM PE** of that index |

The scanner monitors the **index chart** (not individual options) for signals,
which is more reliable because option prices are heavily influenced by IV and
theta — not just direction.

---

## Adjusting sensitivity

- **Supertrend less sensitive (fewer signals):** increase `--sens` to 2.5 or 3.0
- **Supertrend more sensitive (more signals):** decrease `--sens` to 1.5
- **QQE / Diamond parameters** are hardcoded to match Pine Script defaults
  (RSI_Period=14, SF=5, KQE=4.238). Edit `indicators.py` to change them.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Login failed` | Check api-key, client-code, mpin, totp-secret |
| `Only N bars (need ≥80)` | Increase `--polling-seconds` or use a longer interval |
| No Telegram messages | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars |
| `No expiry found` | Market may be closed; run with `--skip-market-hours-check` to test |
| Angel One rate limit | Increase `--polling-seconds` to 120+ when scanning many indices |
