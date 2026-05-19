"""
telegram_notifier.py — sends messages to a Telegram chat.

Required environment variables:
  TELEGRAM_BOT_TOKEN  — bot token from @BotFather
  TELEGRAM_CHAT_ID    — your chat / channel ID
"""

import os
import time
import requests


def send_telegram(message: str, retries: int = 3) -> None:
    """
    Send a Telegram message.  Silently skips if env vars are not set.
    Retries up to `retries` times on network failure.
    """
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping.")
        return

    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":                  chat_id,
        "text":                     message,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                return
            print(f"[Telegram] HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"[Telegram] Timeout on attempt {attempt}/{retries}")
        except Exception as exc:
            print(f"[Telegram] Error: {exc}")

        if attempt < retries:
            time.sleep(2 ** attempt)    # back-off: 2s, 4s
