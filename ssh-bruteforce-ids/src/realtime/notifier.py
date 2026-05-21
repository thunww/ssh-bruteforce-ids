from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load .env from project root
load_dotenv(dotenv_path=Path(".env"))


def _get_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def telegram_is_enabled() -> bool:
    token = _get_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env("TELEGRAM_CHAT_ID")
    return bool(token and chat_id)


def send_telegram(message: str) -> bool:
    token = _get_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_env("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[WARN] Telegram env not configured. Skip sending.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
    }

    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code != 200:
            print(f"[WARN] Telegram send failed: {resp.status_code} {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Telegram exception: {e}")
        return False
