from src.realtime.notifier import telegram_is_enabled, send_telegram

print("Telegram enabled:", telegram_is_enabled())
ok = send_telegram("Test message from SSH IDS project")
print("Send OK:", ok)
