from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


ALERT_THRESHOLD = 0.2
BLOCK_THRESHOLD = 0.4
BLOCK_SECONDS = 300


@dataclass
class IPState:
    consecutive_suspicious: int = 0
    blocked_until: datetime | None = None
    last_action: str = "NORMAL"


@dataclass
class EarlyStopDetector:
    alert_threshold: float = ALERT_THRESHOLD
    block_threshold: float = BLOCK_THRESHOLD
    block_seconds: int = BLOCK_SECONDS
    state_table: dict = field(default_factory=dict)

    def get_state(self, src_ip: str) -> IPState:
        if src_ip not in self.state_table:
            self.state_table[src_ip] = IPState()
        return self.state_table[src_ip]

    def decide(self, src_ip: str, now: datetime, risk_score: float) -> dict:
        state = self.get_state(src_ip)

        # check block status
        if state.blocked_until is not None and now < state.blocked_until:
            return {
                "src_ip": src_ip,
                "action": "BLOCKED",
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        # expired block
        if state.blocked_until is not None and now >= state.blocked_until:
            state.blocked_until = None

        # decision logic
        if risk_score < self.alert_threshold:
            state.consecutive_suspicious = 0
            state.last_action = "NORMAL"
            return {
                "src_ip": src_ip,
                "action": "NORMAL",
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        if self.alert_threshold <= risk_score < self.block_threshold:
            state.consecutive_suspicious += 1
            state.last_action = "ALERT"
            return {
                "src_ip": src_ip,
                "action": "ALERT",
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        # risk >= block_threshold
        state.consecutive_suspicious += 1
        if state.consecutive_suspicious >= 2:
            state.blocked_until = now + timedelta(seconds=self.block_seconds)
            state.last_action = "BLOCK"
            return {
                "src_ip": src_ip,
                "action": "BLOCK",
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        state.last_action = "ALERT"
        return {
            "src_ip": src_ip,
            "action": "ALERT",
            "risk_score": risk_score,
            "consecutive_suspicious": state.consecutive_suspicious,
            "blocked_until": state.blocked_until,
        }
