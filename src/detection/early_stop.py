from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


ALERT_THRESHOLD = 0.2
BLOCK_THRESHOLD = 0.4
BLOCK_SECONDS = 300
LOW_SLOW_ALERT_COUNT_5M = 12
LOW_SLOW_BLOCK_COUNT_15M = 24


@dataclass
class IPState:
    consecutive_suspicious: int = 0
    blocked_until: datetime | None = None
    last_action: str = "NORMAL"
    last_reason: str = "RISK_NORMAL"


@dataclass
class EarlyStopDetector:
    alert_threshold: float = ALERT_THRESHOLD
    block_threshold: float = BLOCK_THRESHOLD
    block_seconds: int = BLOCK_SECONDS
    low_slow_alert_count_5m: int = LOW_SLOW_ALERT_COUNT_5M
    low_slow_block_count_15m: int = LOW_SLOW_BLOCK_COUNT_15M
    state_table: dict = field(default_factory=dict)

    def get_state(self, src_ip: str) -> IPState:
        if src_ip not in self.state_table:
            self.state_table[src_ip] = IPState()
        return self.state_table[src_ip]

    def decide(
        self,
        src_ip: str,
        now: datetime,
        risk_score: float,
        failed_5m: int = 0,
        failed_15m: int = 0,
        username: str | None = None,
    ) -> dict:
        state = self.get_state(src_ip)

        # check block status
        if state.blocked_until is not None and now < state.blocked_until:
            return {
                "src_ip": src_ip,
                "action": "BLOCKED",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        # expired block
        if state.blocked_until is not None and now >= state.blocked_until:
            state.blocked_until = None

        if risk_score >= self.block_threshold:
            state.consecutive_suspicious += 1
            if state.consecutive_suspicious >= 2:
                state.blocked_until = now + timedelta(seconds=self.block_seconds)
                state.last_action = "BLOCK"
                state.last_reason = "RISK_THRESHOLD"
                return {
                    "src_ip": src_ip,
                    "action": "BLOCK",
                    "reason": state.last_reason,
                    "risk_score": risk_score,
                    "consecutive_suspicious": state.consecutive_suspicious,
                    "blocked_until": state.blocked_until,
                }

            state.last_action = "ALERT"
            state.last_reason = "RISK_THRESHOLD"
            return {
                "src_ip": src_ip,
                "action": "ALERT",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        if failed_15m >= self.low_slow_block_count_15m:
            state.consecutive_suspicious += 1
            state.blocked_until = now + timedelta(seconds=self.block_seconds)
            state.last_action = "BLOCK"
            state.last_reason = "LOW_AND_SLOW_15M"
            return {
                "src_ip": src_ip,
                "username": username,
                "action": "BLOCK",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        if failed_5m >= self.low_slow_alert_count_5m:
            state.consecutive_suspicious += 1
            state.last_action = "ALERT"
            state.last_reason = "LOW_AND_SLOW_5M"
            return {
                "src_ip": src_ip,
                "username": username,
                "action": "ALERT",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        if risk_score < self.alert_threshold:
            state.consecutive_suspicious = 0
            state.last_action = "NORMAL"
            state.last_reason = "RISK_NORMAL"
            return {
                "src_ip": src_ip,
                "action": "NORMAL",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }

        if self.alert_threshold <= risk_score < self.block_threshold:
            state.consecutive_suspicious += 1
            state.last_action = "ALERT"
            state.last_reason = "RISK_THRESHOLD"
            return {
                "src_ip": src_ip,
                "action": "ALERT",
                "reason": state.last_reason,
                "risk_score": risk_score,
                "consecutive_suspicious": state.consecutive_suspicious,
                "blocked_until": state.blocked_until,
            }
