from __future__ import annotations


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def compute_risk_score(
    model_prob: float,
    flow_rate_per_window: float,
    interarrival_std: float,
    rst_flow_ratio: float,
    short_flow_ratio: float,
) -> dict:
    # normalize components
    norm_flow_rate = clamp01(flow_rate_per_window / 0.5)
    inv_interarrival_std = 1.0 - clamp01(interarrival_std / 5.0)
    norm_rst_ratio = clamp01(rst_flow_ratio)
    norm_short_ratio = clamp01(short_flow_ratio)
    norm_model_prob = clamp01(model_prob)

    risk_score = (
        0.60 * norm_model_prob +
        0.15 * norm_flow_rate +
        0.10 * inv_interarrival_std +
        0.10 * norm_rst_ratio +
        0.05 * norm_short_ratio
    )

    return {
        "model_prob": norm_model_prob,
        "norm_flow_rate": norm_flow_rate,
        "inv_interarrival_std": inv_interarrival_std,
        "norm_rst_ratio": norm_rst_ratio,
        "norm_short_ratio": norm_short_ratio,
        "risk_score": risk_score,
    }
