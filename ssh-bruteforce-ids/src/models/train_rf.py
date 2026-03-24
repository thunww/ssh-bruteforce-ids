from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_rf_model() -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return model
