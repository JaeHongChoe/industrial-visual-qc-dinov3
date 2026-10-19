from __future__ import annotations

import math


def calibrate_threshold(normal_logits: list[float], target_fpr: float) -> float:
    """Choose the lowest strict-greater threshold that stays within the empirical FPR budget."""
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be strictly between 0 and 1")
    finite = sorted(float(score) for score in normal_logits if math.isfinite(float(score)))
    if not finite:
        raise ValueError("at least one finite normal logit is required")
    allowed_positives = int(math.floor(target_fpr * len(finite)))
    if allowed_positives <= 0:
        return finite[-1]
    threshold_index = max(0, len(finite) - allowed_positives - 1)
    return finite[threshold_index]
