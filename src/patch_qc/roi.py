from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 array")


def _resize_for_analysis(rgb: np.ndarray, max_side: int = 768) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return rgb, scale
    resized = cv2.resize(
        rgb,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _gradient(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )


@dataclass(frozen=True, slots=True)
class _CircleEvidence:
    score: float
    center_y_fraction: float
    radius_fraction: float


def _plate_circle_evidence(rgb: np.ndarray) -> _CircleEvidence:
    height, width = rgb.shape[:2]
    gradient = _gradient(rgb)
    shortest = min(height, width)
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    best = _CircleEvidence(0.0, 0.5, 0.35)
    for center_y_fraction in np.linspace(0.42, 0.62, 9):
        center_x = width * 0.5
        center_y = height * float(center_y_fraction)
        for radius_fraction in np.linspace(0.25, 0.49, 25):
            radius = shortest * float(radius_fraction)
            xs = np.rint(center_x + radius * np.cos(angles)).astype(np.int32)
            ys = np.rint(center_y + radius * np.sin(angles)).astype(np.int32)
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            values = gradient[ys[valid], xs[valid]]
            if values.size < 500:
                continue
            mean_gradient = float(np.mean(np.clip(values, 0.0, 80.0)))
            edge_fraction = float(np.mean(values >= 12.0))
            score = mean_gradient * (0.25 + edge_fraction)
            if score > best.score:
                best = _CircleEvidence(
                    score=score,
                    center_y_fraction=float(center_y_fraction),
                    radius_fraction=float(radius_fraction),
                )
    return best


@dataclass(frozen=True, slots=True)
class _EllipseEvidence:
    score: float
    center_x_fraction: float
    center_y_fraction: float
    radius_x_fraction: float
    radius_y_fraction: float


def _side_plate_ellipse_evidence(rgb: np.ndarray) -> _EllipseEvidence:
    height, width = rgb.shape[:2]
    gradient = _gradient(rgb)
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    best = _EllipseEvidence(0.0, 0.5, 0.82, 0.2, 0.04)
    for center_x_fraction in (0.46, 0.50, 0.54):
        center_x = width * center_x_fraction
        for center_y_fraction in np.linspace(0.64, 0.91, 10):
            center_y = height * float(center_y_fraction)
            for radius_x_fraction in np.linspace(0.12, 0.44, 12):
                radius_x = width * float(radius_x_fraction)
                for aspect in (0.08, 0.12, 0.18, 0.25):
                    radius_y = radius_x * aspect
                    radius_y_fraction = radius_y / height
                    if not 0.018 <= radius_y_fraction <= 0.18:
                        continue
                    xs = np.rint(center_x + radius_x * np.cos(angles)).astype(np.int32)
                    ys = np.rint(center_y + radius_y * np.sin(angles)).astype(np.int32)
                    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                    values = gradient[ys[valid], xs[valid]]
                    if values.size < 260:
                        continue
                    mean_gradient = float(np.mean(np.clip(values, 0.0, 80.0)))
                    edge_fraction = float(np.mean(values >= 12.0))
                    score = mean_gradient * (0.20 + edge_fraction) * (
                        0.65 + radius_x_fraction
                    )
                    if score > best.score:
                        best = _EllipseEvidence(
                            score=score,
                            center_x_fraction=float(center_x_fraction),
                            center_y_fraction=float(center_y_fraction),
                            radius_x_fraction=float(radius_x_fraction),
                            radius_y_fraction=float(radius_y_fraction),
                        )
    return best


def _top_mask(rgb: np.ndarray, evidence: _CircleEvidence) -> np.ndarray:
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    radius = max(2, round(min(height, width) * evidence.radius_fraction))
    cv2.circle(
        mask,
        (width // 2, round(height * evidence.center_y_fraction)),
        radius,
        1,
        -1,
    )
    return mask


def _side_mask(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    left0, left1 = round(width * 0.32), round(width * 0.40)
    right0, right1 = round(width * 0.60), round(width * 0.68)
    left_background = np.median(lab[:, left0:left1], axis=1)
    right_background = np.median(lab[:, right0:right1], axis=1)
    alpha = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    expected = left_background[:, None, :] * (1.0 - alpha) + right_background[:, None, :] * alpha
    residual = np.linalg.norm(lab - expected, axis=2)

    core0, core1 = round(width * 0.45), round(width * 0.55)
    row_signal = np.median(residual[:, core0:core1], axis=1)
    active_rows = (row_signal >= 3.5).astype(np.uint8)[:, None]
    vertical_size = max(3, round(height * 0.025))
    if vertical_size % 2 == 0:
        vertical_size += 1
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    active_rows = cv2.morphologyEx(active_rows, cv2.MORPH_CLOSE, vertical_kernel)
    active_rows = cv2.morphologyEx(
        active_rows,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, vertical_size // 3))),
    )
    active_indices = np.flatnonzero(active_rows[:, 0])
    if active_indices.size < max(3, round(height * 0.02)):
        return np.zeros((height, width), dtype=np.uint8)

    runs = np.split(active_indices, np.flatnonzero(np.diff(active_indices) > 1) + 1)
    center_y = (height - 1) / 2.0

    def distance_to_center(run: np.ndarray) -> tuple[float, int]:
        distance = 0.0 if run[0] <= center_y <= run[-1] else min(
            abs(float(run[0]) - center_y),
            abs(float(run[-1]) - center_y),
        )
        return distance, -int(run.size)

    selected_run = min(runs, key=distance_to_center)
    selected_rows = np.zeros_like(active_rows)
    selected_rows[selected_run, 0] = 1
    row_gate = cv2.dilate(selected_rows, vertical_kernel)

    prior = np.zeros((height, width), dtype=np.uint8)
    prior[
        round(height * 0.015) : round(height * 0.97),
        round(width * 0.28) : round(width * 0.72),
    ] = 1
    candidate = ((residual >= 5.0) & (prior > 0) & (row_gate > 0)).astype(np.uint8)
    kernel_size = max(3, round(min(height, width) * 0.004))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    cleaned = np.zeros_like(candidate)
    minimum_area = max(12, round(height * width * 0.0002))
    core_left, core_right = width * 0.45, width * 0.55
    for label in range(1, component_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if (
            area >= minimum_area
            and x <= core_right
            and x + component_width >= core_left
            and component_width <= width * 0.40
        ):
            cleaned[labels == label] = 1

    safe_core = np.zeros_like(cleaned)
    safe_core[:, core0:core1] = selected_rows
    cleaned |= safe_core
    ellipse = _side_plate_ellipse_evidence(rgb)
    if ellipse.score >= 3.0:
        cv2.ellipse(
            cleaned,
            (
                round(width * ellipse.center_x_fraction),
                round(height * ellipse.center_y_fraction),
            ),
            (
                max(2, round(width * ellipse.radius_x_fraction)),
                max(2, round(height * ellipse.radius_y_fraction)),
            ),
            0,
            0,
            360,
            1,
            -1,
        )
    return cv2.dilate(cleaned, kernel)


def estimate_placement_mask(rgb: np.ndarray, margin: int = 8) -> np.ndarray:
    """Estimate a conservative native product ROI for synthetic placement.

    A large circular plate is localized from its supported boundary. Other views
    use row-relative background residuals plus the central stem/base geometry.
    Analysis is performed on a bounded copy and returned at original resolution.
    """
    _validate_rgb(rgb)
    if margin < 0:
        raise ValueError("margin must be non-negative")

    original_height, original_width = rgb.shape[:2]
    analysis, scale = _resize_for_analysis(rgb)
    circle = _plate_circle_evidence(analysis)
    mask = _top_mask(analysis, circle) if circle.score >= 12.0 else _side_mask(analysis)
    if not mask.any():
        return np.zeros((original_height, original_width), dtype=bool)
    if scale != 1.0:
        mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
    if margin:
        diameter = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.erode(mask, kernel)
    return mask.astype(bool)
