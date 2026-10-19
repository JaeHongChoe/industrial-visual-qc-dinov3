from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SyntheticResult:
    image: np.ndarray
    mask: np.ndarray
    family: str
    parameters: dict[str, float | int | str]


class SyntheticDefectGenerator:
    """Generate varied defects procedurally from aggregate morphology observations."""

    families = ("spot", "discolor", "scratch", "pollution", "fiber", "chipping")
    severities = ("subtle", "medium", "strong")
    severity_probabilities = (0.40, 0.40, 0.20)
    severity_contrast = {"subtle": 0.55, "medium": 1.0, "strong": 1.45}

    def apply(
        self,
        rgb: np.ndarray,
        valid_mask: np.ndarray,
        family: str,
        rng: np.random.Generator,
        severity: str | None = None,
    ) -> SyntheticResult:
        self._validate(rgb, valid_mask, family, severity)
        selected_severity = (
            str(rng.choice(self.severities, p=self.severity_probabilities))
            if severity is None
            else severity
        )
        sharpness = self._sharpness(rgb, valid_mask)
        psf_sigma = float(np.clip(2.2 / (1.0 + math.sqrt(sharpness + 1.0) / 9.0), 0.35, 2.0))
        distance = cv2.distanceTransform(valid_mask.astype(np.uint8), cv2.DIST_L2, 5)
        maximum_support = float(distance.max())
        if maximum_support < 5:
            raise ValueError("valid_mask has no placement region large enough for a defect")

        alpha, parameters = self._make_alpha(
            family=family,
            shape=valid_mask.shape,
            distance=distance,
            maximum_support=maximum_support,
            psf_sigma=psf_sigma,
            severity=selected_severity,
            rng=rng,
        )
        alpha[~valid_mask] = 0.0
        mask = alpha > 0.015
        if not mask.any():
            raise RuntimeError("synthetic geometry became empty after placement validation")

        contrast_scale = self.severity_contrast[selected_severity]
        output = self._blend_lab(rgb, alpha, family, contrast_scale, rng)
        output[alpha <= 0.0] = rgb[alpha <= 0.0]
        parameters = dict(parameters)
        parameters["psf_sigma"] = round(psf_sigma, 6)
        parameters["sharpness"] = round(sharpness, 6)
        parameters["severity"] = selected_severity
        parameters["contrast_scale"] = contrast_scale
        return SyntheticResult(output, mask, family, parameters)

    @staticmethod
    def _validate(
        rgb: np.ndarray,
        valid_mask: np.ndarray,
        family: str,
        severity: str | None,
    ) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if valid_mask.shape != rgb.shape[:2] or valid_mask.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean array matching rgb height and width")
        if family not in SyntheticDefectGenerator.families:
            raise ValueError(f"unsupported family: {family}")
        if severity is not None and severity not in SyntheticDefectGenerator.severities:
            raise ValueError(f"unsupported severity: {severity}")

    @staticmethod
    def _sharpness(rgb: np.ndarray, valid_mask: np.ndarray) -> float:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        values = laplacian[valid_mask]
        return float(values.var()) if values.size else 0.0

    @staticmethod
    def _pick_center(distance: np.ndarray, support: float, rng: np.random.Generator) -> tuple[int, int]:
        candidates = np.argwhere(distance >= support)
        if not len(candidates):
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            return int(y), int(x)
        y, x = candidates[int(rng.integers(0, len(candidates)))]
        return int(y), int(x)

    def _make_alpha(
        self,
        family: str,
        shape: tuple[int, int],
        distance: np.ndarray,
        maximum_support: float,
        psf_sigma: float,
        severity: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        target_ranges = {
            "spot": (6.0, 18.0),
            "discolor": (24.0, 72.0),
            "scratch": (22.0, 78.0),
            "pollution": (18.0, 58.0),
            "fiber": (30.0, 92.0),
            "chipping": (12.0, 38.0),
        }
        low, high = target_ranges[family]
        support = float(min(rng.uniform(low, high), maximum_support * 0.82))
        support = max(4.0, support)
        center = self._pick_center(distance, support, rng)
        if family == "spot":
            return self._spot_alpha(shape, center, support, psf_sigma, rng)
        if family == "discolor":
            return self._discolor_alpha(shape, center, support, psf_sigma, rng)
        if family == "scratch":
            return self._scratch_alpha(shape, center, support, psf_sigma, rng)
        if family == "pollution":
            return self._pollution_alpha(shape, center, support, psf_sigma, rng)
        if family == "fiber":
            return self._fiber_alpha(shape, center, support, psf_sigma, rng)
        return self._chipping_alpha(shape, center, support, psf_sigma, severity, rng)

    @staticmethod
    def _blur(alpha: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return alpha
        return cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)

    def _spot_alpha(self, shape, center, support, psf_sigma, rng):
        alpha = np.zeros(shape, dtype=np.float32)
        y, x = center
        lobes = int(rng.integers(1, 4))
        radius = max(1.2, support * rng.uniform(0.16, 0.42))
        for _ in range(lobes):
            ox = int(round(rng.normal(0, radius * 0.35)))
            oy = int(round(rng.normal(0, radius * 0.35)))
            axes = (max(1, round(radius * rng.uniform(0.65, 1.25))), max(1, round(radius * rng.uniform(0.65, 1.25))))
            cv2.ellipse(alpha, (x + ox, y + oy), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        sigma = psf_sigma + float(rng.uniform(0.2, 1.1))
        alpha = self._blur(alpha, sigma)
        return np.clip(alpha, 0, 1), {"support": round(support, 4), "lobes": lobes, "radius": round(radius, 4)}

    def _discolor_alpha(self, shape, center, support, psf_sigma, rng):
        y, x = center
        base = np.zeros(shape, dtype=np.float32)
        axes = (max(3, round(support * rng.uniform(0.45, 0.78))), max(3, round(support * rng.uniform(0.38, 0.72))))
        cv2.ellipse(base, (x, y), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        base = self._blur(base, max(3.0, support * rng.uniform(0.08, 0.18)) + psf_sigma)
        texture = rng.normal(0, 1, shape).astype(np.float32)
        texture = self._blur(texture, max(5.0, support * 0.15))
        texture -= float(texture.min())
        texture /= float(texture.max() + 1e-6)
        alpha = base * (0.42 + 0.58 * texture)
        return np.clip(alpha, 0, 1), {"support": round(support, 4), "axis_x": axes[0], "axis_y": axes[1]}

    def _scratch_alpha(self, shape, center, support, psf_sigma, rng):
        alpha = np.zeros(shape, dtype=np.float32)
        y, x = center
        angle = float(rng.uniform(0, math.pi))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        half_length = support * rng.uniform(0.55, 0.88)
        curvature = support * rng.uniform(-0.24, 0.24)
        ts = np.linspace(-1.0, 1.0, 25)
        points = []
        for t in ts:
            point = np.array([x, y]) + direction * half_length * t + normal * curvature * (1.0 - t * t)
            points.append(np.rint(point).astype(np.int32))
        width = int(rng.choice((1, 1, 1, 2, 2, 3)))
        cv2.polylines(alpha, [np.asarray(points)], False, 1.0, width, lineType=cv2.LINE_AA)
        if rng.random() < 0.5:
            gap = points[len(points) // 2]
            cv2.circle(alpha, tuple(gap), max(1, width), 0.0, -1)
        if rng.random() < 0.22:
            branch_start = np.asarray(points[int(rng.integers(8, 17))])
            branch_length = support * rng.uniform(0.12, 0.30)
            branch_angle = angle + rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.85)
            branch_end = branch_start + branch_length * np.array(
                [math.cos(branch_angle), math.sin(branch_angle)]
            )
            cv2.line(
                alpha,
                tuple(branch_start),
                tuple(np.rint(branch_end).astype(np.int32)),
                1.0,
                max(1, width - 1),
                lineType=cv2.LINE_AA,
            )
        alpha = self._blur(alpha, psf_sigma * 0.65)
        return np.clip(alpha, 0, 1), {"support": round(support, 4), "length": round(half_length * 2, 4), "width": width}

    def _pollution_alpha(self, shape, center, support, psf_sigma, rng):
        y, x = center
        alpha = np.zeros(shape, dtype=np.float32)
        point_count = int(rng.integers(9, 18))
        points = []
        for angle in np.linspace(0, 2 * math.pi, point_count, endpoint=False):
            radius = support * rng.uniform(0.35, 0.82)
            points.append((round(x + math.cos(angle) * radius), round(y + math.sin(angle) * radius)))
        cv2.fillPoly(alpha, [np.asarray(points, dtype=np.int32)], 1.0)
        texture = rng.uniform(0, 1, shape).astype(np.float32)
        texture = self._blur(texture, rng.uniform(0.8, 2.5))
        texture -= float(texture.min())
        texture /= float(texture.max() + 1e-6)
        alpha *= 0.35 + 0.65 * texture
        alpha = self._blur(alpha, psf_sigma + rng.uniform(0.8, 2.6))
        return np.clip(alpha, 0, 1), {"support": round(support, 4), "vertices": point_count}

    def _fiber_alpha(self, shape, center, support, psf_sigma, rng):
        y, x = center
        alpha = np.zeros(shape, dtype=np.float32)
        angle = float(rng.uniform(0, math.pi))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        start = np.array([x, y]) - direction * support * 0.78
        end = np.array([x, y]) + direction * support * 0.78
        control1 = np.array([x, y]) - direction * support * 0.25 + normal * support * rng.uniform(-0.5, 0.5)
        control2 = np.array([x, y]) + direction * support * 0.25 + normal * support * rng.uniform(-0.5, 0.5)
        points = []
        for t in np.linspace(0.0, 1.0, 48):
            point = (
                (1 - t) ** 3 * start
                + 3 * (1 - t) ** 2 * t * control1
                + 3 * (1 - t) * t**2 * control2
                + t**3 * end
            )
            points.append(np.rint(point).astype(np.int32))
        width = int(rng.choice((1, 1, 1, 1, 2, 2, 3)))
        cv2.polylines(alpha, [np.asarray(points)], False, 1.0, width, lineType=cv2.LINE_AA)
        alpha = self._blur(alpha, psf_sigma * 0.8)
        return np.clip(alpha, 0, 1), {"support": round(support, 4), "length": round(support * 1.56, 4), "width": width}

    def _chipping_alpha(self, shape, center, support, psf_sigma, severity, rng):
        alpha = np.zeros(shape, dtype=np.float32)
        y, x = center
        fragment_ranges = {
            "subtle": (1, 2),
            "medium": (2, 4),
            "strong": (4, 7),
        }
        low, high = fragment_ranges[severity]
        fragments = int(rng.integers(low, high))
        core_radius = max(1.8, support * rng.uniform(0.13, 0.22))
        for index in range(fragments):
            if index == 0:
                fragment_x, fragment_y = x, y
                fragment_radius = core_radius
            else:
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                separation = core_radius * rng.uniform(2.8, 4.4)
                fragment_x = round(x + math.cos(angle) * separation)
                fragment_y = round(y + math.sin(angle) * separation)
                fragment_radius = core_radius * rng.uniform(0.38, 0.72)
            vertex_count = int(rng.integers(7, 14))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            points = []
            for angle in np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False):
                radius = fragment_radius * rng.uniform(0.55, 1.30)
                points.append(
                    (
                        round(fragment_x + math.cos(angle + phase) * radius),
                        round(fragment_y + math.sin(angle + phase) * radius),
                    )
                )
            cv2.fillPoly(alpha, [np.asarray(points, dtype=np.int32)], 1.0)
        alpha = self._blur(alpha, max(0.25, psf_sigma * 0.32))
        return np.clip(alpha, 0, 1), {
            "support": round(support, 4),
            "fragments": fragments,
            "core_radius": round(core_radius, 4),
            "placement": "surface",
        }

    @staticmethod
    def _blend_lab(
        rgb: np.ndarray,
        alpha: np.ndarray,
        family: str,
        contrast_scale: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        if family == "spot":
            sign = -1.0 if rng.random() < 0.5 else 1.0
            residual = np.array([sign * rng.uniform(5, 24), rng.uniform(-2, 2), rng.uniform(-2, 2)])
        elif family == "discolor":
            residual = np.array([rng.uniform(-12, 7), rng.uniform(-9, 9), rng.uniform(-10, 10)])
        elif family == "scratch":
            sign = -1.0 if rng.random() < 0.65 else 1.0
            residual = np.array([sign * rng.uniform(5, 22), rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)])
        elif family == "pollution":
            residual = np.array([-rng.uniform(8, 38), rng.uniform(-5, 5), rng.uniform(-5, 8)])
        elif family == "fiber":
            sign = -1.0 if rng.random() < 0.8 else 1.0
            residual = np.array([sign * rng.uniform(10, 36), rng.uniform(-3, 3), rng.uniform(-3, 3)])
        else:
            residual = np.array([rng.uniform(20, 52), rng.uniform(-4, 4), rng.uniform(-5, 4)])
        residual *= contrast_scale
        modulation = 1.0 + rng.normal(0, 0.08, alpha.shape).astype(np.float32)
        lab += alpha[..., None] * modulation[..., None] * residual.astype(np.float32)
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
