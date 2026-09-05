from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DecodeConfig:
    center_threshold: float = 0.25
    min_component_pixels: int = 3
    min_component_width: int = 2
    min_region_probability: float = 0.12
    height_factor: float = 0.50
    vertical_padding_px: float = 1.0
    horizontal_padding_px: float = 4.0
    min_line_height_px: float = 7.0
    max_line_height_px: float = 90.0
    merge_center_distance_ratio: float = 0.28
    merge_gap_ratio: float = 1.35


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _keep_horizontal_runs(mask: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Remove isolated line-center pixels with a WASM-friendly run filter."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for y in range(h):
        x = 0
        while x < w:
            if not mask[y, x]:
                x += 1
                continue
            x1 = x + 1
            while x1 < w and mask[y, x1]:
                x1 += 1
            if x1 - x >= min_length:
                out[y, x:x1] = True
            x = x1
    return out


def _connected_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return 8-connected components as y/x coordinate arrays.

    This deliberately avoids SciPy/OpenCV so the decoder logic mirrors what can
    later be implemented directly in Rust/WASM.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[np.ndarray, np.ndarray]] = []

    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            ys: list[int] = []
            xs: list[int] = []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append((np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)))
    return components


def _merge_boxes(
    detections: list[Detection],
    center_distance_ratio: float,
    gap_ratio: float,
) -> list[Detection]:
    current = list(detections)
    changed = True
    while changed:
        changed = False
        out: list[Detection] = []
        used = [False] * len(current)
        for i, det_a in enumerate(current):
            if used[i]:
                continue
            x0, y0, x1, y1 = det_a.box
            score = det_a.score
            used[i] = True
            for j, det_b in enumerate(current):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = det_b.box
                ha = y1 - y0
                hb = by1 - by0
                ref_h = max(ha, hb, 1.0)
                cya = (y0 + y1) * 0.5
                cyb = (by0 + by1) * 0.5
                gap = max(0.0, bx0 - x1, x0 - bx1)
                if abs(cya - cyb) <= center_distance_ratio * ref_h and gap <= gap_ratio * ref_h:
                    x0 = min(x0, bx0)
                    y0 = min(y0, by0)
                    x1 = max(x1, bx1)
                    y1 = max(y1, by1)
                    score = max(score, det_b.score)
                    used[j] = True
                    changed = True
            out.append(Detection((x0, y0, x1, y1), score))
        current = out
    return sorted(current, key=lambda d: ((d.box[1] + d.box[3]) * 0.5, d.box[0]))


def decode_maps(
    line_center_logits: np.ndarray,
    log_line_height: np.ndarray,
    region_logits: np.ndarray,
    image_width: int,
    image_height: int,
    config: DecodeConfig | None = None,
) -> list[Detection]:
    """Decode model output maps into line boxes in input-image coordinates."""
    cfg = config or DecodeConfig()
    center_prob = sigmoid(line_center_logits)
    region_prob = sigmoid(region_logits)
    if center_prob.shape != log_line_height.shape or center_prob.shape != region_prob.shape:
        raise ValueError("all output maps must have the same shape")

    mask = _keep_horizontal_runs(center_prob >= cfg.center_threshold, min_length=2)
    oh, ow = center_prob.shape
    sx = image_width / float(ow)
    sy = image_height / float(oh)
    detections: list[Detection] = []

    for ys, xs in _connected_components(mask):
        if len(xs) < cfg.min_component_pixels:
            continue
        gx0 = int(xs.min())
        gx1 = int(xs.max()) + 1
        if gx1 - gx0 < cfg.min_component_width:
            continue

        weights = center_prob[ys, xs] + 1e-4
        center_y_grid = float(np.average(ys + 0.5, weights=weights))
        mean_log_h = float(np.average(log_line_height[ys, xs], weights=weights))
        region_score = float(np.mean(region_prob[ys, xs]))
        if region_score < cfg.min_region_probability:
            continue

        line_h = math.exp(float(np.clip(mean_log_h, -2.0, 4.0))) * sy
        line_h = float(np.clip(line_h, cfg.min_line_height_px, cfg.max_line_height_px))
        center_y = center_y_grid * sy

        x0 = max(0.0, gx0 * sx - cfg.horizontal_padding_px)
        x1 = min(float(image_width), gx1 * sx + cfg.horizontal_padding_px)
        y0 = max(0.0, center_y - line_h * cfg.height_factor - cfg.vertical_padding_px)
        y1 = min(float(image_height), center_y + line_h * cfg.height_factor + cfg.vertical_padding_px)
        score = float(np.average(center_prob[ys, xs], weights=weights))
        detections.append(Detection((x0, y0, x1, y1), score))

    return _merge_boxes(
        detections,
        center_distance_ratio=cfg.merge_center_distance_ratio,
        gap_ratio=cfg.merge_gap_ratio,
    )
