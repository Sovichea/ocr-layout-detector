#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time

import fitz
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ocr_layout_detector.model import OUTPUT_STRIDE, TinyLayoutNet, parameter_count


@dataclass
class PageSample:
    page_no: int
    image: np.ndarray
    center: np.ndarray
    log_height: np.ndarray
    region: np.ndarray


def extract_pdf_dataset(pdf_path: Path, dpi: int) -> list[PageSample]:
    """Build detector targets from vector PDF line/block geometry.

    Extracted Unicode is not used as recognition ground truth. Only visual text
    line and block geometry is used by this detector experiment.
    """
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    pages: list[PageSample] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        sx = pix.width / page.rect.width
        sy = pix.height / page.rect.height

        boxes: list[list[float]] = []
        block_boxes: list[list[float]] = []
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_lines: list[list[float]] = []
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if not text.strip():
                    continue
                x0, y0, x1, y1 = line["bbox"]
                box = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
                if box[2] - box[0] < 4 or box[3] - box[1] < 3:
                    continue
                boxes.append(box)
                block_lines.append(box)
            if block_lines:
                block_boxes.append(
                    [
                        min(b[0] for b in block_lines),
                        min(b[1] for b in block_lines),
                        max(b[2] for b in block_lines),
                        max(b[3] for b in block_lines),
                    ]
                )

        h, w = gray.shape
        oh = math.ceil(h / OUTPUT_STRIDE)
        ow = math.ceil(w / OUTPUT_STRIDE)
        center = np.zeros((oh, ow), dtype=np.float32)
        log_height = np.zeros((oh, ow), dtype=np.float32)
        region = np.zeros((oh, ow), dtype=np.float32)
        ox = ow / w
        oy = oh / h

        for x0, y0, x1, y1 in boxes:
            xa = max(0, int(math.floor(x0 * ox)))
            xb = min(ow, int(math.ceil(x1 * ox)))
            center_y = (y0 + y1) * 0.5 * oy
            line_h = max(1.0, (y1 - y0) * oy)
            half = max(1, int(round(line_h * 0.18)))
            ya = max(0, int(math.floor(center_y)) - half)
            yb = min(oh, int(math.floor(center_y)) + half + 1)
            if xb > xa and yb > ya:
                center[ya:yb, xa:xb] = 1.0
                log_height[ya:yb, xa:xb] = math.log(line_h)

        for x0, y0, x1, y1 in block_boxes:
            xa = max(0, int(math.floor(x0 * ox)))
            xb = min(ow, int(math.ceil(x1 * ox)))
            ya = max(0, int(math.floor(y0 * oy)))
            yb = min(oh, int(math.ceil(y1 * oy)))
            if xb > xa and yb > ya:
                region[ya:yb, xa:xb] = 1.0

        pages.append(PageSample(page_index + 1, gray, center, log_height, region))

    return pages


def sample_batch(pages: list[PageSample], batch_size: int = 4, patch_size: int = 256):
    xs: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    regions: list[np.ndarray] = []

    for _ in range(batch_size):
        page = random.choice(pages)
        h, w = page.image.shape
        size = min(patch_size, h, w)
        y0 = random.randrange(0, max(1, h - size + 1), OUTPUT_STRIDE) if h > size else 0
        x0 = random.randrange(0, max(1, w - size + 1), OUTPUT_STRIDE) if w > size else 0

        gray = page.image[y0:y0 + size, x0:x0 + size].astype(np.float32)
        ink = 1.0 - gray / 255.0
        gain = random.uniform(0.65, 1.35)
        bias = random.uniform(-0.04, 0.05)
        ink = np.clip(ink * gain + bias, 0.0, 1.0)
        if random.random() < 0.45:
            sigma = random.uniform(0.0, 0.045)
            noise = np.random.normal(0.0, sigma, ink.shape).astype(np.float32)
            ink = np.clip(ink + noise, 0.0, 1.0)
        if random.random() < 0.20:
            speckles = np.random.random(ink.shape) < random.uniform(0.0002, 0.0015)
            ink[speckles] = np.maximum(ink[speckles], random.uniform(0.5, 1.0))

        oy0 = y0 // OUTPUT_STRIDE
        ox0 = x0 // OUTPUT_STRIDE
        output_size = size // OUTPUT_STRIDE
        xs.append(ink[None])
        centers.append(page.center[oy0:oy0 + output_size, ox0:ox0 + output_size])
        heights.append(page.log_height[oy0:oy0 + output_size, ox0:ox0 + output_size])
        regions.append(page.region[oy0:oy0 + output_size, ox0:ox0 + output_size])

    x = torch.from_numpy(np.stack(xs))
    center = torch.from_numpy(np.stack(centers))
    height = torch.from_numpy(np.stack(heights))
    region = torch.from_numpy(np.stack(regions))
    if random.random() < 0.25:
        x = F.avg_pool2d(x, 3, 1, 1)
    return x, center, height, region


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    numerator = 2.0 * (prob * target).sum(dim=(1, 2)) + 1.0
    denominator = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0
    return (1.0 - numerator / denominator).mean()


def train(pages: list[PageSample], steps: int, seed: int) -> tuple[nn.Module, list[dict]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = TinyLayoutNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
    history: list[dict] = []
    started = time.time()

    for step in range(1, steps + 1):
        x, center_target, height_target, region_target = sample_batch(pages)
        output = model(x)
        center_logits = output[:, 0]
        height_prediction = output[:, 1]
        region_logits = output[:, 2]

        center_loss = F.binary_cross_entropy_with_logits(
            center_logits,
            center_target,
            pos_weight=torch.tensor(4.0),
        ) + 0.45 * dice_loss(center_logits, center_target)

        height_mask = center_target > 0.5
        height_loss = (
            F.smooth_l1_loss(height_prediction[height_mask], height_target[height_mask])
            if height_mask.any()
            else height_prediction.mean() * 0.0
        )

        region_loss = F.binary_cross_entropy_with_logits(
            region_logits,
            region_target,
            pos_weight=torch.tensor(1.5),
        ) + 0.20 * dice_loss(region_logits, region_target)

        loss = center_loss + 0.18 * height_loss + 0.22 * region_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "center": float(center_loss.detach()),
                "height": float(height_loss.detach()),
                "region": float(region_loss.detach()),
            }
            history.append(row)
            print(row, flush=True)

    print(f"trained {parameter_count(model):,} parameters in {time.time() - started:.2f}s")
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    pages = extract_pdf_dataset(args.pdf, args.dpi)
    model, history = train(pages, args.steps, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "architecture": "tiny-layout-net-v0",
            "parameters": parameter_count(model),
            "stride": OUTPUT_STRIDE,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
