from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .detector import Detection


@dataclass
class CropComposer:
    config: dict[str, Any]

    def compose(self, image: Image.Image, selected: list[Detection]) -> tuple[Image.Image, list[dict[str, Any]]]:
        if not selected:
            return self._empty_canvas(image), []

        max_w = int(self.config.get("max_canvas_width", 1600))
        max_h = int(self.config.get("max_canvas_height", 1600))
        padding = int(self.config.get("tile_padding", 12))
        bg = tuple(self.config.get("background_color", [245, 245, 245]))
        annotate = bool(self.config.get("annotate", True))

        crops = [self._crop_with_margin(image, det) for det in selected]
        cols = max(1, min(len(crops), math.ceil(math.sqrt(len(crops)))))
        tile_w = max(1, (max_w - padding * (cols + 1)) // cols)
        rows = math.ceil(len(crops) / cols)
        tile_h = max(1, (max_h - padding * (rows + 1)) // rows)
        canvas = Image.new("RGB", (max_w, max_h), bg)
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        metadata: list[dict[str, Any]] = []

        for idx, (det, crop, source_box) in enumerate(zip(selected, crops, [c[1] for c in crops])):
            crop_img = crop[0]
            row, col = divmod(idx, cols)
            x = padding + col * (tile_w + padding)
            y = padding + row * (tile_h + padding)
            label_h = 18 if annotate else 0
            resized = _fit_inside(crop_img, tile_w, max(1, tile_h - label_h))
            canvas.paste(resized, (x, y + label_h))
            if annotate:
                label = f"{idx + 1}. {det['label']} {det['score']:.2f}"
                draw.text((x, y), label[:80], fill=(20, 20, 20), font=font)
            metadata.append(
                {
                    "index": idx + 1,
                    "label": det["label"],
                    "score": det["score"],
                    "importance_score": det.get("importance_score"),
                    "source_box": source_box,
                    "original_box": det["box"],
                    "canvas_box": [x, y + label_h, x + resized.width, y + label_h + resized.height],
                    "display_size": [resized.width, resized.height],
                }
            )
        return canvas, metadata

    def _crop_with_margin(self, image: Image.Image, det: Detection) -> tuple[Image.Image, list[int]]:
        width, height = image.size
        x1, y1, x2, y2 = det["box"]
        margin_ratio = float(self.config.get("crop_margin", 0.08))
        mx = int((x2 - x1) * margin_ratio)
        my = int((y2 - y1) * margin_ratio)
        box = [max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)]
        return image.crop(box), box

    def _empty_canvas(self, image: Image.Image) -> Image.Image:
        max_w = int(self.config.get("max_canvas_width", 1600))
        max_h = int(self.config.get("max_canvas_height", 1600))
        resized = _fit_inside(image, max_w, max_h)
        canvas = Image.new("RGB", (max_w, max_h), tuple(self.config.get("background_color", [245, 245, 245])))
        x = (max_w - resized.width) // 2
        y = (max_h - resized.height) // 2
        canvas.paste(resized, (x, y))
        return canvas


def _fit_inside(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / image.width, max_h / image.height, 1.0)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    if new_size == image.size:
        return image.copy()
    return image.resize(new_size, Image.Resampling.LANCZOS)
