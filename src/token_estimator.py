from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ImageTokenEstimator:
    config: dict[str, Any]

    def estimate(self, image_size: tuple[int, int]) -> dict[str, Any]:
        backend = self.config.get("backend", "qwen_like")
        if backend in {"qwen_like", "patch"}:
            return self._qwen_like(image_size)
        raise ValueError(f"Unsupported token estimator backend: {backend}")

    def compare(self, before_size: tuple[int, int], after_size: tuple[int, int]) -> dict[str, Any]:
        before = self.estimate(before_size)
        after = self.estimate(after_size)
        before_tokens = max(1, before["approx_image_tokens"])
        reduction = 1.0 - (after["approx_image_tokens"] / before_tokens)
        return {
            "before": before,
            "after": after,
            "token_reduction_ratio": round(float(reduction), 6),
        }

    def _qwen_like(self, image_size: tuple[int, int]) -> dict[str, Any]:
        width, height = image_size
        patch = int(self.config.get("patch_size", 28))
        pixels = width * height
        min_pixels = int(self.config.get("min_pixels", 3136))
        max_pixels = int(self.config.get("max_pixels", 12845056))
        clamped_pixels = min(max(pixels, min_pixels), max_pixels)
        scale = (clamped_pixels / max(1, pixels)) ** 0.5
        est_w = max(1, int(width * scale))
        est_h = max(1, int(height * scale))
        tokens = math.ceil(est_w / patch) * math.ceil(est_h / patch)
        return {
            "backend": "qwen_like",
            "image_size": [width, height],
            "effective_size": [est_w, est_h],
            "patch_size": patch,
            "approx_image_tokens": int(tokens),
        }
