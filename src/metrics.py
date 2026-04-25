from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class LatencyTracker:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - start, 6)


def build_metrics(
    image_size: tuple[int, int],
    composed_size: tuple[int, int],
    detections: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    token_estimates: dict[str, Any],
    latency: dict[str, float],
) -> dict[str, Any]:
    return {
        "original_image_size": list(image_size),
        "composed_image_size": list(composed_size),
        "num_detections": len(detections),
        "num_selected_crops": len(selected),
        "token_estimates": token_estimates,
        "latency_seconds": latency,
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out
