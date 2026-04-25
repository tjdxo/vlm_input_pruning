from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .detector import Detection
from .small_vlm_context import SceneContext


@dataclass
class CropSelector:
    config: dict[str, Any]

    def select(
        self,
        detections: list[Detection],
        image_size: tuple[int, int],
        question: str,
        scene_context: SceneContext,
    ) -> list[Detection]:
        if not detections:
            return []
        detections = self._pre_filter(detections, image_size)
        if not detections:
            return []
        width, height = image_size
        image_area = max(1, width * height)
        question_terms = _terms(question)
        context_terms = _terms(scene_context.text + " " + " ".join(scene_context.objects))
        weights = self.config.get("weights", {})
        scored = []
        for det in detections:
            box = det["box"]
            area_score = min(1.0, det["area"] / image_area * 4.0)
            centrality = _centrality(box, image_size)
            label_terms = _terms(det["label"].replace("_", " "))
            relevance = 1.0 if question_terms.intersection(label_terms) else 0.0
            context_mention = 1.0 if context_terms.intersection(label_terms) else 0.0
            importance = (
                weights.get("detection_score", 0.45) * float(det["score"])
                + weights.get("area", 0.20) * area_score
                + weights.get("centrality", 0.15) * centrality
                + weights.get("question_relevance", 0.15) * relevance
                + weights.get("context_mention", 0.05) * context_mention
            )
            enriched = dict(det)
            enriched["importance_score"] = round(float(importance), 6)
            enriched["selection_features"] = {
                "area_score": round(float(area_score), 6),
                "centrality": round(float(centrality), 6),
                "question_relevance": relevance,
                "context_mention": context_mention,
            }
            scored.append(enriched)
        min_score = float(self.config.get("min_score", 0.05))
        top_k = int(self.config.get("top_k", 8))
        scored.sort(key=lambda item: item["importance_score"], reverse=True)
        selected = [item for item in scored if item["importance_score"] >= min_score][:top_k]
        return self._enforce_total_crop_area_budget(selected, image_area)

    def _pre_filter(self, detections: list[Detection], image_size: tuple[int, int]) -> list[Detection]:
        filtered = [dict(det) for det in detections]
        if self.config.get("deduplicate_exact_boxes", True):
            filtered = _deduplicate_exact_boxes(filtered)
        if self.config.get("remove_large_crops", True):
            filtered = self._remove_large_effective_crops(filtered, image_size)
        if self.config.get("remove_contained_boxes", True):
            filtered = _remove_contained_boxes(
                filtered,
                tolerance=int(self.config.get("box_containment_tolerance", 0)),
            )
        return filtered

    def _remove_large_effective_crops(
        self,
        detections: list[Detection],
        image_size: tuple[int, int],
    ) -> list[Detection]:
        image_area = max(1, image_size[0] * image_size[1])
        max_ratio = float(self.config.get("max_effective_crop_area_ratio", 0.75))
        kept: list[Detection] = []
        for det in detections:
            enriched = dict(det)
            effective_box = _effective_crop_box(
                enriched["box"],
                image_size,
                float(self.config.get("crop_margin_for_area_budget", 0.0)),
            )
            effective_area = _box_area(effective_box)
            enriched["effective_crop_box"] = effective_box
            enriched["effective_crop_area"] = effective_area
            enriched["effective_crop_area_ratio"] = round(effective_area / image_area, 6)
            if effective_area / image_area < max_ratio:
                kept.append(enriched)
        return kept

    def _enforce_total_crop_area_budget(
        self,
        selected: list[Detection],
        image_area: int,
    ) -> list[Detection]:
        if not self.config.get("enforce_total_crop_area_lte_original", True):
            return selected
        budget = image_area * float(self.config.get("total_crop_area_budget_ratio", 1.0))
        remaining = list(selected)

        def effective_area(det: Detection) -> int:
            return int(det.get("effective_crop_area", det.get("area", 0)))

        while remaining and sum(effective_area(det) for det in remaining) > budget:
            smallest_idx = min(range(len(remaining)), key=lambda idx: effective_area(remaining[idx]))
            remaining.pop(smallest_idx)
        return remaining


def _terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2}


def _centrality(box: list[int], image_size: tuple[int, int]) -> float:
    width, height = image_size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(cx - width / 2.0) / max(1.0, width / 2.0)
    dy = abs(cy - height / 2.0) / max(1.0, height / 2.0)
    return max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5) / 1.41421356237)


def _deduplicate_exact_boxes(detections: list[Detection]) -> list[Detection]:
    seen: set[tuple[int, int, int, int]] = set()
    kept: list[Detection] = []
    for det in detections:
        box = _box_key(det["box"])
        if box in seen:
            continue
        seen.add(box)
        kept.append(det)
    return kept


def _remove_contained_boxes(detections: list[Detection], tolerance: int = 0) -> list[Detection]:
    kept: list[Detection] = []
    for idx, det in enumerate(detections):
        box = det["box"]
        contained = False
        for other_idx, other in enumerate(detections):
            if idx == other_idx:
                continue
            if _is_strictly_contained(box, other["box"], tolerance):
                contained = True
                break
        if not contained:
            kept.append(det)
    return kept


def _is_strictly_contained(inner: list[int], outer: list[int], tolerance: int = 0) -> bool:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    same = _box_key(inner) == _box_key(outer)
    if same:
        return False
    return (
        ix1 >= ox1 - tolerance
        and iy1 >= oy1 - tolerance
        and ix2 <= ox2 + tolerance
        and iy2 <= oy2 + tolerance
    )


def _effective_crop_box(
    box: list[int],
    image_size: tuple[int, int],
    margin_ratio: float,
) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    mx = int((x2 - x1) * margin_ratio)
    my = int((y2 - y1) * margin_ratio)
    return [
        max(0, x1 - mx),
        max(0, y1 - my),
        min(width, x2 + mx),
        min(height, y2 + my),
    ]


def _box_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_key(box: list[int]) -> tuple[int, int, int, int]:
    return tuple(int(round(value)) for value in box)
