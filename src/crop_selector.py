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
        return [item for item in scored if item["importance_score"] >= min_score][:top_k]


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
