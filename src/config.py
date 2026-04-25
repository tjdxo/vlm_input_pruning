from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "frontend": {"parallel": True},
    "small_vlm": {
        "backend": "mock",
        "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "device": "auto",
        "torch_dtype": "auto",
        "attn_implementation": "eager",
        "longest_edge": 512,
        "max_new_tokens": 128,
    },
    "detector": {"backend": "dummy", "confidence_threshold": 0.25, "max_detections": 12},
    "crop_selection": {
        "top_k": 8,
        "min_score": 0.05,
        "deduplicate_exact_boxes": True,
        "remove_contained_boxes": True,
        "box_containment_tolerance": 0,
        "remove_large_crops": True,
        "max_effective_crop_area_ratio": 0.75,
        "enforce_total_crop_area_lte_original": True,
        "total_crop_area_budget_ratio": 1.0,
        "enforce_composed_token_budget": True,
        "max_composed_token_ratio": 1.0,
        "weights": {
            "detection_score": 0.45,
            "area": 0.20,
            "centrality": 0.15,
            "question_relevance": 0.15,
            "context_mention": 0.05,
        },
    },
    "crop_composer": {
        "max_canvas_width": 1600,
        "max_canvas_height": 1600,
        "tile_padding": 12,
        "crop_margin": 0.08,
        "annotate": True,
        "shrink_to_content": True,
        "background_color": [245, 245, 245],
    },
    "token_estimator": {"backend": "qwen_like", "patch_size": 28},
    "main_vlm": {"backend": "dummy"},
    "outputs": {
        "save_detections_visualization": True,
        "save_individual_crops": True,
        "save_final_prompt": True,
        "save_metadata": True,
    },
}


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, loaded)
