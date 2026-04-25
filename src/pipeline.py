from __future__ import annotations

from pathlib import Path
from typing import Any

from .crop_composer import CropComposer
from .crop_selector import CropSelector
from .detector import ObjectDetector, draw_detections
from .image_io import load_image, safe_stem, save_image
from .main_vlm_client import MainVLMClient
from .metrics import LatencyTracker, build_metrics, save_json
from .prompt_builder import build_final_prompt
from .small_vlm_context import SmallVLMContextExtractor
from .token_estimator import ImageTokenEstimator


def run_pipeline(
    image_path: str | Path,
    question: str,
    config: dict[str, Any],
    out_dir: str | Path,
    call_main_vlm: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(image_path)
    latency = LatencyTracker()

    with latency.measure("load_image"):
        image = load_image(image_path)

    with latency.measure("small_vlm_context"):
        scene_context = SmallVLMContextExtractor(config.get("small_vlm", {})).describe(image, question)

    with latency.measure("detector"):
        detections = ObjectDetector(config.get("detector", {})).detect(image)

    crop_selection_config = dict(config.get("crop_selection", {}))
    crop_selection_config.setdefault(
        "crop_margin_for_area_budget",
        config.get("crop_composer", {}).get("crop_margin", 0.0),
    )

    with latency.measure("crop_selection"):
        selected = CropSelector(crop_selection_config).select(
            detections=detections,
            image_size=image.size,
            question=question,
            scene_context=scene_context,
        )

    estimator = ImageTokenEstimator(config.get("token_estimator", {}))
    composer = CropComposer(config.get("crop_composer", {}))
    budget_dropped_crops = 0
    with latency.measure("crop_composition"):
        composed, crop_metadata = composer.compose(image, selected)
        if crop_selection_config.get("enforce_composed_token_budget", True):
            original_tokens = estimator.estimate(image.size)["approx_image_tokens"]
            max_ratio = float(crop_selection_config.get("max_composed_token_ratio", 1.0))
            while selected and estimator.estimate(composed.size)["approx_image_tokens"] > original_tokens * max_ratio:
                drop_idx = _smallest_effective_crop_index(selected)
                selected.pop(drop_idx)
                budget_dropped_crops += 1
                composed, crop_metadata = composer.compose(image, selected)

    composed_path = save_image(composed, out / f"{stem}_composed.jpg")
    crop_image_paths: list[str] = []
    if config.get("outputs", {}).get("save_individual_crops", True):
        with latency.measure("save_individual_crops"):
            crop_image_paths = _save_individual_crops(image, crop_metadata, out, stem)
            for meta, crop_path in zip(crop_metadata, crop_image_paths):
                meta["crop_image_path"] = crop_path

    detections_path = None
    if config.get("outputs", {}).get("save_detections_visualization", True):
        with latency.measure("detections_visualization"):
            detections_path = draw_detections(image, detections, out / f"{stem}_detections.jpg")

    token_estimates = estimator.compare(image.size, composed.size)

    final_prompt = build_final_prompt(scene_context, crop_metadata, question)
    prompt_path = out / f"{stem}_final_prompt.txt"
    if config.get("outputs", {}).get("save_final_prompt", True):
        prompt_path.write_text(final_prompt, encoding="utf-8")

    main_vlm_result = None
    if call_main_vlm:
        with latency.measure("main_vlm"):
            main_vlm_result = MainVLMClient(config.get("main_vlm", {})).generate(final_prompt, composed)

    metrics = build_metrics(
        image_size=image.size,
        composed_size=composed.size,
        detections=detections,
        selected=selected,
        token_estimates=token_estimates,
        latency=latency.timings,
    )
    metrics["budget_dropped_crops"] = budget_dropped_crops
    metadata = {
        "image_path": str(Path(image_path)),
        "question": question,
        "scene_context": scene_context.to_dict(),
        "detections": detections,
        "selected_crops": selected,
        "crop_metadata": crop_metadata,
        "crop_image_paths": crop_image_paths,
        "composed_image_path": str(composed_path),
        "detections_visualization_path": str(detections_path) if detections_path else None,
        "final_prompt_path": str(prompt_path),
        "final_prompt": final_prompt,
        "main_vlm_result": main_vlm_result,
        "metrics": metrics,
    }
    metadata_path = out / f"{stem}_metadata.json"
    if config.get("outputs", {}).get("save_metadata", True):
        save_json(metadata, metadata_path)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _save_individual_crops(
    image,
    crop_metadata: list[dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> list[str]:
    crop_dir = out_dir / f"{stem}_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for meta in crop_metadata:
        x1, y1, x2, y2 = meta["source_box"]
        label = _safe_label(meta["label"])
        path = crop_dir / f"crop_{meta['index']:02d}_{label}.jpg"
        save_image(image.crop((x1, y1, x2, y2)), path)
        paths.append(str(path))
    return paths


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)[:48]


def _smallest_effective_crop_index(selected: list[dict[str, Any]]) -> int:
    return min(
        range(len(selected)),
        key=lambda idx: int(selected[idx].get("effective_crop_area", selected[idx].get("area", 0))),
    )
