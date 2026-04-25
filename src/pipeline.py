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

    with latency.measure("crop_selection"):
        selected = CropSelector(config.get("crop_selection", {})).select(
            detections=detections,
            image_size=image.size,
            question=question,
            scene_context=scene_context,
        )

    with latency.measure("crop_composition"):
        composed, crop_metadata = CropComposer(config.get("crop_composer", {})).compose(image, selected)

    composed_path = save_image(composed, out / f"{stem}_composed.jpg")
    detections_path = None
    if config.get("outputs", {}).get("save_detections_visualization", True):
        with latency.measure("detections_visualization"):
            detections_path = draw_detections(image, detections, out / f"{stem}_detections.jpg")

    estimator = ImageTokenEstimator(config.get("token_estimator", {}))
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
    metadata = {
        "image_path": str(Path(image_path)),
        "question": question,
        "scene_context": scene_context.to_dict(),
        "detections": detections,
        "selected_crops": selected,
        "crop_metadata": crop_metadata,
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
