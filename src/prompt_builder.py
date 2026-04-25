from __future__ import annotations

from typing import Any

from .small_vlm_context import SceneContext


def build_final_prompt(
    scene_context: SceneContext,
    crop_metadata: list[dict[str, Any]],
    question: str,
) -> str:
    crop_lines = []
    for crop in crop_metadata:
        crop_lines.append(
            f"- Crop {crop['index']}: label={crop['label']}, score={crop['score']:.3f}, "
            f"source_box={crop['source_box']}"
        )
    crops_text = "\n".join(crop_lines) if crop_lines else "- No detector crops were selected."
    return (
        "The original high-resolution image was compressed into selected high-resolution crops.\n"
        "Use the scene context for global background and the crop canvas for fine visual details.\n"
        "Do not assume that every object from the original image is visible in the crop canvas; "
        "when needed, use the scene context as a lossy summary of the full scene.\n\n"
        f"Scene context:\n{scene_context.text}\n\n"
        f"Selected crop metadata:\n{crops_text}\n\n"
        f"User question:\n{question}\n\n"
        "Answer using the crop image as the primary evidence for detailed visual claims."
    )
