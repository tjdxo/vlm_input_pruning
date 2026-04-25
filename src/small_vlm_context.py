from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class SceneContext:
    text: str
    objects: list[str] = field(default_factory=list)
    importance_hints: list[str] = field(default_factory=list)
    backend: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "objects": self.objects,
            "importance_hints": self.importance_hints,
            "backend": self.backend,
        }


class SmallVLMContextExtractor:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.backend = self.config.get("backend", "mock")

    def describe(self, image: Image.Image, question: str | None = None) -> SceneContext:
        if self.backend in {"mock", "dummy", None}:
            return self._mock_context(image, question)
        if self.backend in {"hf_caption", "transformers"}:
            return self._hf_caption(image, question)
        raise ValueError(f"Unsupported small VLM backend: {self.backend}")

    def _mock_context(self, image: Image.Image, question: str | None) -> SceneContext:
        width, height = image.size
        aspect = "wide" if width > height else "tall" if height > width else "square"
        text = (
            f"Mock scene context: the original image is a {width}x{height} {aspect} RGB image. "
            "Use the selected high-resolution crops for fine visual details and this text as "
            "a compact substitute for global scene context."
        )
        hints = ["preserve small text", "inspect high-confidence objects", "use crop labels as anchors"]
        return SceneContext(text=text, objects=[], importance_hints=hints, backend="mock")

    def _hf_caption(self, image: Image.Image, question: str | None) -> SceneContext:
        try:
            from transformers import pipeline
        except ImportError as exc:
            fallback = self._mock_context(image, question)
            fallback.text = f"{fallback.text} HF caption backend unavailable: {exc}."
            return fallback

        model_name = self.config.get("model_name") or "Salesforce/blip-image-captioning-base"
        device = 0 if self.config.get("device", "auto") != "cpu" else -1
        try:
            captioner = pipeline("image-to-text", model=model_name, device=device)
            result = captioner(image, max_new_tokens=int(self.config.get("max_new_tokens", 96)))
            text = result[0].get("generated_text", "").strip()
            return SceneContext(text=text or "No caption generated.", backend="hf_caption")
        except Exception as exc:
            fallback = self._mock_context(image, question)
            fallback.text = f"{fallback.text} HF caption failed and mock context was used: {exc}."
            return fallback
