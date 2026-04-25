from __future__ import annotations

from functools import lru_cache
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
        if self.backend in {"smolvlm", "smolvlm_256m", "smolvlm_500m"}:
            return self._smolvlm_context(image, question)
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

    def _smolvlm_context(self, image: Image.Image, question: str | None) -> SceneContext:
        model_name = self.config.get("model_name")
        if not model_name:
            model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        if self.backend == "smolvlm_500m":
            model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"
        elif self.backend == "smolvlm_256m":
            model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

        max_new_tokens = int(self.config.get("max_new_tokens", 128))
        device = str(self.config.get("device", "auto"))
        torch_dtype = str(self.config.get("torch_dtype", "auto"))
        attn_implementation = str(self.config.get("attn_implementation", "eager"))
        longest_edge = int(self.config.get("longest_edge", 512))

        try:
            processor, model, resolved_device = _load_smolvlm(
                model_name,
                device,
                torch_dtype,
                attn_implementation,
                longest_edge,
            )
            prompt = _smolvlm_prompt(question)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                text=chat_prompt,
                images=[image.convert("RGB")],
                return_tensors="pt",
            )
            inputs = inputs.to(resolved_device)

            import torch

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text = texts[0].strip() if texts else ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return SceneContext(
                text=text or "SmolVLM did not return scene context.",
                objects=_extract_object_hints(text),
                importance_hints=[
                    "use SmolVLM scene context for global layout",
                    "use detector crops for fine details",
                    "prefer crop evidence for OCR and object attributes",
                ],
                backend="smolvlm",
            )
        except Exception as exc:
            fallback = self._mock_context(image, question)
            fallback.text = f"{fallback.text} SmolVLM failed and mock context was used: {exc}."
            fallback.backend = "smolvlm_fallback"
            return fallback


def _smolvlm_prompt(question: str | None) -> str:
    task = question or "Describe the image."
    return (
        "You are the fast front-stage vision model for a larger VLM pipeline.\n"
        "Summarize the full image in 2-4 concise sentences.\n"
        "Then list 3-8 visually important objects or regions that may help answer the user question.\n"
        "Do not answer the question directly unless the answer is visually obvious.\n\n"
        f"User question:\n{task}"
    )


def _extract_object_hints(text: str) -> list[str]:
    hints: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" -\t")
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if 2 <= len(line) <= 80:
            hints.append(line)
        if len(hints) >= 8:
            break
    return hints


@lru_cache(maxsize=2)
def _load_smolvlm(
    model_name: str,
    device: str,
    torch_dtype: str,
    attn_implementation: str,
    longest_edge: int,
):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    if torch_dtype == "auto":
        if resolved_device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif resolved_device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, torch_dtype.replace("torch.", ""))

    processor_kwargs: dict[str, Any] = {}
    if longest_edge > 0:
        processor_kwargs["size"] = {"longest_edge": longest_edge}
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if resolved_device == "cuda":
        model_kwargs["_attn_implementation"] = attn_implementation
    model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    model = model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device
