from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


@dataclass
class MainVLMClient:
    config: dict[str, Any]

    def generate(self, prompt: str, image: Image.Image | None = None) -> dict[str, Any]:
        backend = self.config.get("backend", "dummy")
        if backend in {"dummy", "mock", None}:
            return {
                "backend": "dummy",
                "response": "Dummy main VLM response. Inspect final_prompt.txt and composed_image.jpg.",
            }
        if backend == "openai_compatible":
            return self._openai_compatible(prompt, image)
        raise ValueError(f"Unsupported main VLM backend: {backend}")

    def _openai_compatible(self, prompt: str, image: Image.Image | None) -> dict[str, Any]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("openai_compatible backend requires `pip install requests`.") from exc

        endpoint = self.config.get("endpoint_url")
        if not endpoint:
            raise ValueError("main_vlm.endpoint_url is required for openai_compatible backend.")
        api_key = os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"), "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({"type": "image_url", "image_url": {"url": _to_data_url(image)}})
        payload = {
            "model": self.config.get("model", "qwen-vl-placeholder"),
            "messages": [{"role": "user", "content": content}],
        }
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=int(self.config.get("timeout_seconds", 120)),
        )
        response.raise_for_status()
        return {"backend": "openai_compatible", "raw": response.json()}


def _to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
