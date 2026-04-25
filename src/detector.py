from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Detection = dict[str, Any]


@dataclass
class ObjectDetector:
    config: dict[str, Any]

    def detect(self, image: Image.Image) -> list[Detection]:
        backend = self.config.get("backend", "dummy")
        if backend in {"dummy", "mock", None}:
            return self._dummy_detect(image)
        if backend == "yolo":
            return self._yolo_detect(image)
        raise ValueError(f"Unsupported detector backend: {backend}")

    def _dummy_detect(self, image: Image.Image) -> list[Detection]:
        width, height = image.size
        boxes = [
            ("center_region", 0.70, [0.25 * width, 0.20 * height, 0.75 * width, 0.75 * height]),
            ("upper_left_detail", 0.55, [0.05 * width, 0.05 * height, 0.38 * width, 0.38 * height]),
            ("lower_right_detail", 0.50, [0.58 * width, 0.55 * height, 0.95 * width, 0.92 * height]),
        ]
        detections = []
        for label, score, box in boxes[: int(self.config.get("max_detections", 12))]:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            detections.append(_make_detection(label, score, [x1, y1, x2, y2]))
        return detections

    def _yolo_detect(self, image: Image.Image) -> list[Detection]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO backend requires `pip install ultralytics`.") from exc

        model_name = self.config.get("yolo_model", "yolov8n.pt")
        conf = float(self.config.get("confidence_threshold", 0.25))
        max_det = int(self.config.get("max_detections", 12))
        model = YOLO(model_name)
        results = model.predict(image, conf=conf, max_det=max_det, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections
        names = results[0].names
        for box in results[0].boxes:
            xyxy = [int(round(v)) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = str(names.get(cls_id, cls_id))
            detections.append(_make_detection(label, score, xyxy))
        return detections


def _make_detection(label: str, score: float, box: list[int]) -> Detection:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return {"label": label, "score": float(score), "box": box, "area": int(area)}


def draw_detections(image: Image.Image, detections: list[Detection], out_path: str | Path) -> Path:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 30), width=3)
        text_w, text_h = _text_size(draw, label, font)
        bbox = (x1, y1, x1 + text_w + 4, y1 + text_h + 4)
        draw.rectangle(bbox, fill=(255, 80, 30))
        draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255), font=font)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out)
    return out


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textlength"):
        try:
            return int(draw.textlength(text, font=font)), 12
        except Exception:
            pass
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return font.getsize(text)
