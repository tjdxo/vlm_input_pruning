from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_pipeline
from src.token_estimator import ImageTokenEstimator
from src.image_io import load_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark variants over an image folder.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--out-dir", default=ROOT / "examples" / "outputs" / "benchmark", type=Path)
    parser.add_argument("--detector-backend", choices=["dummy", "yolo"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.detector_backend:
        config["detector"]["backend"] = args.detector_backend
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    jsonl_path = args.out_dir / "benchmark_results.jsonl"
    csv_path = args.out_dir / "benchmark_results.csv"
    estimator = ImageTokenEstimator(config.get("token_estimator", {}))

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for image_path in tqdm(images, desc="benchmark"):
            image = load_image(image_path)
            original_tokens = estimator.estimate(image.size)["approx_image_tokens"]
            result = run_pipeline(image_path, args.question, config, args.out_dir / image_path.stem)
            after_tokens = result["metrics"]["token_estimates"]["after"]["approx_image_tokens"]
            variants = [
                {
                    "variant": "A_original_image_only",
                    "approx_image_tokens": original_tokens,
                    "has_context": False,
                    "has_crop_canvas": False,
                },
                {
                    "variant": "B_detector_crop_canvas_only",
                    "approx_image_tokens": after_tokens,
                    "has_context": False,
                    "has_crop_canvas": True,
                },
                {
                    "variant": "C_small_vlm_context_plus_crop_canvas",
                    "approx_image_tokens": after_tokens,
                    "has_context": True,
                    "has_crop_canvas": True,
                },
                {
                    "variant": "proposed_context_plus_selected_highres_crops",
                    "approx_image_tokens": after_tokens,
                    "has_context": True,
                    "has_crop_canvas": True,
                },
            ]
            for variant in variants:
                row = {
                    "image": str(image_path),
                    "question": args.question,
                    "num_detections": result["metrics"]["num_detections"],
                    "num_selected_crops": result["metrics"]["num_selected_crops"],
                    "token_reduction_ratio": result["metrics"]["token_estimates"]["token_reduction_ratio"],
                    "latency_total": round(sum(result["metrics"]["latency_seconds"].values()), 6),
                    **variant,
                }
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"jsonl: {jsonl_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
