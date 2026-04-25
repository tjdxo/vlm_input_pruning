from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid VLM input pruning on one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--out-dir", default=ROOT / "examples" / "outputs", type=Path)
    parser.add_argument("--detector-backend", choices=["dummy", "yolo"], default=None)
    parser.add_argument("--small-vlm-backend", choices=["mock", "hf_caption"], default=None)
    parser.add_argument("--call-main-vlm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.detector_backend:
        config["detector"]["backend"] = args.detector_backend
    if args.small_vlm_backend:
        config["small_vlm"]["backend"] = args.small_vlm_backend
    result = run_pipeline(
        image_path=args.image,
        question=args.question,
        config=config,
        out_dir=args.out_dir,
        call_main_vlm=args.call_main_vlm,
    )
    metrics = result["metrics"]
    print(f"composed_image: {result['composed_image_path']}")
    print(f"final_prompt: {result['final_prompt_path']}")
    print(f"metadata: {result['metadata_path']}")
    print(f"detections_visualization: {result['detections_visualization_path']}")
    print(f"approx_token_reduction: {metrics['token_estimates']['token_reduction_ratio']:.3f}")
    print(f"latency_seconds: {metrics['latency_seconds']}")


if __name__ == "__main__":
    main()
