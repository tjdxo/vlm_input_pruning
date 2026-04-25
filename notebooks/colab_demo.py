"""Thin Colab demo wrapper.

Upload or clone this repository, install requirements, then run:

    !python scripts/run_single_image.py \
      --image /content/sample.jpg \
      --question "Describe the important objects while preserving fine details." \
      --out-dir /content/outputs
"""

from pathlib import Path

from src.config import load_config
from src.pipeline import run_pipeline


def run_demo(image_path: str, question: str, out_dir: str = "/content/outputs"):
    config = load_config(Path("configs/default.yaml"))
    return run_pipeline(image_path, question, config, out_dir)
