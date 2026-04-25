"""Colab helper functions.

The main Colab experience is available in `notebooks/colab_demo.ipynb`.
This module is kept thin so notebook cells can import the same helper if needed.
"""

from pathlib import Path

from src.config import load_config
from src.pipeline import run_pipeline


def run_demo(image_path: str, question: str, out_dir: str = "/content/outputs"):
    config = load_config(Path("configs/default.yaml"))
    return run_pipeline(image_path, question, config, out_dir)
