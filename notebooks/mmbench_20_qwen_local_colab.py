"""Colab cells for comparing baseline Qwen vs this pruning process.

This notebook runs Qwen locally in the Colab runtime on the first 20 MMBench
samples:

1. baseline: Qwen + original image + MMBench prompt
2. pruned: this repo's crop-canvas preprocessing, then Qwen + crop canvas

It records accuracy, elapsed time, GPU memory, and approximate image-token
reduction. The official MMBench metric is accuracy; the runtime/GPU numbers are
extra local measurements for this experiment.
"""

# %% [markdown]
# # MMBench 20: baseline Qwen vs crop-canvas process
#
# Use a GPU runtime in Colab. This notebook downloads MMBench head-20 samples,
# loads Qwen locally, and compares:
#
# - baseline: original image -> Qwen
# - pruned: original image -> crop-canvas process -> Qwen
#
# The default model is `Qwen/Qwen2.5-VL-3B-Instruct` because it is lighter than
# 7B/72B variants and is more practical for Colab. You can change `MODEL_ID`.

# %%
import os
import subprocess
from pathlib import Path

REPO_DIR = Path("/content/vlm_input_pruning")
REPO_URL = "https://github.com/tjdxo/vlm_input_pruning.git"

if not REPO_DIR.exists():
    token = ""
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")

    clone_url = REPO_URL
    if token:
        clone_url = (
            "https://x-access-token:"
            + token
            + "@github.com/tjdxo/vlm_input_pruning.git"
        )

    subprocess.run(
        ["git", "clone", clone_url, str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull"],
        check=True,
    )

os.chdir(REPO_DIR)
print("cwd:", Path.cwd())

# %%
import sys

# Qwen2.5-VL support requires a recent Transformers. qwen-vl-utils handles
# local image inputs for the chat template.
packages = [
    "-r",
    "requirements.txt",
    "datasets",
    "accelerate",
    "qwen-vl-utils",
    "transformers>=4.57.1",
]

USE_4BIT = False
USE_YOLO_DETECTOR = True

if USE_4BIT:
    packages.append("bitsandbytes")

if USE_YOLO_DETECTOR:
    packages.append("ultralytics")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", *packages],
    check=True,
)

print("deps ready")

# %%
# Main settings.
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SAMPLE_SIZE = 20
MMBENCH_DATASET = "HuggingFaceM4/MMBench_dev"

# Lower this if Colab runs out of memory. Qwen visual tokens use 28x28 patches.
QWEN_MIN_VISUAL_TOKENS = 256
QWEN_MAX_VISUAL_TOKENS = 1280

MAX_NEW_TOKENS = 16
OUT_ROOT = Path("/content/mmbench_20_qwen_compare")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("model:", MODEL_ID)
print("samples:", SAMPLE_SIZE)

# %%
import base64
import io
import json
import math
import re
import threading
import time
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from IPython.display import display
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config import load_config
from src.pipeline import run_pipeline

LETTERS = ["A", "B", "C", "D"]
IMAGE_DIR = OUT_ROOT / "images"
PRUNED_DIR = OUT_ROOT / "pruned"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PRUNED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def to_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def answer_letter(row: dict[str, Any]) -> str:
    value = row.get("answer", None)
    if value is None:
        value = row.get("label", None)
    if isinstance(value, int):
        return LETTERS[value]
    text = clean_text(value).upper()
    if text.isdigit():
        return LETTERS[int(text)]
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse answer label: {value!r}")


def valid_choices(row: dict[str, Any]) -> list[str]:
    return [letter for letter in LETTERS if clean_text(row.get(letter))]


def build_mmbench_prompt(row: dict[str, Any]) -> str:
    hint = clean_text(row.get("hint"))
    question = clean_text(row.get("question"))
    lines = []

    if hint:
        lines.extend(["Context:", hint, ""])

    lines.extend(["Question:", question, "", "Options:"])

    for letter in LETTERS:
        option = clean_text(row.get(letter))
        if option:
            lines.append(f"{letter}. {option}")

    lines.extend(["", "Answer with one uppercase option letter only."])
    return "\n".join(lines)


def extract_choice(text: str, choices: list[str]) -> str:
    valid = "".join(choices)
    upper = text.upper()
    patterns = [
        rf"^\s*([{valid}])\b",
        rf"\banswer\s*[:\-]?\s*([{valid}])\b",
        rf"\boption\s*[:\-]?\s*([{valid}])\b",
        rf"\(([{valid}])\)",
        rf"\b([{valid}])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""


def query_gpu_memory_mb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            values.append(float(line))
    return sum(values) if values else None


def torch_memory_mb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "allocated": None,
            "reserved": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved": torch.cuda.memory_reserved() / (1024 * 1024),
        "peak_allocated": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def measure_step(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        before_torch = torch_memory_mb()
        torch.cuda.reset_peak_memory_stats()
    else:
        before_torch = torch_memory_mb()

    before_gpu = query_gpu_memory_mb()
    peak_gpu = before_gpu
    stop_event = threading.Event()

    def sampler():
        nonlocal peak_gpu
        while not stop_event.is_set():
            current = query_gpu_memory_mb()
            if current is not None:
                peak_gpu = current if peak_gpu is None else max(peak_gpu, current)
            time.sleep(0.2)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stop_event.set()
        thread.join(timeout=1)

    after_gpu = query_gpu_memory_mb()
    after_torch = torch_memory_mb()
    before_alloc = before_torch.get("allocated")
    peak_alloc = after_torch.get("peak_allocated")
    peak_delta = None
    if before_alloc is not None and peak_alloc is not None:
        peak_delta = max(0.0, peak_alloc - before_alloc)

    return {
        "result": result,
        "seconds": elapsed,
        "gpu_mem_before_mb": before_gpu,
        "gpu_mem_peak_mb": peak_gpu,
        "gpu_mem_after_mb": after_gpu,
        "torch_allocated_before_mb": before_alloc,
        "torch_peak_allocated_mb": peak_alloc,
        "torch_peak_delta_mb": peak_delta,
        "torch_peak_reserved_mb": after_torch.get("peak_reserved"),
    }


def sum_latency(metrics: dict[str, Any]) -> float:
    values = metrics.get("latency_seconds", {}).values()
    return float(sum(float(value) for value in values))


print("helpers ready")

# %%
min_pixels = QWEN_MIN_VISUAL_TOKENS * 28 * 28
max_pixels = QWEN_MAX_VISUAL_TOKENS * 28 * 28

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

load_kwargs = {
    "torch_dtype": "auto",
    "device_map": "auto",
}

if USE_4BIT:
    from transformers import BitsAndBytesConfig

    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    **load_kwargs,
)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("loaded:", MODEL_ID)
print("device:", device)
print("gpu memory MB:", query_gpu_memory_mb())

# %%
def run_qwen(prompt: str, image_path: str | Path) -> str:
    path = Path(image_path)
    image_uri = path.resolve().as_uri()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs
    del generated_ids
    del generated_ids_trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text[0].strip()


print("qwen runner ready")

# %%
dataset = load_dataset(MMBENCH_DATASET, split="train")
samples = dataset.select(range(SAMPLE_SIZE))

config = load_config("configs/default.yaml")
config["small_vlm"]["backend"] = "mock"
config["detector"]["backend"] = "yolo" if USE_YOLO_DETECTOR else "dummy"
config["outputs"]["save_individual_crops"] = False

records = []

for idx, row in enumerate(samples):
    row = dict(row)
    sample_id = clean_text(row.get("index")) or str(idx)
    choices = valid_choices(row)
    answer = answer_letter(row)
    prompt = build_mmbench_prompt(row)

    image = to_rgb_image(row["image"])
    image_path = IMAGE_DIR / f"{idx:02d}_{sample_id}.jpg"
    image.save(image_path, quality=92)

    baseline = measure_step(lambda: run_qwen(prompt, image_path))
    baseline_text = baseline["result"]
    baseline_choice = extract_choice(baseline_text, choices)

    preprocess = measure_step(
        lambda image_path=image_path, prompt=prompt, idx=idx: run_pipeline(
            image_path=image_path,
            question=prompt,
            config=config,
            out_dir=PRUNED_DIR / f"{idx:02d}_{sample_id}",
            call_main_vlm=False,
        )
    )
    pruned_result = preprocess["result"]
    pruned_prompt = (
        pruned_result["final_prompt"]
        + "\n\nFor this MMBench question, return only one uppercase option letter."
    )
    composed_path = pruned_result["composed_image_path"]

    pruned = measure_step(lambda: run_qwen(pruned_prompt, composed_path))
    pruned_text = pruned["result"]
    pruned_choice = extract_choice(pruned_text, choices)

    token_estimates = pruned_result["metrics"].get("token_estimates", {})
    before_tokens = token_estimates.get("before", {}).get("approx_image_tokens")
    after_tokens = token_estimates.get("after", {}).get("approx_image_tokens")

    record = {
        "row": idx,
        "sample_id": sample_id,
        "answer": answer,
        "choices": "".join(choices),
        "baseline_choice": baseline_choice,
        "pruned_choice": pruned_choice,
        "baseline_correct": baseline_choice == answer,
        "pruned_correct": pruned_choice == answer,
        "baseline_seconds": baseline["seconds"],
        "pruned_preprocess_seconds": preprocess["seconds"],
        "pruned_pipeline_reported_seconds": sum_latency(pruned_result["metrics"]),
        "pruned_qwen_seconds": pruned["seconds"],
        "pruned_total_seconds": preprocess["seconds"] + pruned["seconds"],
        "baseline_gpu_peak_mb": baseline["gpu_mem_peak_mb"],
        "pruned_preprocess_gpu_peak_mb": preprocess["gpu_mem_peak_mb"],
        "pruned_qwen_gpu_peak_mb": pruned["gpu_mem_peak_mb"],
        "baseline_torch_peak_delta_mb": baseline["torch_peak_delta_mb"],
        "pruned_preprocess_torch_peak_delta_mb": preprocess["torch_peak_delta_mb"],
        "pruned_qwen_torch_peak_delta_mb": pruned["torch_peak_delta_mb"],
        "original_image_tokens_est": before_tokens,
        "composed_image_tokens_est": after_tokens,
        "token_reduction_ratio_est": token_estimates.get("token_reduction_ratio"),
        "num_detections": pruned_result["metrics"].get("num_detections"),
        "num_selected_crops": pruned_result["metrics"].get("num_selected_crops"),
        "image_path": str(image_path),
        "composed_image_path": composed_path,
        "baseline_text": baseline_text,
        "pruned_text": pruned_text,
    }
    records.append(record)

    print(
        f"[{idx + 1:02d}/{SAMPLE_SIZE}]",
        "answer=",
        answer,
        "baseline=",
        baseline_choice or "?",
        "pruned=",
        pruned_choice or "?",
    )

results_df = pd.DataFrame(records)
results_path = OUT_ROOT / "mmbench_20_qwen_results.csv"
results_df.to_csv(results_path, index=False)

print("saved:", results_path)
display(results_df)

# %%
def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


pruned_gpu_peak = results_df[
    ["pruned_preprocess_gpu_peak_mb", "pruned_qwen_gpu_peak_mb"]
].max(axis=1)
pruned_torch_peak_delta = results_df[
    [
        "pruned_preprocess_torch_peak_delta_mb",
        "pruned_qwen_torch_peak_delta_mb",
    ]
].max(axis=1)

summary = pd.DataFrame(
    [
        {
            "variant": "baseline_qwen_original_image",
            "accuracy": mean_or_nan(results_df["baseline_correct"]),
            "avg_seconds": mean_or_nan(results_df["baseline_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(results_df["baseline_gpu_peak_mb"]),
            "avg_torch_peak_delta_mb": mean_or_nan(
                results_df["baseline_torch_peak_delta_mb"]
            ),
            "avg_token_reduction_ratio_est": 0.0,
        },
        {
            "variant": "qwen_with_crop_canvas_process",
            "accuracy": mean_or_nan(results_df["pruned_correct"]),
            "avg_seconds": mean_or_nan(results_df["pruned_total_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(pruned_gpu_peak),
            "avg_torch_peak_delta_mb": mean_or_nan(pruned_torch_peak_delta),
            "avg_token_reduction_ratio_est": mean_or_nan(
                results_df["token_reduction_ratio_est"]
            ),
        },
        {
            "variant": "crop_canvas_preprocess_only",
            "accuracy": math.nan,
            "avg_seconds": mean_or_nan(results_df["pruned_preprocess_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(
                results_df["pruned_preprocess_gpu_peak_mb"]
            ),
            "avg_torch_peak_delta_mb": mean_or_nan(
                results_df["pruned_preprocess_torch_peak_delta_mb"]
            ),
            "avg_token_reduction_ratio_est": mean_or_nan(
                results_df["token_reduction_ratio_est"]
            ),
        },
    ]
)

summary_path = OUT_ROOT / "mmbench_20_qwen_summary.csv"
summary.to_csv(summary_path, index=False)

print("saved:", summary_path)
display(summary)

print()
print("Notes:")
print("- This is head-20 single-pass, not the full official MMBench protocol.")
print("- Official MMBench scoring is accuracy; time/GPU are extra local logs.")
print("- GPU memory includes the loaded Qwen model weights.")

# %%
import shutil

archive_path = shutil.make_archive(
    "/content/mmbench_20_qwen_compare",
    "zip",
    root_dir="/content",
    base_dir="mmbench_20_qwen_compare",
)

from google.colab import files

files.download(archive_path)
