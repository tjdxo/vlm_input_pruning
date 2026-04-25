"""Colab cells for a 20-sample MMBench comparison.

Copy cells from this file into Colab, or open it as a notebook-style script.
The comparison is:

1. baseline: original image + MMBench prompt
2. pruned: scene-context prompt + composed crop canvas from this project

Official MMBench focuses on answer accuracy. Runtime and GPU memory below are
local add-on measurements for this experiment.
"""

# %% [markdown]
# # MMBench 20-sample comparison
#
# Run this after cloning the repository and changing into
# `/content/vlm_input_pruning`.
#
# The VLM must be available through an OpenAI-compatible chat-completions
# endpoint that accepts image_url content. For a remote API, Colab GPU memory
# will not include the remote server's GPU usage.

# %%
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/content/vlm_input_pruning")

if not REPO_DIR.exists():
    raise RuntimeError(
        "Clone the repo first, then run this cell from /content/vlm_input_pruning."
    )

os.chdir(REPO_DIR)
print("cwd:", Path.cwd())

# %%
# Dependencies for this evaluation.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        "requirements.txt",
        "datasets",
    ],
    check=True,
)

# Set True for a more realistic detector path. It downloads YOLO weights.
USE_YOLO_DETECTOR = False

if USE_YOLO_DETECTOR:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ultralytics",
        ],
        check=True,
    )

print("deps ready")

# %%
# Evaluation settings.
SAMPLE_SIZE = 20
RANDOM_SEED = 42
MMBENCH_DATASET = "HuggingFaceM4/MMBench_dev"

# This quick run is single-pass. Full official MMBench commonly uses CircularEval,
# which multiplies the number of VLM calls per question.
USE_CIRCULAR_EVAL = False

# Fill these in for your VLM server/API.
VLM_ENDPOINT_URL = ""
VLM_MODEL = ""
VLM_API_KEY = ""

# If you use Colab Secrets, add OPENAI_API_KEY or VLM_API_KEY there.
try:
    from google.colab import userdata

    VLM_API_KEY = (
        VLM_API_KEY
        or userdata.get("VLM_API_KEY")
        or userdata.get("OPENAI_API_KEY")
        or ""
    )
except Exception:
    VLM_API_KEY = VLM_API_KEY or os.environ.get("VLM_API_KEY", "")

if not VLM_ENDPOINT_URL:
    raise RuntimeError("Set VLM_ENDPOINT_URL before running the benchmark.")

if not VLM_MODEL:
    raise RuntimeError("Set VLM_MODEL before running the benchmark.")

print("VLM config ready:", VLM_MODEL)

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
import requests
from datasets import load_dataset
from IPython.display import display
from PIL import Image

from src.config import load_config
from src.pipeline import run_pipeline

LETTERS = ["A", "B", "C", "D"]
OUT_ROOT = Path("/content/mmbench_20_outputs")
IMAGE_DIR = OUT_ROOT / "images"
BASELINE_DIR = OUT_ROOT / "baseline"
PRUNED_DIR = OUT_ROOT / "pruned"

for path in [IMAGE_DIR, BASELINE_DIR, PRUNED_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def options_from_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    options = []
    for letter in LETTERS:
        text = clean_text(row.get(letter))
        if text:
            options.append((letter, text))
    return options


def answer_from_row(row: dict[str, Any]) -> str:
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


def build_mmbench_prompt(row: dict[str, Any]) -> str:
    hint = clean_text(row.get("hint"))
    question = clean_text(row.get("question"))
    lines = []
    if hint:
        lines.extend(["Context:", hint, ""])
    lines.extend(["Question:", question, "", "Options:"])
    for letter, option in options_from_row(row):
        lines.append(f"{letter}. {option}")
    lines.extend(["", "Answer with one uppercase option letter only."])
    return "\n".join(lines)


def to_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_response_text(raw: dict[str, Any]) -> str:
    choices = raw.get("choices") or []
    if not choices:
        return json.dumps(raw, ensure_ascii=False)[:2000]
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_choice(text: str, valid_choices: list[str]) -> str:
    valid = "".join(valid_choices)
    patterns = [
        rf"^\s*([{valid}])\b",
        rf"\banswer\s*[:\-]?\s*([{valid}])\b",
        rf"\boption\s*[:\-]?\s*([{valid}])\b",
        rf"\(([{valid}])\)",
        rf"\b([{valid}])\b",
    ]
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""


def call_openai_compatible_vlm(prompt: str, image: Image.Image) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"

    payload = {
        "model": VLM_MODEL,
        "temperature": 0,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        VLM_ENDPOINT_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


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
    if not values:
        return None
    return sum(values)


def reset_torch_peak_memory() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def torch_peak_memory_mb() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        return allocated, reserved
    except Exception:
        return None, None


def measure_step(fn):
    has_torch_cuda = reset_torch_peak_memory()
    before = query_gpu_memory_mb()
    peak = before
    stop_event = threading.Event()

    def sampler():
        nonlocal peak
        while not stop_event.is_set():
            current = query_gpu_memory_mb()
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(0.2)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed = time.perf_counter() - started
        stop_event.set()
        thread.join(timeout=1)
    after = query_gpu_memory_mb()
    torch_alloc, torch_reserved = torch_peak_memory_mb()
    return {
        "result": result,
        "seconds": elapsed,
        "gpu_mem_before_mb": before,
        "gpu_mem_after_mb": after,
        "gpu_mem_peak_mb": peak,
        "torch_cuda_measured": has_torch_cuda,
        "torch_peak_allocated_mb": torch_alloc,
        "torch_peak_reserved_mb": torch_reserved,
    }


def total_latency_seconds(metrics: dict[str, Any]) -> float:
    values = metrics.get("latency_seconds", {}).values()
    return float(sum(float(value) for value in values))


print("helpers ready")

# %%
raw_ds = load_dataset(MMBENCH_DATASET, split="train")
sample_ds = raw_ds.shuffle(seed=RANDOM_SEED).select(range(SAMPLE_SIZE))

print("dataset:", MMBENCH_DATASET)
print("sample rows:", len(sample_ds))
print("columns:", sample_ds.column_names)

# %%
config = load_config("configs/default.yaml")
config["small_vlm"]["backend"] = "mock"
config["detector"]["backend"] = "yolo" if USE_YOLO_DETECTOR else "dummy"
config["outputs"]["save_individual_crops"] = False

records = []

for idx, row in enumerate(sample_ds):
    row = dict(row)
    sample_id = clean_text(row.get("index")) or str(idx)
    image = to_rgb_image(row["image"])
    image_path = IMAGE_DIR / f"sample_{idx:03d}_{sample_id}.jpg"
    image.save(image_path, quality=92)

    prompt = build_mmbench_prompt(row)
    valid_choices = [letter for letter, _ in options_from_row(row)]
    answer = answer_from_row(row)

    baseline_measured = measure_step(
        lambda prompt=prompt, image=image: call_openai_compatible_vlm(prompt, image)
    )
    baseline_raw = baseline_measured["result"]
    baseline_text = extract_response_text(baseline_raw)
    baseline_choice = extract_choice(baseline_text, valid_choices)

    preprocess_measured = measure_step(
        lambda image_path=image_path, prompt=prompt, idx=idx: run_pipeline(
            image_path=image_path,
            question=prompt,
            config=config,
            out_dir=PRUNED_DIR / f"sample_{idx:03d}_{sample_id}",
            call_main_vlm=False,
        )
    )
    pruned_result = preprocess_measured["result"]
    composed_image = Image.open(pruned_result["composed_image_path"]).convert("RGB")
    pruned_prompt = pruned_result["final_prompt"]
    pruned_prompt = (
        pruned_prompt
        + "\n\nFor this MMBench question, return only one uppercase option letter."
    )

    pruned_measured = measure_step(
        lambda prompt=pruned_prompt, image=composed_image: call_openai_compatible_vlm(
            prompt,
            image,
        )
    )
    pruned_raw = pruned_measured["result"]
    pruned_text = extract_response_text(pruned_raw)
    pruned_choice = extract_choice(pruned_text, valid_choices)

    metrics = pruned_result["metrics"]
    token_estimates = metrics.get("token_estimates", {})

    record = {
        "row": idx,
        "sample_id": sample_id,
        "answer": answer,
        "valid_choices": "".join(valid_choices),
        "baseline_choice": baseline_choice,
        "pruned_choice": pruned_choice,
        "baseline_correct": baseline_choice == answer,
        "pruned_correct": pruned_choice == answer,
        "baseline_seconds": baseline_measured["seconds"],
        "pruned_preprocess_seconds": preprocess_measured["seconds"],
        "pruned_pipeline_reported_seconds": total_latency_seconds(metrics),
        "pruned_vlm_seconds": pruned_measured["seconds"],
        "pruned_total_seconds": (
            preprocess_measured["seconds"] + pruned_measured["seconds"]
        ),
        "baseline_gpu_peak_mb": baseline_measured["gpu_mem_peak_mb"],
        "pruned_preprocess_gpu_peak_mb": preprocess_measured["gpu_mem_peak_mb"],
        "pruned_vlm_gpu_peak_mb": pruned_measured["gpu_mem_peak_mb"],
        "baseline_torch_peak_allocated_mb": baseline_measured[
            "torch_peak_allocated_mb"
        ],
        "pruned_preprocess_torch_peak_allocated_mb": preprocess_measured[
            "torch_peak_allocated_mb"
        ],
        "pruned_vlm_torch_peak_allocated_mb": pruned_measured[
            "torch_peak_allocated_mb"
        ],
        "original_image_tokens": token_estimates.get("before", {}).get(
            "approx_image_tokens"
        ),
        "composed_image_tokens": token_estimates.get("after", {}).get(
            "approx_image_tokens"
        ),
        "token_reduction_ratio": token_estimates.get("token_reduction_ratio"),
        "num_detections": metrics.get("num_detections"),
        "num_selected_crops": metrics.get("num_selected_crops"),
        "image_path": str(image_path),
        "composed_image_path": pruned_result["composed_image_path"],
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
results_path = OUT_ROOT / "mmbench_20_results.csv"
results_df.to_csv(results_path, index=False)

print("saved:", results_path)
display(results_df)

# %%
def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce")
    return float(values.mean()) if values.notna().any() else math.nan


summary = pd.DataFrame(
    [
        {
            "variant": "baseline_original_image",
            "accuracy": mean_or_nan(results_df["baseline_correct"]),
            "avg_seconds": mean_or_nan(results_df["baseline_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(results_df["baseline_gpu_peak_mb"]),
            "avg_torch_peak_allocated_mb": mean_or_nan(
                results_df["baseline_torch_peak_allocated_mb"]
            ),
        },
        {
            "variant": "pruned_crop_canvas_total",
            "accuracy": mean_or_nan(results_df["pruned_correct"]),
            "avg_seconds": mean_or_nan(results_df["pruned_total_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(
                results_df[
                    [
                        "pruned_preprocess_gpu_peak_mb",
                        "pruned_vlm_gpu_peak_mb",
                    ]
                ].max(axis=1)
            ),
            "avg_torch_peak_allocated_mb": mean_or_nan(
                results_df[
                    [
                        "pruned_preprocess_torch_peak_allocated_mb",
                        "pruned_vlm_torch_peak_allocated_mb",
                    ]
                ].max(axis=1)
            ),
        },
        {
            "variant": "pruned_preprocess_only",
            "accuracy": math.nan,
            "avg_seconds": mean_or_nan(results_df["pruned_preprocess_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(
                results_df["pruned_preprocess_gpu_peak_mb"]
            ),
            "avg_torch_peak_allocated_mb": mean_or_nan(
                results_df["pruned_preprocess_torch_peak_allocated_mb"]
            ),
        },
        {
            "variant": "pruned_vlm_only",
            "accuracy": math.nan,
            "avg_seconds": mean_or_nan(results_df["pruned_vlm_seconds"]),
            "avg_gpu_peak_mb": mean_or_nan(results_df["pruned_vlm_gpu_peak_mb"]),
            "avg_torch_peak_allocated_mb": mean_or_nan(
                results_df["pruned_vlm_torch_peak_allocated_mb"]
            ),
        },
    ]
)

summary_path = OUT_ROOT / "mmbench_20_summary.csv"
summary.to_csv(summary_path, index=False)

print("saved:", summary_path)
display(summary)

print()
print("Notes:")
print("- This is a 20-sample single-pass smoke comparison, not a full official run.")
print("- MMBench official scoring is accuracy-oriented; GPU/time are added here.")
print("- For remote APIs, Colab GPU memory does not show the provider's GPU usage.")

# %%
import shutil

archive_base = Path("/content/mmbench_20_outputs")
archive_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir="/content",
    base_dir="mmbench_20_outputs",
)

from google.colab import files

files.download(archive_path)
