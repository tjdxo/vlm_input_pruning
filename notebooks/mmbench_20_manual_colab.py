"""Colab cells for manually inspecting 20 MMBench samples.

This notebook does not call any VLM endpoint. It downloads the first 20 rows
from MMBench, saves images/prompts/answers, and creates CSV templates where
you can manually enter baseline and pruned-model predictions.
"""

# %% [markdown]
# # MMBench 20-sample manual set
#
# This is for manual testing. It downloads 20 MMBench examples and exports:
#
# - `/content/mmbench_20_manual/images/*.jpg`
# - `/content/mmbench_20_manual/mmbench_20_questions.csv`
# - `/content/mmbench_20_manual/mmbench_20_answer_key.csv`
# - `/content/mmbench_20_manual/mmbench_20_manual_results_template.csv`
#
# MMBench official scoring is answer accuracy. GPU usage and elapsed time are
# not official MMBench metrics; if you test manually, record them separately.

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

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "datasets",
        "pandas",
        "pillow",
    ],
    check=True,
)

print("deps ready")

# %%
import json
from typing import Any

import pandas as pd
from datasets import load_dataset
from IPython.display import Markdown, display
from PIL import Image

DATASET_NAME = "HuggingFaceM4/MMBench_dev"
SAMPLE_SIZE = 20
OUT_DIR = Path("/content/mmbench_20_manual")
IMAGE_DIR = OUT_DIR / "images"
PROMPT_DIR = OUT_DIR / "prompts"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)


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
        import io

        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def answer_letter(row: dict[str, Any]) -> str:
    value = row.get("answer", None)
    if value is None:
        value = row.get("label", None)
    if isinstance(value, int):
        return "ABCD"[value]
    text = clean_text(value).upper()
    if text.isdigit():
        return "ABCD"[int(text)]
    for letter in "ABCD":
        if letter in text:
            return letter
    return text


def build_prompt(row: dict[str, Any], hide_answer: bool = True) -> str:
    lines = []
    hint = clean_text(row.get("hint"))
    question = clean_text(row.get("question"))

    if hint:
        lines.extend(["Context:", hint, ""])

    lines.extend(["Question:", question, "", "Options:"])

    for letter in "ABCD":
        option = clean_text(row.get(letter))
        if option:
            lines.append(f"{letter}. {option}")

    lines.extend(["", "Answer with one uppercase option letter only."])

    if not hide_answer:
        lines.extend(["", f"Answer key: {answer_letter(row)}"])

    return "\n".join(lines)


dataset = load_dataset(DATASET_NAME, split="train")
samples = dataset.select(range(SAMPLE_SIZE))

question_rows = []
answer_rows = []
template_rows = []

for i, row in enumerate(samples):
    row = dict(row)
    sample_id = clean_text(row.get("index")) or str(i)
    image = to_rgb_image(row["image"])

    image_path = IMAGE_DIR / f"{i:02d}_{sample_id}.jpg"
    prompt_path = PROMPT_DIR / f"{i:02d}_{sample_id}.txt"
    image.save(image_path, quality=92)

    prompt = build_prompt(row, hide_answer=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    question_rows.append(
        {
            "row": i,
            "sample_id": sample_id,
            "image_path": str(image_path),
            "prompt_path": str(prompt_path),
            "question": clean_text(row.get("question")),
            "hint": clean_text(row.get("hint")),
            "A": clean_text(row.get("A")),
            "B": clean_text(row.get("B")),
            "C": clean_text(row.get("C")),
            "D": clean_text(row.get("D")),
            "category": clean_text(row.get("category")),
            "l2_category": clean_text(row.get("l2-category")),
            "source": clean_text(row.get("source")),
            "prompt": prompt,
        }
    )
    answer_rows.append(
        {
            "row": i,
            "sample_id": sample_id,
            "answer": answer_letter(row),
        }
    )
    template_rows.append(
        {
            "row": i,
            "sample_id": sample_id,
            "answer": answer_letter(row),
            "baseline_original_answer": "",
            "pruned_answer": "",
            "baseline_seconds": "",
            "pruned_seconds": "",
            "baseline_gpu_peak_mb": "",
            "pruned_gpu_peak_mb": "",
            "notes": "",
        }
    )

questions_df = pd.DataFrame(question_rows)
answers_df = pd.DataFrame(answer_rows)
template_df = pd.DataFrame(template_rows)

questions_path = OUT_DIR / "mmbench_20_questions.csv"
answers_path = OUT_DIR / "mmbench_20_answer_key.csv"
template_path = OUT_DIR / "mmbench_20_manual_results_template.csv"

questions_df.to_csv(questions_path, index=False)
answers_df.to_csv(answers_path, index=False)
template_df.to_csv(template_path, index=False)

metadata = {
    "dataset": DATASET_NAME,
    "sample_size": SAMPLE_SIZE,
    "selection": "head",
    "questions_csv": str(questions_path),
    "answer_key_csv": str(answers_path),
    "manual_results_template_csv": str(template_path),
}
(OUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("saved:", OUT_DIR)
display(questions_df[["row", "sample_id", "question", "A", "B", "C", "D"]])

# %%
for i in range(SAMPLE_SIZE):
    row = questions_df.iloc[i]
    display(Markdown(f"## Sample {i:02d} / id {row['sample_id']}"))
    display(Image.open(row["image_path"]))
    display(Markdown("```text\n" + row["prompt"] + "\n```"))

# %%
# After manually filling baseline_original_answer and pruned_answer in the
# template CSV, upload or edit it at:
print("manual results template:")
print(template_path)

display(template_df)

# %%
import shutil

archive_path = shutil.make_archive(
    "/content/mmbench_20_manual",
    "zip",
    root_dir="/content",
    base_dir="mmbench_20_manual",
)

from google.colab import files

files.download(archive_path)
