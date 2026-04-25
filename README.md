# VLM Input Pruning

Colab-first Python prototype for reducing visual-token cost on high-resolution VLM inputs without simply resizing away fine details.

The pipeline compresses global scene context into text with a small front-stage VLM, keeps important regions as high-resolution crops with a detector, composes those crops into one canvas, and sends `context text + crop canvas + original question` to a main VLM interface.

## Why This Helps

High-resolution images can create many visual tokens, increasing GPU memory use and latency. A plain resize lowers token count but can destroy small text, object attributes, and other fine details.

This project uses a hybrid representation:

- global background and scene layout are represented as text from a small VLM or mock context extractor
- important objects are preserved as high-resolution crops from the original image
- the final VLM sees a smaller composed crop image instead of the full original image
- metrics record approximate image tokens and latency so baselines can be compared

The current implementation is intentionally lightweight: mock mode runs end-to-end without model downloads, and YOLO / HF captioning are optional lazy-loaded backends.

## Connection To `inmare/jjs`

The referenced `jjs` repository appears to explore Qwen3-VL, YOLO crop augmentation, Qwen-only versus YOLO+Qwen comparisons, 2-stage VLM structures, SmolVLM experiments, llama.cpp / GGUF, OpenAI-compatible chat completions, and benchmark logging.

This repo keeps those ideas but reorganizes them as a Colab-first Python package:

- Qwen-only vs crop-assisted comparison maps to `scripts/run_benchmark.py`
- YOLO crop experiments map to `src/detector.py`, `src/crop_selector.py`, and `src/crop_composer.py`
- 2-stage VLM maps to `src/small_vlm_context.py` plus `src/main_vlm_client.py`
- prompt/token/latency logging maps to `src/prompt_builder.py`, `src/token_estimator.py`, and `src/metrics.py`
- llama.cpp / OpenAI-compatible serving is represented as an adapter path in `src/main_vlm_client.py`

## Project Layout

```text
vlm_input_pruning/
  configs/default.yaml
  notebooks/colab_demo.py
  scripts/run_single_image.py
  scripts/run_benchmark.py
  src/
    image_io.py
    small_vlm_context.py
    detector.py
    crop_selector.py
    crop_composer.py
    token_estimator.py
    main_vlm_client.py
    prompt_builder.py
    pipeline.py
    metrics.py
```

## Colab Quick Start

```python
!git clone <repo_url>
%cd vlm_input_pruning
!pip install -r requirements.txt

!python scripts/run_single_image.py \
  --image /content/sample.jpg \
  --question "Describe the important objects while preserving fine details." \
  --out-dir /content/outputs
```

Mock mode is the default. It does not download YOLO, SmolVLM, Qwen, or any other heavy model.

Outputs:

- `*_composed.jpg`: tiled crop canvas
- `*_detections.jpg`: detection visualization
- `*_final_prompt.txt`: prompt for the main VLM
- `*_metadata.json`: detections, selected crops, token estimates, latency breakdown

## Optional YOLO Mode

Install Ultralytics in Colab:

```python
!pip install ultralytics
```

Run with YOLO:

```python
!python scripts/run_single_image.py \
  --image /content/sample.jpg \
  --question "What text is visible in the important objects?" \
  --detector-backend yolo \
  --out-dir /content/outputs
```

The default YOLO model is `yolov8n.pt`, configured in `configs/default.yaml`.

## Optional Small VLM Caption Mode

Mock context is the default. To try a Hugging Face image captioning backend:

```python
!pip install transformers accelerate torch
!python scripts/run_single_image.py \
  --image /content/sample.jpg \
  --question "Describe the important objects." \
  --small-vlm-backend hf_caption \
  --out-dir /content/outputs
```

No Hugging Face token is hardcoded. If a model needs authentication, configure it through the Colab environment.

## Benchmark

```python
!python scripts/run_benchmark.py \
  --image-dir /content/images \
  --question "Answer using fine visual details where relevant." \
  --out-dir /content/benchmark_outputs
```

The benchmark writes JSONL and CSV rows for:

- `A_original_image_only`
- `B_detector_crop_canvas_only`
- `C_small_vlm_context_plus_crop_canvas`
- `proposed_context_plus_selected_highres_crops`

The MVP records approximate image tokens rather than model-exact tokenization. This is enough for relative comparisons across original image input and composed crop canvas input.

## Main VLM Adapter

`src/main_vlm_client.py` supports:

- `dummy`: default, saves prompt/image only
- `openai_compatible`: posts text plus base64 image to an OpenAI-compatible chat completions endpoint

This leaves room to attach Qwen3-VL, llama.cpp `llama-server`, or another OpenAI-compatible VLM later without changing the pipeline.

## Next Experiments

- Replace heuristic label relevance with CLIP or SigLIP image-text relevance
- Add question-aware crop expansion for OCR and small-object queries
- Add SmolVLM or another small VLM for structured scene context and object hints
- Compare answer quality for full image, resized image, crop-only, and context+crop inputs
- Add exact token accounting for a chosen Qwen-VL processor
- Add ablations for crop count, canvas size, crop margin, and annotation on/off
