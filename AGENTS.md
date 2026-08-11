# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SubStation AI is a substation equipment instance segmentation pipeline — from dataset management through YOLO training to deployment as a FastAPI inference service with a Tkinter GUI client.

## Essential Commands

### Environment Setup
```bash
# Quick install (recommended)
./scripts/install_conda_env_quick.sh

# Or via requirements
conda create -n substation_seg python=3.10 -y && conda activate substation_seg
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Data Processing (all via `scripts/main.py`)
```bash
# Statistics only
python scripts/main.py --dataset_path /path/to/dataset --mode stats

# Visualize mask overlays (saved to disk, not GUI)
python scripts/main.py --dataset_path /path/to/dataset --mode visualize --samples_per_class 5

# Convert dataset to YOLO polygon segmentation format (train/val split)
python scripts/main.py --dataset_path /path/to/dataset --mode yolo --output_yolo_path ./yolo_out

# Crop individual objects from dataset (preserves JSON annotations, coordinates remapped)
python scripts/main.py --dataset_path /path/to/dataset --mode convert --output_yolo_path ./cropped_out

# Validate YOLO dataset correctness (renders masks onto images)
python scripts/yolo_validator.py --yolo_path ./yolo_out --samples_per_class 10
```

### Training
```bash
# Train with YOLOv8 (recommended baseline)
python scripts/train_yolo.py --dataset_path ./yolo_out --yolo_version yolov8 --model_size s --epochs 300 --batch_size 32 --img_size 640

# Train with YOLO26
python scripts/train_yolo.py --dataset_path ./yolo_out --yolo_version yolo26 --model_size x --epochs 60 --batch_size 6 --img_size 1024

# Validate trained model
python scripts/train_yolo.py --mode val --dataset_path ./yolo_out --weights ./runs/train/exp/weights/best.pt

# Resume training
python scripts/train_yolo.py --dataset_path ./yolo_out --yolo_version yolov8 --resume ./runs/train/exp/weights/last.pt
```

### Inference Service
```bash
# Start the FastAPI server
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000

# Start the GUI client (separate terminal)
cd service && python launch_client.py
```

### Testing
```bash
# All tests
pytest tests/ -v

# Specific categories
pytest tests/test_environment.py -v   # environment/config checks
pytest tests/test_data.py -v          # data loading/format tests
pytest tests/test_training.py -v      # quick training (1 epoch, CPU)
pytest tests/test_validation.py -v    # inference/mask tests (needs network)

# Interactive test runner
python tests/run_tests.py
```

### Docker
```bash
docker-compose up train           # training container
docker-compose --profile tensorboard up tensorboard   # TensorBoard at :6006
```

## Architecture

### Data Processing Pipeline (`scripts/`)
The pipeline is composed of **independent modules**, each handling one responsibility:

- **`data_loader.py`** — Walks the dataset directory tree, loads images + JSON annotations. Supports COCO and LabelMe JSON formats. The directory name is used as the class label.
- **`statistics.py`** — Computes class distribution, sample counts, mask area stats. No image I/O.
- **`visualization.py`** — Renders semi-transparent colored masks over images, saves to disk. No GUI required.
- **`yolo_formatter.py`** — Converts polygon annotations to YOLO segmentation format (normalized coordinates). Produces `images/`, `labels/`, `classes.txt`, and `data.yaml` with train/val split.
- **`dataset_cropper.py`** — Crops individual objects by their bounding boxes, remaps polygon coordinates, preserves JSON format. Supports class mapping (CN→EN), rectangle-first strategy, ignore_classes, and multi-dataset batch processing via `.txt` files.
- **`main.py`** — CLI orchestrator. Parses args, dispatches to the appropriate module for `stats`/`visualize`/`full`/`yolo`/`convert` modes.

### Training System (`scripts/trainers/`)
Uses a **strategy pattern** with `BaseTrainer` (ABC) defining the interface:
- **`yolov8_trainer.py`** — Ultralytics YOLOv8 instance segmentation (recommended baseline)
- **`yolov26_trainer.py`** — Ultralytics YOLO26 instance segmentation (newer, supports n/s/m/l/x)
- **`yolov6_trainer.py`** — Meituan YOLOv6 instance segmentation

`scripts/train_yolo.py` is the unified CLI entrypoint. It auto-generates `data.yaml` from the dataset, prepares train/val splits, and handles class filtering (`--ignore_classes`). Training configs in `train_configs/` provide YAML-based parameter overrides.

Key detail: `train_yolo.py` checks for 3 dataset directory structures — standard YOLO format (`images/train/`, `labels/train/`), yolo_formatter output (`train/images/`, `val/images/`), and category-organized (`class_name/images/`, `class_name/labels/`).

### Inference Service (`service/`)
- **`app.py`** — FastAPI app. Single endpoint `POST /infer`. Accepts base64-encoded image + ROI list, returns detections with mask contours in original image coordinates. Requests/responses automatically logged to `params_log/`.
- **`inference.py`** — YOLO inference core. Crops ROIs, runs model prediction, extracts contours via OpenCV `findContours`, remaps coordinates back to original image space.
- **`schemas.py`** — Pydantic models: `InferenceRequest`, `InferenceResponse`, `ROI`, `DetectionResult` with JSON save/load.
- **`client_app.py`** — Tkinter GUI for interactive ROI drawing and inference visualization.
- **`logger.py`** — Thread-safe logger with date-based log file rotation.

The service caches models by `(weights_path, device, conf_threshold, img_size)` key.

### Dataset Format Requirements
- **Input**: Directory tree where each subdirectory is a class name, containing paired `.jpg`/`.json` files (same name, different extension)
- **JSON**: COCO format (`segmentation`, `image_width`, `image_height`) or LabelMe format (`shapes[].points`, `shapes[].label`, `imageWidth`, `imageHeight`)
- **YOLO output**: Polygon format — each line is `class_id x1 y1 x2 y2 x3 y3 ...` (normalized 0–1)

### Project Conventions
- All paths in `scripts/` are resolved relative to the script location via `os.path.dirname(os.path.abspath(__file__))`
- Thread safety: `dataset_cropper.py` uses a `threading.Lock` for shared stats/counters
- The `scripts/` directory is added to `sys.path` by entry scripts (not by package install)
- Training auto-handles CUDA OOM by halving batch size and retrying
- `.gitignore` excludes: `weights/`, `DATASET/`, `runs*/`, `logs/`, `params_log/`, `*.pt` (indirectly via `weights/`)

## Common Gotchas

### "Label class X exceeds dataset class count Y" error
This is almost always caused by **stale Ultralytics `.cache` files** in the dataset directories. When class definitions change (new classes added, classes re-indexed, or `--ignore_classes` used), the cache files in `train/labels.cache` and `val/labels.cache` are NOT automatically invalidated. Fix:
```bash
find /path/to/yolo_data -name "*.cache" -delete
```
Or use `--force_regenerate_data_config` which now cleans cache files automatically.
