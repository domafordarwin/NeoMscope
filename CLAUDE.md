# NeoMscope — Onion Cell Division Detector

## Project Overview
양파 표피세포 이미지에서 세포 분열 단계(Inter / Pro / Meta / Ana / Telo) 5단계를 자동 분류·계수하는 시스템. 과학전람회 출품용.

**v2.0 (진행 중, 2026-05-01~)**: PC + Mask R-CNN(TF1.x) → **Pi 5 + AI HAT+ 2 (Hailo-10H) + YOLOv11-det (PyTorch/Ultralytics)** 마이그레이션.

- **Domain**: Computer Vision / Biological Imaging
- **현행 모델**: YOLOv11s-det (학습 예정), 백업 YOLOv8s-det
- **레거시 모델**: Matterport Mask R-CNN — `legacy/` 보존, 부트스트랩 라벨링 입력으로 활용
- **Classes**: 5 division stages (no background — YOLO det 표준)
- **Input**: 640×640 square (Hailo-10H 컴파일 타겟)
- **bkit Level**: Dynamic

## Tech Stack

### v2.0 (현행)
- **학습**: PyTorch + Ultralytics 8.3+ on Kaggle (T4/P100 16GB VRAM)
- **변환**: ONNX (opset 17) → Hailo Dataflow Compiler 4.x+ (WSL2 Ubuntu)
- **추론**: HailoRT 4.x + GStreamer on Pi 5 + AI HAT+ 2 (Hailo-10H, 40 INT4 / 20 INT8 TOPS, 8GB LPDDR4X)
- **개발 PC**: Windows 11 + Quadro P2000 4GB (코드/ONNX/HEF용 — 학습엔 부족)
- **Tooling**: ruff + black + pytest + uv (Python 3.11)

### 레거시 (`legacy/`)
- TF 1.15 + Keras 2.2.5 + Matterport mrcnn (Python 3.7)
- conda env `neomscope-legacy` — `legacy/README.md` 참조

## Repository Layout

```
.
├── pyproject.toml                # Python 3.11+ deps + ruff/black/pytest
├── setup.sh                      # Pi 5 bootstrap (HailoRT install + venv + self-test)
├── README.md
├── CLAUDE.md                     # 이 파일
├── inference/                    # Pi 5 추론 (Domain + App layer)
│   ├── types.py                  # Detection, PipelineConfig, CLASS_NAMES (source of truth)
│   ├── pipeline.py               # HailoInferencePipeline (mock=True for dev PC)
│   ├── postprocess.py            # YOLO det decode + render_overlay/summary
│   ├── _camera.py                # FrameSource ABC: CV2 / ImageFolder
│   ├── live_detect.py            # FR-05 — neomscope-live (USB webcam realtime)
│   ├── batch_detect.py           # FR-06 — neomscope-batch (folder inference)
│   └── capture_and_detect.py     # FR-07 — neomscope-capture (interactive demo)
├── tools/                        # Dev PC tooling (label/export/compile)
│   ├── bootstrap_labeling.py     # FR-01 — runs in legacy venv (Py 3.7 + TF 1.15 + mrcnn)
│   ├── package_for_roboflow.py   # Pair labels with resized images for upload (3.2 GB → 9.5 MB)
│   ├── validate_dataset.py       # Pre-training health check
│   ├── visualize_labels.py       # Render bbox overlays for review (--max-size + --jpg)
│   ├── export_onnx.py            # PyTorch → ONNX with parity check
│   └── compile_hef.sh            # ONNX → .hef via Hailo DFC (run in WSL2)
├── training/
│   ├── data.yaml                 # YOLO dataset definition (5 classes)
│   └── notebooks/
│       └── train-yolo11-det-kaggle.ipynb   # Self-contained Kaggle notebook
├── tests/                        # 75 unit tests, all passing
├── models/{pt,onnx,hef}/         # gitignored — train artifacts
├── datasets/                     # gitignored — Roboflow output
├── captured_raw_images/          # 22 sample images (committed)
├── raws/JPEG_Export_Data/        # gitignored — 105 raw 35-MB JPEGs (3.2 GB)
├── weights/                      # gitignored — legacy .h5 (245 MB)
├── legacy/                       # Mask R-CNN v1.0 보존
└── docs/                         # PDCA documents (plan / design / status)
```

## Working with this codebase

### Source of truth invariants
- **Class taxonomy**: `inference/types.py::CLASS_NAMES`. Mirror in `training/data.yaml`. Used by all postprocess + visualization paths.
- **YOLO det label format**: `<class_id> <cx> <cy> <w> <h>` normalized [0,1]. NOT segmentation polygons.
- **Image input size**: 640×640 (set in `PipelineConfig.image_size`). Must match HEF compile target.
- **Hardware**: AI HAT+ 2 with Hailo-10H — Hailo-8/8L tooling (Model Zoo v2.x branch) does NOT work. Use `master` branch + DFC 4.x+.

### Why YOLO-det, not -seg
[legacy/dataset_config.py:88-94](legacy/dataset_config.py#L88-L94) drew bbox-shaped rectangles as masks during training, so the legacy "instance segmentation" was effectively bbox-only. det is functionally equivalent and ~30% faster to train + simpler to deploy. Documented in `docs/02-design/features/aihat-yolo-port.design.md` §1.3.

### Bootstrap labeling
Original COCO-style annotations are missing. `tools/bootstrap_labeling.py` runs the legacy Mask R-CNN on the 127 raw images and emits YOLO det labels for Roboflow review. Runs in the conda `neomscope-legacy` env only (Python 3.7 + TF 1.15 + mrcnn).

### Mock inference path
`HailoInferencePipeline(cfg, mock=True)` bypasses Hailo entirely and returns a synthetic detection — used by 7 unit tests and lets `neomscope-batch --mock` run on the dev PC without an NPU.

### Class imbalance
Bootstrap labels show heavy Ana skew (Ana 250 vs Inter 4 = 62.5:1). Ultralytics auto-handles minor imbalance via mosaic/mixup; for severe cases use `cls_pw` weights or oversampling — decide in Phase 2 (Kaggle training) after the first mAP report.

## Commands

### Dev PC (Windows 11 + uv venv at `.venv/`, Python 3.11)
```powershell
# Tests
.venv\Scripts\python.exe -m pytest tests/unit/

# Lint
.venv\Scripts\python.exe -m ruff check .

# Bootstrap → YOLO labels (requires the legacy conda env)
conda run -n neomscope-legacy python tools/bootstrap_labeling.py `
    --weights weights/mask_rcnn_onioncell_1020_0089.h5 `
    --images raws/JPEG_Export_Data captured_raw_images `
    --output datasets/onioncell

# Package for Roboflow (resize 35MB → 80KB JPEGs + zip)
.venv\Scripts\python.exe tools/package_for_roboflow.py `
    --labels datasets/onioncell/labels/auto `
    --images raws/JPEG_Export_Data captured_raw_images `
    --output runs/roboflow-upload --zip

# Validate post-Roboflow dataset
.venv\Scripts\python.exe tools/validate_dataset.py datasets/onioncell --strict

# Export PyTorch → ONNX (after Kaggle training)
.venv\Scripts\python.exe tools/export_onnx.py --weights models/pt/best.pt

# Smoke test inference scripts on dev PC (no NPU)
.venv\Scripts\python.exe -m inference.batch_detect --input captured_raw_images --output /tmp/out --mock
```

### WSL2 Ubuntu (Hailo DFC)
```bash
ROOT="/mnt/c/WorkSpace/Project/neoscope/deploy_detect_onioncell"
bash "$ROOT/tools/compile_hef.sh" \
    "$ROOT/models/onnx/best.onnx" \
    "$ROOT/models/hef/best.hef" \
    "$ROOT/datasets/onioncell/calib_imgs"
```

### Pi 5 (after `setup.sh` + scp HEF)
```bash
neomscope-live --camera /dev/video0
neomscope-batch --input captured/ --output results/
neomscope-capture --camera 0
```

## Conventions
- Korean comments/labels are intentional (전람회 출품용) — preserve.
- Output dirs (`detection_results_*`) are runtime artifacts — code may write but do not commit results.
- Timestamp filenames: `YYYY-MM-DD_HH-MM-SS.jpg` (see [captured_raw_images/](captured_raw_images/)).
- Type hints: 100% on new code (`inference/`, `tools/`, `training/`, `tests/`). Legacy untouched.
- Lint: ruff config in pyproject.toml. Run `ruff check .` before commit.

## bkit Integration
이 프로젝트는 bkit Vibecoding Kit과 함께 운영된다.
- PDCA 상태: [docs/.pdca-status.json](docs/.pdca-status.json)
- Plan: [docs/01-plan/features/aihat-yolo-port.plan.md](docs/01-plan/features/aihat-yolo-port.plan.md) (v0.4)
- Design: [docs/02-design/features/aihat-yolo-port.design.md](docs/02-design/features/aihat-yolo-port.design.md) (v0.2)
- 워크플로: `/pdca plan` → `/pdca design` → `/pdca do` → `/pdca analyze` → `/pdca report`
- 레벨: Dynamic
