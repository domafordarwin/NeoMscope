# NeoMscope — Onion Cell Division Detector

## Project Overview
Mask R-CNN 기반 양파 세포 분열 자동 검출 시스템. 과학전람회용 개발 버전(v1.0).
현미경 또는 웹캠으로 촬영한 양파 표피세포 이미지에서 세포 분열 단계(전기/중기/후기/말기 등)를 자동 분류·계수한다.

- **Domain**: Computer Vision / Biological Imaging
- **Model**: Matterport Mask R-CNN (TensorFlow/Keras)
- **Classes**: 5 division stages + background (NUM_CLASSES = 1 + 5)
- **Input**: 512×512 square (microscope/webcam captures)
- **bkit Level**: Dynamic

## Tech Stack
- Python (TensorFlow, Keras, scikit-image, OpenCV, imutils, NumPy, Matplotlib, PIL)
- Mask R-CNN (`mrcnn` package — Matterport implementation)
- Jupyter notebooks for training/evaluation experiments

## Repository Layout
```
.
├── capture_and_detect.py       # Capture + detection combined pipeline
├── live_detect.py              # Real-time webcam detection
├── detect_onioncell.py         # Batch detection on saved images
├── take_webcam_pictures.py     # Image capture utility
├── config_onion_20201022.py    # OnionCellConfig (anchors, epochs, loss weights)
├── dataset_config.py           # Dataset loader / class definitions
├── object_detection_classes_onion.txt
├── colors.txt                  # Per-class visualization colors
├── train-onion-v2.ipynb        # Training notebook
├── evaluate-onion.ipynb        # Evaluation notebook
├── detect-onion-v3.ipynb       # Inference notebook
├── captured_raw_images/        # Sample capture inputs
├── detection_results_captured/ # Output: batch detection results (gitignored)
├── detection_results_live/     # Output: live detection frames (gitignored)
├── weights/                    # Trained .h5 weights (gitignored — too large for GitHub)
└── docs/                       # PDCA / bkit documentation
```

## Working with this codebase

### Weights are NOT in the repo
`weights/mask_rcnn_onioncell_1020_0089.h5` (~245MB) is gitignored. To run inference, place the trained weights file in `weights/` locally. For sharing, use Git LFS, Google Drive, or release artifacts — never commit directly.

### Coupling between scripts
`config_onion_20201022.py` and `dataset_config.py` are imported via `from ... import *` from most entry points. Renaming or moving these files breaks every detector script — update all import sites at once.

### Class count is hardcoded
`NUM_CLASSES = 1 + 5` (background + 5 division stages) appears in multiple files. Changing the class taxonomy requires synchronized edits in `PredictionConfig`, `OnionCellConfig`, `dataset_config.py`, `object_detection_classes_onion.txt`, and `colors.txt`.

### Jupyter notebooks are source of truth for training
`train-onion-v2.ipynb` is the canonical training pipeline. The `.py` scripts are deployment-side (capture/detect/serve), not training.

## Commands
- **Live detection**: `python live_detect.py`
- **Batch detection**: `python detect_onioncell.py`
- **Capture only**: `python take_webcam_pictures.py`
- **Capture + detect**: `python capture_and_detect.py`

## Conventions
- Korean comments and tagline are intentional (전람회 출품용); preserve them.
- Output directories (`detection_results_*`) are runtime artifacts — code may write into them but do not commit results.
- Timestamp filenames use `YYYY-MM-DD_HH-MM-SS.jpg` format (see `captured_raw_images/`).

## bkit Integration
이 프로젝트는 bkit Vibecoding Kit과 함께 운영된다.
- PDCA 상태: `docs/.pdca-status.json`
- 추천 워크플로: `/pdca plan` → `/pdca design` → `/pdca do` → `/pdca analyze`
- 레벨: Dynamic (Python ML 백엔드 + 스크립트 실행)
