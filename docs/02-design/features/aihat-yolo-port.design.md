---
template: design
version: 1.2
feature: aihat-yolo-port
date: 2026-05-01
author: domafordarwin
project: NeoMscope
version: 1.0
---

# aihat-yolo-port Design Document

> **Summary**: Mask R-CNN(TF1.x) → **YOLOv11-det**(PyTorch/Ultralytics) 마이그레이션 + **Pi 5 + AI HAT+ 2 (Hailo-10H)** 엣지 추론 파이프라인 설계. 학습은 **Kaggle**, 라벨은 **부트스트랩**.
>
> **Project**: NeoMscope
> **Version**: 1.0
> **Author**: domafordarwin
> **Date**: 2026-05-01
> **Status**: Draft (v0.2)
> **Planning Doc**: [aihat-yolo-port.plan.md](../../01-plan/features/aihat-yolo-port.plan.md) (v0.4)

### Pipeline References

| Phase | Document | Status |
|-------|----------|--------|
| Phase 1 | Schema (5 division stages) | 🔵 본 문서 §3에 통합 |
| Phase 2 | Coding Conventions | 🔵 본 문서 §10에 통합 |
| Phase 3 | Mockup | ⏭ 시각화는 §7.4 참조 |
| Phase 4 | API Spec | ⏭ N/A (외부 API 없음) |

---

## 1. Overview

### 1.1 Design Goals

1. **Edge-first**: 모든 추론은 Pi 5 + AI HAT+ 2에서 수행. 학습은 PC, 배포는 Pi.
2. **재현성**: README + `setup.sh` 한 번으로 신규 머신·SD카드에서 재현 가능.
3. **단계별 검증**: COCO 변환 → 학습 → ONNX 검증 → HEF 검증 → Pi 추론 검증 — 각 단계 출력이 다음 단계 입력의 신뢰 기준.
4. **저위험**: 안정 stack(YOLOv11-seg, GStreamer + hailo_apps_infra) 우선, 백업 경로 명시.
5. **레거시 보존**: 기존 Mask R-CNN 코드는 `legacy/`로 이전(삭제 X). 전람회에서 비교 데모로도 활용.

### 1.2 Design Principles

- **단방향 데이터 흐름**: COCO → YOLO → ONNX → HEF → 추론. 역방향 변환 없음.
- **경계만 방어**: 외부 I/O(파일·카메라·모델 로드)에서만 예외 처리, 내부는 fail-fast.
- **모듈 단일 책임**: 변환/학습/내보내기/추론을 별도 디렉터리·별도 진입점.
- **상태 비저장**: 추론 모듈은 모델 핸들 외에 상태 없음. 멀티 인스턴스 안전.
- **설정은 YAML/CLI로**: 코드 수정 없이 임계값·경로·모델 변경 가능.

### 1.3 ✅ Resolved Design Decisions (2026-05-01)

**1.3.1 Detector: YOLOv11-det 확정** (seg 미채택)
- 이유: [dataset_config.py:88-94](../../../dataset_config.py#L88-L94)에서 polygon이 아닌 `bbox` 사각형을 마스크로 사용. 현 모델 = bbox 검출. det로 충분.
- 효과: 학습 시간 ~30% 감소, ONNX 출력 텐서 4종→2종, postprocess 단순화 (마스크 디코딩 불필요).

**1.3.2 라벨 데이터: 부트스트랩 라벨링** (COCO JSON 부재)
- 발견: 어노테이션 파일이 모든 디렉터리·이웃 프로젝트에 부재. 이미지만 보존.
- 데이터 위치: **[raws/JPEG_Export_Data/](../../../raws/JPEG_Export_Data/) 105장** (2020-10-07 촬영, `0065-09` 시리즈) + [captured_raw_images/](../../../captured_raw_images/) 22장 (2021-2022).
- 전략: 기존 Mask R-CNN(`weights/mask_rcnn_onioncell_1020_0089.h5`)으로 자동 추론 → bbox 추출 → Roboflow에서 시각 보정 → YOLO det 라벨.

**1.3.3 학습 환경: Kaggle**
- 발견: 학습 PC = Quadro P2000 4GB VRAM(2017, Pascal). YOLO 학습 부족.
- 결정: Kaggle T4/P100 16GB VRAM, 30시간/주 무료. Colab 백업.
- 학습 PC 역할: 코드 개발, 부트스트랩 라벨링(레거시 환경), ONNX 검증, **WSL2 기반 HEF 컴파일**, 시연 준비.

**1.3.4 도구체인 호스트: Windows 11 + WSL2**
- 발견: Hailo Dataflow Compiler는 Linux 전용.
- 결정: 학습 PC는 Windows 11 → WSL2 + Ubuntu 22.04에 DFC 4.x+ 설치.

---

## 2. Architecture

### 2.1 Component Diagram

```
┌──────────  학습 PC (Windows 11 + WSL2 + Ubuntu) — 4GB VRAM Quadro P2000  ─────┐
│                                                                                │
│  [raws/JPEG_Export_Data/ 105장] + [captured_raw_images/ 22장] = 127장          │
│        │                                                                       │
│        ▼  (legacy Python 3.7-3.8 venv + TF 1.15 + mrcnn 패키지)               │
│  tools/bootstrap_labeling.py                                                  │
│   = legacy/detect_onioncell.py 호출 → bbox 추출 → YOLO det .txt 생성          │
│        │                                                                       │
│        ▼                                                                       │
│  Roboflow Web UI: 시각 검수 + 신뢰도 < 0.5 케이스 보정                        │
│        │                                                                       │
│        ▼                                                                       │
│  [데이터셋 export: YOLO det 형식]                                             │
│        │                                                                       │
│        ▼                                                                       │
│  tools/validate_dataset.py ──→ [통계·시각 검수 리포트]                        │
│        │                                                                       │
│        ▼  (Kaggle Dataset 또는 GitHub release로 업로드)                       │
└────────┼─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────  Kaggle Notebook (T4/P100 GPU 16GB VRAM, 무료)  ──────────────┐
│                                                                              │
│  training/notebooks/train-yolo11-det-kaggle.ipynb                           │
│   ├─ pip install ultralytics                                                │
│   ├─ data.yaml 생성                                                          │
│   ├─ YOLO('yolo11s.pt').train(...)   ← det 사전학습                          │
│   └─ best.pt 다운로드 → 로컬 models/pt/                                     │
│                                                                              │
└────────┬─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────  학습 PC (Windows 11 + WSL2 Ubuntu) — ONNX/HEF 컴파일  ──────────────┐
│                                                                              │
│  tools/export_onnx.py (Windows native Python OK) ──→ models/onnx/best.onnx │
│        │                                                                     │
│        ▼  (WSL2 Ubuntu 안)                                                  │
│  tools/compile_hef.sh (Hailo DFC 4.x+)  ──[calibration set 256장]──┐        │
│        │                                                            │        │
│        ▼                                                            ▼        │
│  models/hef/best.hef                                        [calib_imgs/]    │
│        │                                                                     │
└────────┼─────────────────────────────────────────────────────────────────────┘
         │
         │  scp / USB drive
         ▼
┌─────────────────────  Pi 5 + AI HAT+ 2 (Hailo-10H, 8GB)  ──────────────┐
│                                                                          │
│  [USB Webcam] ──┐                                                        │
│                 ▼                                                        │
│       inference/pipeline.py                                              │
│         GStreamer:                                                       │
│           v4l2src → videoconvert → hailonet(best.hef)                   │
│                  → hailofilter(postprocess) → hailooverlay → sink        │
│                 │                                                        │
│         ┌───────┴───────┬──────────────────────┐                         │
│         ▼               ▼                      ▼                         │
│  inference/        inference/           inference/                       │
│  live_detect.py    batch_detect.py      capture_and_detect.py            │
│  (FR-05, 실시간)   (FR-06, 폴더 일괄)   (FR-07, 캡처+추론)              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (Single Frame, 실시간)

```
USB Webcam (BGR, 1280×720, 30 FPS)
    │
    ▼ v4l2src
[Frame] (raw BGR)
    │
    ▼ videoconvert + videoscale
[Frame] (640×640 RGB, letterbox padded)
    │
    ▼ hailonet (best.hef)
[Raw output tensors] — det는 seg보다 출력 단순
  - bbox_proto:  (8400, 4)         # YOLO grid bboxes (cx, cy, w, h)
  - cls_proto:   (8400, 5)         # 5 classes (sigmoid scores)
    │
    ▼ inference/postprocess.py (CPU on Pi, 가벼움)
[Detection list]
  - List[Detection(bbox, class_id, conf)]
  - confidence threshold (default 0.25)
  - NMS (IoU=0.5)
  - bbox unscale to original frame
    │
    ▼ overlay
[Annotated frame with bboxes + class labels + counts]
    │
    ▼ fpsdisplaysink / appsink
[Display on HDMI / save to file]
```

### 2.3 Dependencies

| Component | Depends On | Purpose |
|-----------|-----------|---------|
| `tools/convert_coco_to_yolo.py` | `pycocotools`, `numpy`, `opencv-python` | COCO → YOLO 변환 |
| `tools/validate_dataset.py` | `matplotlib`, `pyyaml` | 시각 검수, 클래스 분포 검사 |
| `training/train.py` | `ultralytics` (8.3+), `torch` (2.x, CUDA) | 학습 |
| `tools/export_onnx.py` | `ultralytics`, `onnx`, `onnxruntime` | ONNX export + 검증 |
| `tools/compile_hef.sh` | Hailo Dataflow Compiler 4.x+ | HEF 컴파일 |
| `inference/pipeline.py` | `hailo-platform` (HailoRT 4.x+), GStreamer 1.20+, `hailo_apps_infra` | Pi 추론 코어 |
| `inference/postprocess.py` | `numpy`, `opencv-python` (NPU offload, 후처리만 CPU) | 후처리·시각화 |
| `inference/live_detect.py` | `pipeline`, `postprocess`, `picamera2` (선택), `opencv-python` | 실시간 진입점 |

---

## 3. Data Model

### 3.1 Domain Schema (5 Division Stages)

| class_id | 영문 코드 | 한글 | 의미 | 색상 (BGR) |
|---------|-----------|------|------|-----------|
| 0 | `Inter` | 간기 | 분열하지 않는 정상 상태 | `(0, 50, 255)` (빨강) |
| 1 | `Pro` | 전기 | 염색체 응축 시작 | `(0, 255, 0)` (초록) |
| 2 | `Meta` | 중기 | 염색체가 적도판에 정렬 | `(255, 0, 0)` (파랑) |
| 3 | `Ana` | 후기 | 염색분체 양극 이동 | `(255, 255, 0)` (시안) |
| 4 | `Telo` | 말기 | 핵막 재형성 | `(0, 255, 255)` (노랑) |

> ⚠️ Mask R-CNN의 `NUM_CLASSES = 1 + 5`(background 포함)와 달리, YOLO는 **background를 별도 클래스로 표현하지 않음**. 5개 클래스만 정의.

### 3.2 Class Definition Source of Truth

```yaml
# training/data.yaml (YOLO 표준)
path: ./datasets/onioncell
train: images/train
val: images/val
test: images/test
names:
  0: Inter
  1: Pro
  2: Meta
  3: Ana
  4: Telo
```

### 3.3 YOLO Detection Label Format

각 이미지 `IMG_001.jpg`에 대해 `IMG_001.txt`:

```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

- 좌표는 [0, 1] 정규화 (`cx, cy` = bbox 중심, `w, h` = bbox 폭·높이)
- 같은 이미지에 여러 객체 = 여러 줄
- background는 줄로 표현하지 않음 (없음 = 없음)
- Ultralytics 표준 포맷 — 별도 변환 도구 불필요

### 3.4 Train/Val/Test Split

| Split | 비율 | 목적 |
|-------|-----|------|
| train | 70% | 학습 |
| val   | 20% | 학습 중 mAP 모니터링 + early stop |
| test  | 10% | 최종 보고용 holdout (학습 절대 금지) |

분할 정책:
- `random_state=42` 고정
- 클래스 분포 stratified split (`sklearn.model_selection.train_test_split`)
- 같은 이미지의 어노테이션은 같은 split에 (이미지 단위 분할)

### 3.5 Calibration Dataset (HEF 컴파일용)

- `train` set에서 무작위 256장 (전체 분포 대표)
- 별도 `calib_imgs/`로 복사 (변형 없음)
- Hailo DFC가 INT8 양자화 시 사용

---

## 4. Bootstrap Labeling & Training Specifications

### 4.1 Bootstrap Labeling 입력

**입력 데이터**:
- 이미지: [raws/JPEG_Export_Data/](../../../raws/JPEG_Export_Data/) 105장 + [captured_raw_images/](../../../captured_raw_images/) 22장 = **127장**
- 모델: `weights/mask_rcnn_onioncell_1020_0089.h5` (245MB, gitignore. 로컬에 보관 가정)
- 클래스 매핑: `legacy/object_detection_classes_onion.txt` (`Inter, Pro, Meta, Ana, Telo`)

**환경 (Phase 0/1에서 일회성 구성)**:
```bash
# 학습 PC에서 (uv 활용)
uv venv --python 3.8 .legacy-venv
source .legacy-venv/Scripts/activate     # Windows: .legacy-venv\Scripts\activate.ps1
uv pip install tensorflow==1.15 keras==2.2.5 mrcnn-custom \
               numpy==1.19 scikit-image==0.17 opencv-python==4.5 imutils Pillow
```

> ⚠️ R-12 폴백: 위 환경 구성 실패 시 **Roboflow 수작업 라벨링**으로 즉시 전환 (1-3일 추가).

### 4.2 Bootstrap 알고리즘

```python
# tools/bootstrap_labeling.py
"""
기존 Mask R-CNN으로 raw 이미지 자동 라벨링.
실행: python tools/bootstrap_labeling.py \
        --weights weights/mask_rcnn_onioncell_1020_0089.h5 \
        --images raws/JPEG_Export_Data captured_raw_images \
        --output datasets/onioncell/labels/auto/ \
        --conf-min 0.5 \
        --review-min 0.5
"""
def bootstrap(weights_path: Path, image_dirs: list[Path], output_dir: Path,
              conf_min: float = 0.5, review_min: float = 0.5) -> BootstrapReport:
    # 1. legacy mrcnn 모델 로드
    sys.path.insert(0, "legacy")
    from legacy.config_onion_20201022 import OnionCellConfig
    from mrcnn.model import MaskRCNN

    cfg = InferenceConfig()  # batch=1
    model = MaskRCNN(mode="inference", model_dir="logs/", config=cfg)
    model.load_weights(weights_path, by_name=True)

    # 2. 이미지 순회
    review_list = []   # 신뢰도 낮은 케이스, 수작업 검토 필요
    for img_path in iter_images(image_dirs):
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # 3. 추론 — bbox + class만 사용 (mask는 §1.3 발견에 따라 이미 사각형)
        results = model.detect([img], verbose=0)[0]
        boxes = results['rois']           # (N, 4) [y1, x1, y2, x2]
        cls_ids = results['class_ids']    # (N,) 1-indexed (mrcnn) → 0-indexed (yolo)
        scores = results['scores']        # (N,)

        # 4. 신뢰도 필터 + YOLO 변환
        lines = []
        needs_review = False
        for box, cls_id, score in zip(boxes, cls_ids, scores):
            if score < conf_min:
                continue
            if score < review_min:
                needs_review = True
            y1, x1, y2, x2 = box
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w, h = (x2 - x1) / W, (y2 - y1) / H
            yolo_cls = cls_id - 1   # mrcnn 1-indexed → yolo 0-indexed
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 5. 출력
        out_txt = output_dir / f"{img_path.stem}.txt"
        out_txt.write_text("\n".join(lines))

        if needs_review or len(lines) == 0:
            review_list.append(img_path.name)

    return BootstrapReport(
        total=count, review_count=len(review_list),
        review_files=review_list,
    )
```

**Roboflow 검수 워크플로**:
1. `datasets/onioncell/images/` + `datasets/onioncell/labels/auto/`를 zip → Roboflow 프로젝트에 업로드 (YOLOv8 형식)
2. Roboflow Annotate에서 `review_list.json`의 파일들 우선 검토 + 모든 이미지 빠른 시각 검수
3. 잘못된 bbox 수정·추가·삭제, 클래스 오분류 수정
4. **Generate** → YOLOv11/v8 형식 export → `datasets/onioncell/` 위 `labels/{train,val,test}/` 분할 자동 생성

**검증 항목** (`tools/validate_dataset.py`):
1. 모든 좌표 ∈ [0, 1]
2. `w, h` > 0 (degenerate bbox 없음)
3. 클래스 분포 (불균형 < 10:1 권장)
4. 빈 라벨 파일 비율 < 10% (현미경 이미지에 항상 세포가 있는 건 아님)
5. 이미지·라벨 1:1 매핑 (라벨 누락 0)
6. **부트스트랩 신뢰도 통계**: 평균 score, < 0.5 비율

### 4.3 학습 하이퍼파라미터 (Kaggle Notebook)

```python
# training/notebooks/train-yolo11-det-kaggle.ipynb (의사 셀)

# Cell 1: 설치
!pip install ultralytics

# Cell 2: 데이터셋 다운로드 (Kaggle Dataset 또는 GitHub release URL)
!wget -q https://github.com/domafordarwin/NeoMscope/releases/download/dataset-v1/onioncell.zip
!unzip -q onioncell.zip -d /kaggle/working/datasets/

# Cell 3: data.yaml 생성
import yaml
yaml.dump({
    'path': '/kaggle/working/datasets/onioncell',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'names': {0: 'Inter', 1: 'Pro', 2: 'Meta', 3: 'Ana', 4: 'Telo'}
}, open('data.yaml', 'w'))

# Cell 4: 학습
from ultralytics import YOLO
model = YOLO('yolo11s.pt')   # ⭐ det 사전학습 (seg 아님)
results = model.train(
    data='data.yaml',
    epochs=100,
    patience=20,                  # early stop
    batch=16,                     # T4 16GB 기준 여유
    imgsz=640,                    # AI HAT+ 2 입력 사이즈와 일치
    device=0,                     # T4/P100
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    weight_decay=0.0005,
    warmup_epochs=3,
    # Augmentation (현미경·세포 도메인 특화)
    hsv_h=0.015,                  # 현미경 광원 색온도 변화
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=180,                  # ⭐ 세포 회전 불변 → 풀 회전
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.5,                   # ⭐ 상하반전 OK (세포)
    mosaic=1.0,
    mixup=0.1,
    close_mosaic=10,
    # Kaggle 환경
    project='/kaggle/working/runs',
    name='yolo11s-det-onioncell',
    plots=True,
    save_period=10,               # 10 epoch마다 체크포인트 (R-11)
)

# Cell 5: 결과 다운로드
from IPython.display import FileLink
FileLink('/kaggle/working/runs/yolo11s-det-onioncell/weights/best.pt')
```

**실행 시간 추정 (T4 16GB)**:
- 127장 × 100 epoch ≈ 30-60분 (early stop 고려)
- Kaggle 30시간/주 한도의 2-3% 사용

**도메인 특화 augmentation 정당화**:
- `degrees=180`, `flipud=0.5`: 양파 세포 분열 단계는 회전·반전과 무관 → 데이터 부풀리기 효과
- `hsv_h/s/v` 강하게: 현미경 광원·노출에 따라 색조가 다양 → 일반화 향상

### 4.4 (deprecated) seg 분기 — det 채택으로 비활성

§1.3.1에 따라 본 설계는 **det 단일 경로**. seg 경로는 `legacy/`에 보존된 기존 동작이 그 역할을 대신함.

### 4.5 ONNX Export (학습 PC, Windows native Python OK)

```python
# tools/export_onnx.py
from ultralytics import YOLO
from pathlib import Path

model = YOLO('models/pt/best.pt')   # Kaggle에서 다운로드한 best.pt
model.export(
    format='onnx',
    opset=17,                     # Hailo DFC 4.x 호환 범위 (16-19)
    dynamic=False,                # ⭐ Hailo는 정적 입력 shape만 지원
    simplify=True,                # onnxsim 통과 (그래프 단순화)
    imgsz=640,
    half=False,                   # FP32로 export, INT8 양자화는 DFC가
    nms=False,                    # ⭐ NMS는 Pi에서 후처리 (HEF에 NMS 포함 X)
)
```

**ONNX 검증** (export 후 즉시):
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('best.onnx')
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
ort_out = sess.run(None, {'images': dummy})

torch_out = model.predict(dummy_tensor)  # PyTorch 동일 입력
# allclose 비교: max abs diff < 1e-3
```

### 4.6 HEF Compilation (WSL2 Ubuntu)

```bash
# tools/compile_hef.sh — WSL2 Ubuntu 안에서 실행
#!/bin/bash
set -euo pipefail

# 0. WSL2 Ubuntu 안에서 Windows 파일시스템 마운트 경로
ROOT="/mnt/c/WorkSpace/Project/neoscope/deploy_detect_onioncell"

ONNX="${1:-$ROOT/models/onnx/best.onnx}"
HEF="${2:-$ROOT/models/hef/best.hef}"
CALIB="${3:-$ROOT/datasets/onioncell/calib_imgs}"
ARCH="hailo10h"   # ⭐ Hailo-10H 명시

# 1. Hailo Model Zoo master branch에서 yolo11s det config 사용 (seg 아님)
hailomz compile \
  --ckpt "$ONNX" \
  --calib-path "$CALIB" \
  --yaml hailo_model_zoo/cfg/networks/yolo11s.yaml \
  --hw-arch "$ARCH" \
  --output-dir "$(dirname "$HEF")"

echo "✅ HEF compiled: $HEF"
```

**Calibration 가이드라인**:
- 256장 (8 배치 × 32장)
- 5 클래스 모두 포함되도록 stratified sampling
- 학습 set에서만 (val/test 사용 금지)
- 변형 없는 원본 (augmentation 적용 X)

### 4.8 Pi 5 배포

| 산출물 | 경로 (Pi 5) | 비고 |
|--------|-------------|------|
| `best.hef` | `~/neomscope/models/hef/` | scp로 전송 |
| 추론 코드 | `~/neomscope/inference/` | git clone (gitignore된 weights 제외) |
| 환경 | `~/neomscope/.venv/` | `python -m venv` + `pip install -e .` |

---

## 5. Inference Pipeline (Pi 5)

### 5.1 GStreamer Graph

```
                                    appsink (Python)
                                         ▲
                                         │
v4l2src device=/dev/video0 ─────────────►│
  ↓                                      │
videoconvert                             │
  ↓                                      │
video/x-raw,format=RGB,width=640,        │
            height=640                   │
  ↓                                      │
queue (max-size-buffers=2 leaky=2) ──────┤  ← 버퍼링 제한 (지연 최소화)
  ↓                                      │
hailonet hef-path=models/hef/best.hef ──►│  ← Hailo NPU 추론
  ↓                                      │
hailofilter so-path=libyolo11_seg_post.so│  ← (선택) C++ 후처리 .so
  ↓                                      │
hailooverlay font-thickness=2            │  ← (선택) bbox/마스크 오버레이
  ↓                                      │
videoconvert                             │
  ↓                                      │
fpsdisplaysink sync=false                │  ← FPS 표시 + HDMI 출력
```

**대안 (소프트 폴백)**: `hailofilter`/`hailooverlay`가 미동작 시 `appsink`로 raw 텐서를 받아 Python에서 후처리(`inference/postprocess.py`).

### 5.2 Postprocessing

```python
# inference/postprocess.py — det 단순화 (마스크 디코딩 없음)
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) 원본 픽셀 좌표
    class_id: int                      # 0..4
    confidence: float                  # 0..1

def decode_yolo_det_output(
    raw_outputs: dict[str, np.ndarray],
    orig_shape: tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    bbox_proto = raw_outputs['bbox']    # (8400, 4) — cx, cy, w, h normalized
    cls_proto = raw_outputs['cls']      # (8400, 5) — sigmoid scores

    # 1. confidence 필터
    cls_scores = cls_proto.max(axis=1)
    cls_ids = cls_proto.argmax(axis=1)
    keep = cls_scores >= conf_threshold

    # 2. cx,cy,w,h → x1,y1,x2,y2 (640×640 letterbox 좌표)
    bx, by, bw, bh = bbox_proto[keep].T
    x1, y1 = bx - bw/2, by - bh/2
    x2, y2 = bx + bw/2, by + bh/2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 3. NMS (cv2.dnn.NMSBoxes 또는 직접 구현)
    keep_idx = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), cls_scores[keep].tolist(),
        conf_threshold, iou_threshold,
    )

    # 4. 원본 이미지 좌표로 unscale (letterbox 역변환)
    detections = []
    for i in keep_idx:
        bbox_orig = unletterbox_bbox(boxes_xyxy[i], orig_shape, 640)
        detections.append(Detection(
            bbox=tuple(map(int, bbox_orig)),
            class_id=int(cls_ids[keep][i]),
            confidence=float(cls_scores[keep][i]),
        ))
    return detections
```

> **단순화 효과**: seg 대비 약 50줄 감소, 의존성 감소(`sigmoid`, `crop_to_bbox`, `mask resize` 불필요), Pi CPU 부하 감소.

### 5.3 Three Inference Scripts

#### 5.3.1 `inference/live_detect.py` (FR-05)

```python
# 의사코드
def main(args):
    pipeline = HailoInferencePipeline(args.hef, args.config)
    cap = cv2.VideoCapture(args.camera)   # USB 또는 picamera2

    fps_counter = FPSCounter(window=60)
    while True:
        ok, frame = cap.read()
        if not ok: break

        detections = pipeline.infer(frame)
        annotated = render_overlay(frame, detections, CLASS_COLORS)

        fps = fps_counter.tick()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), ...)
        cv2.putText(annotated, f"Counts: {count_by_class(detections)}", (10, 60), ...)

        cv2.imshow("NeoMscope Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()
```

#### 5.3.2 `inference/batch_detect.py` (FR-06)

```python
def main(args):
    pipeline = HailoInferencePipeline(args.hef, args.config)
    images = sorted(Path(args.input).glob('*.jpg'))

    results_summary = []
    for img_path in tqdm(images):
        frame = cv2.imread(str(img_path))
        detections = pipeline.infer(frame)
        annotated = render_overlay(frame, detections, CLASS_COLORS)

        out_img = Path(args.output) / img_path.name
        cv2.imwrite(str(out_img), annotated)

        results_summary.append({
            'image': img_path.name,
            'counts': count_by_class(detections),
            'detections': [d.to_dict() for d in detections],
        })

    (Path(args.output) / 'summary.json').write_text(json.dumps(results_summary, indent=2))
    pipeline.close()
```

#### 5.3.3 `inference/capture_and_detect.py` (FR-07)

```python
def main(args):
    pipeline = HailoInferencePipeline(args.hef, args.config)
    cap = cv2.VideoCapture(args.camera)

    print("[Space] capture+detect, [q] quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.imshow("Preview", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            raw = Path(args.output) / 'raw' / f'{ts}.jpg'
            cv2.imwrite(str(raw), frame)

            detections = pipeline.infer(frame)
            annotated = render_overlay(frame, detections, CLASS_COLORS)
            out = Path(args.output) / 'detected' / f'{ts}.jpg'
            cv2.imwrite(str(out), annotated)

            print(f"  → counts: {count_by_class(detections)}, saved {out.name}")
        elif key == ord('q'):
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()
```

### 5.4 Visualization

```python
# inference/postprocess.py
CLASS_COLORS = {
    0: (0, 50, 255),   # Inter - 빨강
    1: (0, 255, 0),    # Pro - 초록
    2: (255, 0, 0),    # Meta - 파랑
    3: (255, 255, 0),  # Ana - 시안
    4: (0, 255, 255),  # Telo - 노랑
}

def render_overlay(frame: np.ndarray, dets: list[Detection], colors: dict) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        color = colors[d.class_id]
        # bbox 반투명 채움 (det에는 마스크 없음 — bbox 자체를 시각화)
        overlay = out.copy()
        cv2.rectangle(overlay, d.bbox[:2], d.bbox[2:], color, -1)  # filled
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
        # bbox 외곽선 + 라벨
        cv2.rectangle(out, d.bbox[:2], d.bbox[2:], color, 2)
        label = f"{CLASS_NAMES[d.class_id]} {d.confidence:.2f}"
        cv2.putText(out, label, (d.bbox[0], d.bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
```

> 한글 라벨이 필요할 경우 PIL + NanumGothic 폰트로 별도 렌더 함수. 시연용은 영문 약어로 통일 권장 (R-08).

---

## 6. Error Handling

| 경계 | 에러 케이스 | 처리 |
|------|------------|------|
| **모델 로드** | HEF 파일 없음 / 손상 | 즉시 에러 메시지 + 다운로드 안내 + 종료 |
| **카메라** | 디바이스 없음 / 권한 없음 | "카메라 X. lsof /dev/video0 확인" 메시지 + 종료 |
| **프레임 캡처** | 일시적 실패 (USB 흔들림) | 5회까지 retry, 그 후 종료 |
| **NPU 추론** | HailoRT 런타임 에러 | 1회 재초기화 시도, 실패 시 종료 |
| **출력 디렉터리** | 쓰기 권한 없음 | 시작 시 권한 확인, 즉시 알림 |
| **변환 단계** | COCO JSON 파싱 실패 | 라인 번호 + 컨텍스트 출력 |
| **변환 단계** | 클래스 매핑 누락 | 미정의 category_id 목록 출력 후 중단 |

내부 함수는 fail-fast (예외 그대로 전파). 진입점 main()에서만 try/except로 사용자 친화 메시지로 변환.

---

## 7. Security Considerations

본 프로젝트는 외부에 노출되지 않는 로컬·전람회 데모이므로 일반 웹 보안 항목은 N/A. 단:

- [ ] 카메라 권한: `video` 그룹만 허용 (Pi OS 기본)
- [ ] 학습 데이터에 개인정보 없음을 학습 전 확인
- [ ] HEF 파일은 학습 PC와 Pi 둘 다에 백업 (R-07)
- [ ] 시연 시 `~/.bash_history` 등 프롬프트에 자격증명 노출 없도록 환경변수만 사용

---

## 8. Test Plan

### 8.1 Test Scope

| Type | Target | Tool |
|------|--------|------|
| Unit | 라벨 변환 round-trip (COCO → YOLO → COCO 좌표 일치) | pytest |
| Unit | data.yaml 파싱 / 클래스 매핑 | pytest |
| Unit | postprocess: NMS, mask decode 결정성 | pytest |
| Smoke | inference/* 모듈 import 시 모델 미존재 에러 graceful | pytest |
| Integration | ONNX vs PyTorch 출력 차이 < 1e-3 | export_onnx.py 내장 |
| Integration | HEF 단일 추론 sanity (calib 이미지 1장) | compile_hef.sh 후속 |
| E2E (수동) | Pi 5: live_detect 30초간 30+ FPS 유지 | 수동 |
| E2E (수동) | Pi 5: batch_detect 23장 처리 < 5초 | 수동 |
| E2E (수동) | Pi 5: capture_and_detect 5회 캡처 모두 결과 저장 | 수동 |

### 8.2 Test Cases (핵심)

- [ ] **Happy**: 정상 COCO JSON → YOLO 변환 → 라벨 5클래스 모두 존재
- [ ] **Happy**: 학습 1 epoch 완료 → best.pt 생성 → ONNX export 성공
- [ ] **Happy**: Pi에서 best.hef 로드 → 더미 1장 추론 → 5클래스 중 하나 출력
- [ ] **Edge**: bbox 좌표가 이미지 경계 밖 (clamp 동작 확인)
- [ ] **Edge**: 빈 어노테이션 이미지 → 빈 .txt 생성 (라벨 누락 X)
- [ ] **Edge**: 한 이미지에 50+ 어노테이션 → NMS 후 중복 제거
- [ ] **Edge**: 신뢰도 낮은 검출 (< 0.25) → 표시 안 됨
- [ ] **Error**: HEF 손상 시 명확한 에러 메시지
- [ ] **Error**: 카메라 미연결 시 명확한 에러 메시지

### 8.3 Acceptance Criteria (Plan §4 Definition of Done과 매핑)

| Plan FR | Test 검증 방법 |
|---------|---------------|
| FR-01 | Unit: COCO→YOLO 변환, Integration: 시각 검수 50장 |
| FR-02 | `validate_dataset.py` 결과 통과율 ≥ 98% |
| FR-03 | Smoke: train.py 1 epoch 동작 |
| FR-04 | Integration: ONNX·HEF sanity |
| FR-05 | E2E: live_detect 30+ FPS |
| FR-06 | E2E: batch_detect 정확도 |
| FR-07 | E2E: capture_and_detect 5회 |
| FR-08 | E2E: 콘솔 출력 확인 |
| FR-09 | Visualization: 색상 매핑 콘솔 print로 확인 |
| FR-10 | E2E: 신규 SD카드 setup.sh 실행 |
| FR-11 | Error path 수동 시나리오 |

---

## 9. Clean Architecture (Python ML 변형)

표준 웹 4-layer 대신 **ML Edge 3-layer**로 재정의.

| Layer | 역할 | 위치 |
|-------|-----|------|
| **Application** | 진입점·CLI 파싱·오케스트레이션 | `inference/live_detect.py`, `batch_detect.py`, `capture_and_detect.py`, `training/train.py` |
| **Domain (Core)** | 추론 로직·후처리·타입 (외부 의존성 격리) | `inference/pipeline.py`, `inference/postprocess.py`, `inference/types.py` |
| **Infrastructure** | HailoRT/GStreamer/카메라/파일 I/O | `inference/_gst.py`, `inference/_camera.py`, `tools/_hailo_dfc.py` |

**의존성 규칙**:
```
Application ──→ Domain ←── Infrastructure
                  ↑
               (Domain은 numpy/dataclass 외 외부 의존 X)
```

**Import 규칙**:
| From | Can Import | Cannot |
|------|-----------|--------|
| `inference/live_detect.py` (App) | pipeline, postprocess, types | `_gst`, `_camera` 직접 |
| `inference/pipeline.py` (Domain) | postprocess, types, numpy | `_gst` 직접 (의존성 주입으로) |
| `inference/postprocess.py` (Domain) | types, numpy, cv2 | 다른 inference 모듈 |
| `inference/_gst.py` (Infra) | types만 | App, Domain |

---

## 10. Coding Convention

### 10.1 Naming

| 대상 | 규칙 | 예시 |
|------|-----|------|
| 모듈 파일 | `snake_case.py` | `convert_coco_to_yolo.py` |
| 패키지 디렉터리 | `snake_case` | `inference/`, `tools/` |
| 클래스 | `PascalCase` | `HailoInferencePipeline`, `Detection` |
| 함수 | `snake_case` | `decode_yolo_seg_output()` |
| 상수 | `UPPER_SNAKE_CASE` | `CLASS_COLORS`, `DEFAULT_CONF_THRESHOLD` |
| Private | `_leading_underscore` | `_decode_mask_proto()` |
| 타입(dataclass) | `PascalCase` | `Detection`, `PipelineConfig` |

### 10.2 Type Hints

- **신규 모듈 100%** type hint
- 공개 API는 PEP 604 union 문법 (`int | None`)
- `dataclass` 또는 `TypedDict` 사용 (자유 dict 지양)
- 레거시(`legacy/`)는 그대로 유지 (점진 적용 X)

### 10.3 Import Order (isort + Ruff)

```python
# 1. stdlib
import json
from dataclasses import dataclass
from pathlib import Path

# 2. third-party
import cv2
import numpy as np
from ultralytics import YOLO

# 3. local
from inference.types import Detection
from inference.postprocess import decode_yolo_seg_output
```

### 10.4 Environment Variables

| Variable | 의미 | 기본 |
|----------|-----|------|
| `NEOMSCOPE_HEF_PATH` | HEF 파일 경로 | `models/hef/best.hef` |
| `NEOMSCOPE_CAMERA` | 카메라 디바이스 또는 파일 | `/dev/video0` |
| `NEOMSCOPE_CONF_THRESHOLD` | 신뢰도 임계값 | `0.25` |
| `NEOMSCOPE_IOU_THRESHOLD` | NMS IoU 임계값 | `0.5` |
| `NEOMSCOPE_OUTPUT_DIR` | 결과 저장 경로 | `detection_results_live/` 또는 `_captured/` |
| `NEOMSCOPE_IMG_SIZE` | 입력 사이즈 | `640` |

`pyproject.toml`에 default 정의, CLI `--config`로 override 가능.

### 10.5 Tooling

| 도구 | 설정 위치 | 역할 |
|-----|----------|------|
| `ruff` | `pyproject.toml` `[tool.ruff]` | lint + import sort |
| `black` | `pyproject.toml` `[tool.black]` | format (line=100) |
| `mypy` | `pyproject.toml` `[tool.mypy]` | (선택) type check, strict=False |
| `pytest` | `pyproject.toml` `[tool.pytest.ini_options]` | unit + integration |

---

## 11. Implementation Guide

### 11.1 File Structure (Plan §6.3 보완)

```
NeoMscope/
├── pyproject.toml                          # 프로젝트 메타·의존성·도구 설정
├── setup.sh                                # Pi 5 부트스트랩 (FR-10)
├── README.md                               # 학습·배포·시연 가이드
├── CLAUDE.md
├── .gitignore
├── docs/                                    # PDCA 문서
├── training/
│   ├── __init__.py
│   ├── train.py                            # FR-03 진입점
│   ├── eval.py                             # 베이스라인 비교
│   ├── data.yaml                           # 5-class 정의
│   └── notebooks/
│       └── train-yolo11-seg.ipynb
├── tools/
│   ├── __init__.py
│   ├── inspect_coco.py                     # §4.2 분기 결정
│   ├── convert_coco_to_yolo.py             # FR-01
│   ├── validate_dataset.py                 # FR-02
│   ├── visualize_labels.py                 # 시각 검수
│   ├── export_onnx.py                      # ONNX export + 검증
│   └── compile_hef.sh                      # HEF 컴파일
├── inference/
│   ├── __init__.py
│   ├── types.py                            # Detection, PipelineConfig dataclass
│   ├── pipeline.py                         # HailoInferencePipeline
│   ├── postprocess.py                      # decode + render
│   ├── _gst.py                             # GStreamer 래퍼 (private)
│   ├── _camera.py                          # 카메라 추상화 (private)
│   ├── live_detect.py                      # FR-05 진입점
│   ├── batch_detect.py                     # FR-06 진입점
│   └── capture_and_detect.py               # FR-07 진입점
├── tests/
│   ├── unit/
│   │   ├── test_convert_coco_to_yolo.py
│   │   ├── test_postprocess.py
│   │   └── test_validate_dataset.py
│   ├── integration/
│   │   ├── test_onnx_parity.py
│   │   └── fixtures/
│   │       └── tiny_coco.json              # 3 이미지·5 어노테이션 미니 데이터셋
│   └── conftest.py
├── models/
│   ├── pt/         (gitignore)
│   ├── onnx/       (gitignore)
│   └── hef/        (gitignore)
├── datasets/                                # (gitignore)
│   └── onioncell/
│       ├── images/{train,val,test}/
│       ├── labels/{train,val,test}/
│       ├── calib_imgs/
│       └── data.yaml -> ../../training/data.yaml
├── captured_raw_images/                     # 추론 샘플 (commit)
├── detection_results_live/      .gitkeep
├── detection_results_captured/  .gitkeep
└── legacy/                                  # 기존 Mask R-CNN (이전 단계)
    ├── README.md
    ├── detect_onioncell.py
    ├── live_detect.py
    ├── capture_and_detect.py
    ├── config_onion_20201022.py
    └── dataset_config.py
```

### 11.2 Implementation Order (의존성 그래프 기반)

> 같은 그룹 내 항목은 병렬 작업 가능. 그룹 간은 순차.

**그룹 0: 환경·이전** (Day 0–1)
1. [ ] `pyproject.toml` 작성 (deps, ruff, black, pytest 설정)
2. [ ] `setup.sh` 작성 (Pi 5 부트스트랩)
3. [ ] 기존 코드 `legacy/`로 이동 (git mv)
4. [ ] `legacy/README.md` 작성 (사용 안내)

**그룹 1: 부트스트랩 라벨링·검증** (Day 1–3)
5. [ ] `inference/types.py` (Detection, PipelineConfig) — 다른 모듈이 의존
6. [ ] **레거시 venv 구성** (Python 3.8 + TF 1.15 + mrcnn) — R-12 폴백 트리거 지점
7. [ ] `tools/bootstrap_labeling.py` (기존 모델로 자동 추론 → YOLO det `.txt`)
8. [ ] Roboflow에 자동 라벨 결과 업로드 + 시각 검수·보정 (수작업)
9. [ ] Roboflow에서 보정 완료 데이터 export (YOLO det 형식, train/val/test 70/20/10)
10. [ ] `tools/validate_dataset.py`
11. [ ] `tools/visualize_labels.py` (시각 검수 보조)
12. [ ] `training/data.yaml` (5-class 정의)
13. [ ] **Gate 1** (Plan §8): 시각 검수 통과율 ≥ 98%

**그룹 2: Kaggle 학습·내보내기** (Day 3–5)
14. [ ] Roboflow export 데이터셋을 GitHub Release로 업로드 (또는 Kaggle Dataset)
15. [ ] `training/notebooks/train-yolo11-det-kaggle.ipynb` 작성
16. [ ] Kaggle에서 학습 실행 → best.pt 산출 → 로컬 다운로드
17. [ ] `training/eval.py` (베이스라인 비교)
18. [ ] `tools/export_onnx.py` 실행 (학습 PC Windows Python)
19. [ ] **Gate 2** (Plan §8): mAP ≥ 베이스라인 90%

**그룹 3: HEF 컴파일 (WSL2)** (Day 5–6)
20. [ ] **WSL2 + Ubuntu 22.04 + Hailo DFC 4.x+ 환경 구성** (R-13)
21. [ ] Calibration set 64-128장 추출 (Roboflow 검수 통과 train set 무작위)
22. [ ] `tools/compile_hef.sh` (WSL2 안에서 실행)
23. [ ] HEF sanity (Pi 5에서 단일 이미지 추론)
24. [ ] **Gate 3** (Plan §8): HEF 정상 추론

**그룹 4: 추론 모듈** (Day 6–8)
25. [ ] `inference/_camera.py` (USB/CSI/파일 추상화)
26. [ ] `inference/postprocess.py` (det decode + render) + 단위 테스트
27. [ ] `inference/pipeline.py` (HailoInferencePipeline)
28. [ ] `inference/_gst.py` (필요 시)
29. [ ] `inference/live_detect.py` (FR-05)
30. [ ] `inference/batch_detect.py` (FR-06)
31. [ ] `inference/capture_and_detect.py` (FR-07)
32. [ ] **Gate 4** (Plan §8): FPS NFR 충족 (≥40 FPS at s)

**그룹 5: 통합·문서** (Day 8–10)
33. [ ] `tests/integration/test_onnx_parity.py`
34. [ ] `README.md` 갱신 (학습 + 배포 + 시연)
35. [ ] 신규 SD카드 재현 검증
36. [ ] 시연 시나리오 리허설 ×5
37. [ ] `docs/03-analysis/aihat-yolo-port.benchmark.md`
38. [ ] **Gate 5** (Plan §8): Match Rate ≥ 90%

### 11.3 Rollback Strategy

| 실패 단계 | Rollback |
|----------|---------|
| 그룹 1 (부트스트랩) | 레거시 venv 구성 실패 → **Roboflow 수작업 라벨링으로 즉시 전환** (R-12). 1-3일 추가. |
| 그룹 1 (라벨 품질) | 부트스트랩 결과 노이즈 많음 → 신뢰도 임계값 상향(0.5→0.7) + 수작업 보정 비율 증가 |
| 그룹 2 (학습) | mAP 미달 시 (a) augmentation 강화 (b) `m` 사이즈 (c) Roboflow 보정 더 (d) 데이터 추가 |
| 그룹 3 (WSL2) | WSL2 막힘 → Hailo 공식 Docker 이미지 (R-13) |
| 그룹 3 (HEF) | DFC 컴파일 실패 → (a) opset 변경 (b) ONNX 재export (c) **YOLOv8-det 백업으로 분기** |
| 그룹 4 (추론) | GStreamer 막힐 시 (a) `hailo_apps_infra` Python 래퍼 (b) HailoRT 직접 + OpenCV 카메라 |
| 모든 단계 | `legacy/` 그대로 살아있어 시연만큼은 폴백 가능 (PC + 기존 Mask R-CNN) |

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-01 | Initial design draft (YOLOv11-seg + Hailo-10H 기반). §1.3에 dataset_config bbox-as-mask 발견 명시 + §4.2 분기점 추가. | domafordarwin / Claude |
| 0.2 | 2026-05-01 | **3대 변경 반영**: ① **det 확정** (seg→det, §1.3.1). ② **Kaggle 학습 환경** (P2000 4GB 부족, §1.3.3). ③ **부트스트랩 라벨링** (COCO 부재, §1.3.2). §2.1 컴포넌트 다이어그램 학습PC↔Kaggle 분리, §4 전체 재작성, §5.2 postprocess 단순화, §11.2 그룹 1 부트스트랩 워크플로로 교체. | domafordarwin / Claude |
