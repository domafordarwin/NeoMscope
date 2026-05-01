# Datasets

이 폴더는 YOLO 학습용 데이터셋이 위치하는 곳입니다. **내용물은 git에 커밋되지 않습니다** (`.gitignore`).

## 구조

```
datasets/
└── onioncell/
    ├── images/
    │   ├── train/   ← 학습용 (70%)
    │   ├── val/     ← 검증용 (20%)
    │   └── test/    ← 최종 평가용 (10%)
    ├── labels/
    │   ├── train/   ← YOLO det `.txt`
    │   ├── val/
    │   └── test/
    └── calib_imgs/  ← Hailo DFC INT8 calibration용 (train set 무작위 64-128장)
```

## 생성 방법

이 데이터셋은 **두 단계 워크플로**로 생성됩니다:

### 1단계: 부트스트랩 라벨링 (학습 PC에서)

```bash
# 레거시 venv 활성화
source .legacy-venv/Scripts/activate

# 기존 Mask R-CNN으로 자동 라벨링 (raws/JPEG_Export_Data/ + captured_raw_images/)
python tools/bootstrap_labeling.py \
    --weights weights/mask_rcnn_onioncell_1020_0089.h5 \
    --images raws/JPEG_Export_Data captured_raw_images \
    --output datasets/onioncell/labels/auto/ \
    --conf-min 0.5 \
    --review-min 0.5
```

### 2단계: Roboflow 검수 + Export

1. `datasets/onioncell/images/` + `labels/auto/`를 zip으로 묶어 Roboflow 프로젝트에 업로드
2. Roboflow Annotate에서 시각 검수 + 보정 (특히 `bootstrap-report.json`의 review_files)
3. Roboflow에서 **Generate** → YOLOv11/v8 Detect 형식 export
4. zip 다운로드 → `datasets/onioncell/`에 풀어서 위 구조로 배치

## 원본 이미지

- **`raws/JPEG_Export_Data/`** (105장, 2020-10-07): 메인 학습 데이터. **3.2GB, gitignored**.
- **`captured_raw_images/`** (22장, 2021-2022): 추가 학습/테스트 데이터. 작아서 git에 포함됨.

원본 데이터는 본 저장소에 포함되지 않으므로, 신규 머신에서 학습을 재현하려면 별도로 받아야 합니다 (외장 SSD 또는 클라우드 드라이브).
