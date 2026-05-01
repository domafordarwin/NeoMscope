# Legacy — Mask R-CNN onion cell detector (v1.0, 2020-2022)

이 폴더는 NeoMscope v1.0 시절 사용된 **Mask R-CNN(TF1.x, Matterport) 기반** 코드입니다.
v2.0(YOLOv11-det + Pi 5 + AI HAT+ 2)으로 마이그레이션하면서 보존했습니다.

## 보존 이유

1. **부트스트랩 라벨링의 입력**: [tools/bootstrap_labeling.py](../tools/bootstrap_labeling.py)가 이 코드의 추론 결과를 학습 라벨로 사용합니다.
2. **시연 폴백**: 새 시스템에 문제 발생 시 PC + 기존 모델로 시연 가능.
3. **비교 자료**: 전람회에서 "기존 방식 vs 신 방식" 성능 비교 데모.

## 사용법 (별도 venv 필요)

이 코드는 Python 3.7-3.8 + TensorFlow 1.15 + `mrcnn` 패키지 환경에서만 동작합니다.
현 프로젝트의 메인 venv(.venv, Python 3.11)와 분리하여 별도 환경 구성:

```bash
# 학습 PC에서 (uv 활용)
uv venv --python 3.8 .legacy-venv
source .legacy-venv/Scripts/activate     # PowerShell: .legacy-venv\Scripts\Activate.ps1

uv pip install \
    "tensorflow==1.15.*" \
    "keras==2.2.5" \
    "numpy==1.19.*" \
    "scikit-image==0.17.*" \
    "opencv-python==4.5.*" \
    "Pillow" "imutils"

# Matterport mrcnn 설치 (PyPI 미배포)
uv pip install "git+https://github.com/matterport/Mask_RCNN.git"
```

가중치는 별도로 받아서 `weights/mask_rcnn_onioncell_1020_0089.h5`에 위치시키세요 (gitignored).

## 파일 안내

| 파일 | 용도 |
|------|------|
| `detect_onioncell.py` | 정지 이미지 배치 추론 |
| `live_detect.py` | USB 웹캠 실시간 추론 |
| `capture_and_detect.py` | 캡처 + 추론 통합 |
| `take_webcam_pictures.py` | 데이터 수집용 캡처 도구 |
| `config_onion_20201022.py` | Mask R-CNN OnionCellConfig (anchors, epochs, loss weights) |
| `dataset_config.py` | COCO-style 데이터셋 로더. **`load_mask`가 polygon이 아닌 bbox로 사각형 마스크를 생성**하는 것이 v2.0 마이그레이션 시 발견됨 — YOLO det 채택의 직접적 근거. |
| `train-onion-v2.ipynb` | 학습 노트북 |
| `evaluate-onion.ipynb` | 평가 노트북 |
| `detect-onion-v3.ipynb` | 추론 노트북 |
| `colors.txt` | 5-class 시각화 색상(BGR/RGB 혼재). v2.0에서는 `inference/postprocess.py`의 `CLASS_COLORS` 상수로 대체. |
| `object_detection_classes_onion.txt` | 클래스 이름 5종 (Inter, Pro, Meta, Ana, Telo). v2.0에서는 `training/data.yaml`이 source of truth. |

## ⚠️ 수정 금지

이 폴더의 파일은 **변경하지 마세요**. 부트스트랩 라벨링이 동일 입출력에 의존합니다.
새 기능은 `inference/`, `tools/`, `training/`에 추가하세요.

## 마이그레이션 변경 요약

| 항목 | Legacy (v1.0) | Current (v2.0) |
|------|---------------|----------------|
| 모델 | Mask R-CNN (Matterport, TF1.x) | YOLOv11-det (PyTorch, Ultralytics) |
| 출력 | bbox + 사각형 마스크 (사실상 bbox) | bbox + class |
| 추론 디바이스 | PC + GPU (또는 CPU) | Pi 5 + AI HAT+ 2 (Hailo-10H) |
| 입력 사이즈 | 512×512 | 640×640 |
| 학습 환경 | 로컬 GPU | Kaggle T4/P100 |
| Python | 3.7-3.8 | 3.11 |
| 패키지 | mrcnn, TF1.x | ultralytics, hailo-platform |
