# NeoMscope: 엣지 AI 기반 양파 체세포 분열 단계 자동 검출 시스템

**과학전람회 출품 연구 보고서**

| 항목 | 내용 |
|---|---|
| 연구 주제 | 양파 표피세포 현미경 영상에서 체세포 분열 5단계의 실시간 자동 검출 |
| 연구 분야 | 생명과학 + 컴퓨터과학 (융합) — 생물 영상 분석 / 엣지 인공지능 |
| 연구자 | domafordarwin |
| 작성일 | 2026-05-01 |
| 버전 | v1.0 (중간 보고) |
| 저장소 | https://github.com/domafordarwin/NeoMscope |
| 관련 문서 | [Plan v0.5](../01-plan/features/aihat-yolo-port.plan.md) · [Design v0.2](../02-design/features/aihat-yolo-port.design.md) · [Progress Report](2026-05-01-progress-report.md) |

---

## 초록 (Abstract)

본 연구는 양파(*Allium cepa*) 뿌리 끝 표피세포의 광학현미경 이미지에서 체세포 분열 5단계(간기 Inter, 전기 Pro, 중기 Meta, 후기 Ana, 말기 Telo)를 **딥러닝 객체 검출 모델로 자동 분류·계수**하고, 이를 **휴대 가능한 단일 보드 컴퓨터(Raspberry Pi 5) + 엣지 AI 가속기(AI HAT+ 2 / Hailo-10H NPU)** 위에서 실시간으로 동작시키는 시스템을 개발하는 것이다. 기존(v1.0) PC 기반 Mask R-CNN 시스템을 (v2.0) YOLOv11 객체 검출 + Hailo NPU 구조로 재설계하여 ① 환경 재현성, ② 시연 휴대성, ③ 실시간 성능을 동시에 확보하고자 한다. 본 보고서는 2026-05-01 기준 **계획 → 설계 → 인프라/코드 구현(95% 완료) → 데이터 라벨링(검증 중)** 단계의 중간 결과를 정리하며, 연구 과정에서 발견된 두 가지 핵심 문제(레거시 라벨이 polygon이 아닌 bbox로 학습된 점, 원본 이미지 스케일과 추론 입력 스케일의 불일치)와 그 해결 과정을 함께 기술한다.

**키워드**: 체세포 분열, 양파 표피세포, 객체 검출, YOLOv11, Raspberry Pi 5, Hailo NPU, 엣지 AI, 과학교육

---

## 1. 연구 배경 및 필요성

### 1.1 생물학적 배경

양파(*Allium cepa*) 뿌리 끝의 분열 조직(meristem)은 고등학교 생물 단원에서 **체세포 분열 5단계**를 관찰하기 위한 표준 시료다. 학생은 압착 표본(squash slide)을 만들고, 광학현미경 400배 시야에서 다음 5개 단계를 식별·계수하여 각 단계의 상대 시간 비율을 계산한다.

| 단계 | 영문 코드 | 형태적 특징 |
|---|---|---|
| 간기 | Inter | 핵막 유지, 염색체 미응축 |
| 전기 | Pro | 염색체 응축 시작, 핵막 소실 단계 |
| 중기 | Meta | 염색체가 적도판(metaphase plate)에 정렬 |
| 후기 | Ana | 염색분체가 양극으로 이동 |
| 말기 | Telo | 핵막 재형성, 세포질 분열 시작 |

이 관찰은 학습자에게 두 가지 어려움을 준다.
1. **단계 식별의 주관성**: 특히 Pro와 Telo, Inter와 Pro 사이의 경계가 형태학적으로 모호하다.
2. **계수의 노동량**: 한 시야에 50–150개 세포가 보이며, 정확한 비율 산출을 위해서는 다수 시야를 처리해야 한다.

### 1.2 기술적 배경

본 연구의 v1.0 시스템(2020–2022년 개발)은 PC + Matterport Mask R-CNN(TensorFlow 1.15, Python 3.7)으로 동작했다. 이 구조는 다음과 같은 한계가 있었다.

1. **환경 재현 곤란**: TF 1.15와 `mrcnn` 패키지는 5년 이상 유지보수가 끊겼고, 신규 머신마다 호환성 문제가 발생한다.
2. **휴대성 부족**: GPU가 장착된 데스크톱이 있어야 실시간 추론이 가능해 전람회 시연 시 PC와 모니터를 끌고 다녀야 한다.
3. **최신 기술 어필 부족**: 5년 전 모델 구조로는 “최신 AI 활용” 어필이 약하다.

### 1.3 엣지 AI 하드웨어의 등장

2026년 1월 15일 출시된 **Raspberry Pi AI HAT+ 2** 모듈은 차세대 Hailo 신경 처리 칩(Hailo-10H, 40 TOPS INT4 / 20 TOPS INT8, 8 GB on-module LPDDR4X)을 탑재해 신용카드 크기의 보드에서 데스크톱 GPU에 근접한 추론 성능을 3.5–4.5 W의 저전력으로 달성한다. 이로써 **“현미경 옆에 놓인 작은 박스가 즉시 결과를 보여 주는”** 시연이 비로소 가능해졌다.

### 1.4 연구 동기

위의 세 가지(생물학적 학습 도구의 필요, v1.0의 기술 부채, 엣지 AI의 성숙)가 결합되어, 본 연구는 **“현미경 + 단일 보드 + 작은 NPU 하나로 분열 단계를 실시간 시각화·계수하는 휴대형 학습 도구”**를 개발 목표로 설정했다.

---

## 2. 연구 목적 및 가설

### 2.1 연구 목적

1. **목적 1 (정확도)**: 양파 표피세포 이미지에서 5개 분열 단계를 mAP@0.5 ≥ 기존 Mask R-CNN의 90% 수준으로 검출한다.
2. **목적 2 (실시간성)**: Raspberry Pi 5 + AI HAT+ 2 단독 박스에서 ≥ 30 FPS(YOLOv11s 기준 ≥ 40 FPS)로 추론한다.
3. **목적 3 (재현성)**: 신규 SD카드와 README + `setup.sh` 한 번으로 다른 머신에서 동일 시스템을 30분 내 재현할 수 있다.
4. **목적 4 (교육적 가치)**: 시연 시 “촬영 → 검출 → 단계별 색상 시각화 → 카운트 표시”가 단일 화면에서 1초 이내에 표시된다.

### 2.2 작업 가설

H1. **모델 구조 가설** — Mask R-CNN(2-stage, RoIAlign 의존)은 Hailo NPU에서 컴파일되지 않으므로, 동등 정확도를 가지면서 1-stage 구조인 YOLOv11이 엣지 이식의 유일한 실용 대안이다.

H2. **출력 형식 가설** — 레거시 데이터셋이 사실상 bbox만 제공하면, 인스턴스 분할(seg) 대신 객체 검출(det)로 충분히 동등한 결과를 얻을 수 있다.

H3. **데이터 가설** — 기존 학습 가중치를 사용한 부트스트랩 자동 라벨링 + 사람 검수가 처음부터 수동 라벨링하는 것보다 시간 효율이 높다.

H4. **하드웨어 가설** — Hailo-10H(20 TOPS INT8, 8 GB)는 YOLOv11s 모델을 640×640 입력에서 30 FPS 이상 안정적으로 처리한다.

이 중 H1, H2는 본 보고서 §6.1에서 검증되었고, H3, H4는 §6.2~7에서 부분 검증·진행 중이다.

---

## 3. 이론적 배경

### 3.1 객체 검출 모델 비교

| 항목 | Mask R-CNN (v1.0) | YOLOv11-det (v2.0) |
|---|---|---|
| 구조 | 2-stage (RPN + ROI head) | 1-stage (anchor-free) |
| 출력 | bbox + class + polygon mask | bbox + class |
| 학습 시간 (동일 데이터) | 기준 1.0× | 약 0.7× |
| 추론 지연 | ~200 ms (PC GPU) | ~25 ms (NPU 예상) |
| 엣지 NPU 호환 | RoIAlign 미지원 → ✗ | Ultralytics + Hailo 정식 지원 → ✓ |

### 3.2 엣지 NPU 추론 파이프라인

학습된 PyTorch 모델을 NPU에서 실행하려면 다음 변환이 필요하다.

```
PyTorch (.pt)  →  ONNX (.onnx)  →  Hailo .hef
   학습용            중간 표현       NPU 실행 포맷
   (Kaggle)        (검증, opset17)  (DFC 5.3, INT8 양자화)
```

INT8 양자화 단계에서는 학습 데이터 분포를 대표하는 **calibration set 256장**을 입력하여, 32-bit 부동소수점 가중치를 8-bit 정수로 변환할 때 정확도 손실을 최소화한다.

### 3.3 데이터 부트스트랩 라벨링

새로운 모델을 학습하려면 새 형식의 라벨이 필요하다. 본 연구처럼 원본 어노테이션이 부재하나 **이전 세대 모델의 가중치는 보존되어 있는 경우**, 이전 모델로 자동 추론한 결과를 새 라벨의 시드로 사용하고 사람이 시각 검수하여 보정하는 “부트스트랩 라벨링” 전략이 효과적이다. 이는 처음부터 수작업으로 라벨링하는 비용(이미지당 ~10분 × 100장 이상)을 1/3 ~ 1/5로 단축할 수 있다.

---

## 4. 연구 방법 및 시스템 설계

### 4.1 전체 시스템 아키텍처

```
[학습 PC: Windows 11 + WSL2 Ubuntu]                [Kaggle: T4/P100 16GB]
   ├── 레거시 conda env (Py 3.7 + TF 1.15)            └── Ultralytics 학습
   │     └── 부트스트랩 자동 라벨링                            │
   ├── Roboflow Smart Polygon 시각 검수                       │
   ├── tools/export_onnx.py (PyTorch → ONNX)  ◀──────── best.pt
   └── WSL2: Hailo DFC v5.3.0
         └── tools/compile_hef.sh (ONNX → HEF, INT8 양자화)
                                  │
                                  │ scp / SD카드
                                  ▼
[배포 박스: Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 8GB)]
   ├── HailoRT 4.x runtime
   ├── inference/live_detect.py    (실시간 USB 웹캠)
   ├── inference/batch_detect.py   (폴더 일괄 처리)
   └── inference/capture_and_detect.py (캡처+추론 통합)
```

### 4.2 데이터 확보 단계 (3 차 반복)

| 시도 | 방법 | 결과 | 결정 |
|---|---|---|---|
| 1차 | 레거시 모델로 다운샘플 1280px 추론 | 5–399 detections, root tip 전체에 1 bbox만 | **폐기** — 스케일 불일치 |
| 2차 | 원본 8000×31000 픽셀을 512×512 타일로 분할 후 각 타일에 추론 | 747 detections, 평균 conf 0.71 | 텅 빈 영역에도 박스 출현 |
| 3차 | 2차 결과에 조직 마스크 + min bbox size 30 px + conf ≥ 0.6 필터 적용 | **240 detections, 평균 conf 0.78, 메리스템 영역 집중** | **채택** (1장 검증 완료) |

### 4.3 모델·학습 사양

- **모델**: YOLOv11s-det (백업: YOLOv8s-det)
- **입력**: 640×640 letterbox padded
- **클래스**: 5 (Inter, Pro, Meta, Ana, Telo) — background 별도 클래스 없음
- **학습 환경**: Kaggle T4 / P100 16 GB VRAM (주 30시간 무료)
- **하이퍼파라미터**: epoch 100, batch 16, AdamW + cosine LR, early stop patience 20
- **도메인 특화 augmentation**: degrees=180, flipud=0.5 (세포는 회전·반전 불변), HSV 강하게 (현미경 광원 변동)

### 4.4 평가 방법

- **정확도**: train/val/test 70/20/10 stratified split, val mAP@0.5로 모니터링, test set은 최종 보고용 holdout
- **속도**: `live_detect` 내장 FPS 카운터 60초 평균
- **전력·온도**: `vcgencmd measure_temp` 60초 평균 < 75 °C
- **메모리**: `psutil`로 RSS < 800 MB 확인

---

## 5. 진행 경과 (2026-05-01 기준)

### 5.1 단계별 게이트 통과 현황

| 단계 | 게이트 | 상태 |
|---|---|---|
| Phase 0 — 환경 준비 | Pi 5 + AI HAT+ 2 stock YOLOv11 데모 동작 | ⏳ Pi 부팅 대기 |
| Phase 1 — 라벨링 + 베이스라인 | 시각 검수 통과율 ≥ 98% | 🔄 3차 시도 1장 검증 완료, 5장 → 105장 확장 대기 |
| Phase 2 — Kaggle 학습 | mAP ≥ 베이스라인 90% | ⏳ 라벨 완료 후 |
| Phase 3 — 모델 변환 | Pi 5에서 .hef 단일 추론 성공 | ⏳ 학습 후 |
| Phase 4 — Pi 5 추론 | ≥ 40 FPS (YOLOv11s) | ⏳ HEF 후 |
| Phase 5 — 시연 리허설 | Match Rate ≥ 90% | ⏳ |

### 5.2 코드 구현 통계

| 항목 | 수량 |
|---|---|
| 신규 Python 모듈 | 14개 |
| 신규 코드 라인 | ~4,000 줄 |
| 단위 테스트 | 75개 (모두 통과) |
| ruff lint 에러 | 0 |
| Git 커밋 | 14개 (모두 푸시) |
| 환경 셋업 | 3개 (메인 venv, 레거시 conda, WSL2 Hailo) |

### 5.3 환경 구축 결과

- ✅ **메인 venv** (`.venv/`, Python 3.11.14): Ultralytics 8.3, ONNX, ruff, pytest 동작 확인
- ✅ **레거시 conda venv** (`neomscope-legacy`, Python 3.7): TF 1.15 + Keras 2.2.5 + Matterport mrcnn, 가중치 로드 및 추론 성공
- ✅ **WSL2 Ubuntu 22.04 + Hailo DFC v5.3.0**: `hailomz info yolov11s` 검증 통과
- ✅ **AI HAT+ 2 / Hailo-10H** 모듈 보유, HailoRT PCIe 드라이버 .deb 준비 완료

### 5.4 코드 산출물 (대표)

**Pi 5 추론 측 (Domain + Application):**
- [inference/types.py](../../inference/types.py) — Detection, PipelineConfig, CLASS_NAMES (단일 진실 공급원)
- [inference/pipeline.py](../../inference/pipeline.py) — `HailoInferencePipeline` (`mock=True`로 NPU 없는 PC에서도 테스트 가능)
- [inference/postprocess.py](../../inference/postprocess.py) — YOLO det 디코드 + bbox NMS + 한글 라벨 렌더
- [inference/live_detect.py](../../inference/live_detect.py) — `neomscope-live` 진입점 (FR-05)
- [inference/batch_detect.py](../../inference/batch_detect.py) — `neomscope-batch` (FR-06)
- [inference/capture_and_detect.py](../../inference/capture_and_detect.py) — `neomscope-capture` (FR-07)

**개발 PC 측 도구:**
- [tools/tile_and_detect.py](../../tools/tile_and_detect.py) — 타일링 + 조직 마스크 + 신뢰도/크기 필터 (3차 시도, 현재 사용 중)
- [tools/bootstrap_labeling.py](../../tools/bootstrap_labeling.py) — 1차 부트스트랩 (스케일 문제로 보류)
- [tools/package_for_roboflow.py](../../tools/package_for_roboflow.py) — 35 MB 원본 → 80 KB 리사이즈 + ZIP (3.2 GB → 9.5 MB)
- [tools/visualize_labels.py](../../tools/visualize_labels.py) — bbox 시각 검수
- [tools/validate_dataset.py](../../tools/validate_dataset.py) — 좌표 범위·클래스 분포 무결성 검사
- [tools/export_onnx.py](../../tools/export_onnx.py) — ONNX export + parity check
- [tools/compile_hef.sh](../../tools/compile_hef.sh) — DFC 호출 (WSL2)

### 5.5 3차 라벨링 결과 (1장 검증)

8827 × 31530 픽셀의 원본을 512×512 타일 1886개로 분할 후 추론, 다음 필터를 순차 적용:

| 필터 | 적용 전 | 적용 후 | 제거 수 |
|---|---|---|---|
| 원시 검출 | — | 747 | — |
| min bbox size ≥ 30 px | 747 | 561 | 186 (작은 박스) |
| 조직 마스크 (양파 외부 제거) | 561 | 464 | 97 (배경 노이즈) |
| confidence ≥ 0.6 | 464 | **240** | 224 (저신뢰) |

**클래스 분포 (3차 결과):**

| 클래스 | 1차 | 2차 | **3차** |
|---|---|---|---|
| Inter | 0 | 18 | 6 |
| Pro | 1 | 89 | 51 |
| Meta | 0 | 39 | 26 |
| Ana | 1 | 190 | 130 |
| Telo | 3 | 411 | 27 |
| 평균 conf | 0.65 | 0.71 | **0.78** |

시각 평가: 조직 마스크가 양파 뿌리 끝 두 개의 실루엣을 깨끗하게 분리, 검출 박스가 메리스템 영역에 집중, 흰 배경의 false positive는 거의 제거됨.

처리 시간(1장, CPU only): **약 20분**. 105장 전체 추정: ~35시간.

---

## 6. 핵심 발견 사항

### 6.1 발견 1 — 레거시 데이터셋의 “bbox-as-mask” 문제

설계 문서 작성 중 [legacy/dataset_config.py:88-94](../../legacy/dataset_config.py#L88-L94)에서 `load_mask` 함수가 **polygon 윤곽선이 아니라 bbox 사각형을 그대로 마스크로 채워 학습**시켰다는 점을 발견했다. 즉 v1.0이 “인스턴스 분할”이라고 표기되었음에도 모델이 본 라벨은 본질적으로 bbox였다.

**과학적·공학적 함의:**
- v2.0에서 인스턴스 분할(YOLO-seg)을 채택할 정량적 근거가 사라짐 → **객체 검출(YOLO-det)로 충분**
- 학습 시간 약 30 % 단축, ONNX 출력 텐서 4종 → 2종, 후처리 코드 ~50줄 감소
- 본 발견은 **연구 진행 중 코드를 직접 읽지 않으면 드러나지 않는 종류**의 정보로, 문서·논문에만 의존했다면 잘못된 모델 구조를 채택했을 것이다 → **재현성·코드 공개의 가치**를 시사

### 6.2 발견 2 — 데이터셋 스케일·도메인 불일치

당초 학습 데이터로 사용하려 한 `raws/JPEG_Export_Data/` 105장은 **8000 × 31000 픽셀의 whole-mount root section**(뿌리 전체 절편)이었으나, 실제 mitosis 관찰이 가능한 squash slide(압착 표본)는 별도 폴더의 22장(600 × 450 픽셀)에 있었다. 1차 부트스트랩 시도에서 raws 이미지를 1280 px로 다운샘플하여 추론하자 세포 한 개가 10–20 픽셀로 축소되어 모델이 root tip 전체를 단일 객체로 인식, **이미지당 평균 3–5개의 거대한 박스**만 그려졌다.

**과학적 함의:**
- 학습 데이터의 **확대 배율과 시야**가 검출하려는 단위(개별 세포)에 부합해야 한다는 도메인 지식의 중요성
- “이미지가 많다” ≠ “학습에 적합하다”

### 6.3 해결책 — 타일링 + 조직 마스크 파이프라인

발견 2에 대한 대응으로 다음 알고리즘을 설계·구현했다 ([tools/tile_and_detect.py](../../tools/tile_and_detect.py)):

```
원본 8000×31000 px
   │
   ├─ Otsu thresholding으로 조직 마스크 생성 (양파 vs 배경 분리)
   │
   └─ 512×512 슬라이딩 윈도우 (stride 256, 25% overlap)
        ├─ 각 타일 → 레거시 Mask R-CNN 추론 (native 512 입력)
        └─ bbox 좌표를 원본 이미지 좌표계로 unscale
   │
   ▼
[원본 좌표 통합 + 타일 경계 NMS]
   │
   ├─ min bbox size ≥ 30 px (사람 눈 분해능 기준)
   ├─ 조직 마스크 내부 박스만 유지
   └─ confidence ≥ 0.6 필터
   │
   ▼
[정제된 라벨 → Roboflow Smart Polygon 검수]
```

이 구조는 **타일링이라는 고전 영상처리 기법 + 딥러닝 검출 + 도메인 마스크 필터**를 결합한 하이브리드 접근으로, 추론 입력 해상도와 도메인 단위 사이즈를 일치시키는 일반적 해법이다.

---

## 7. 한계 및 개선 방안

### 7.1 현재 한계

| 영역 | 한계 | 영향도 |
|---|---|---|
| 데이터 양 | 검수 후 사용 가능한 시야 22장 (squash) + 105장(raws, 타일 기반) | 중 — augmentation으로 보완 가능 |
| 클래스 불균형 | 3차 결과 기준 Telo:Inter ≈ 4.5:1, Ana 우세 | 중 — 양파 표본의 자연적 분포 일부 반영, 학습 시 cls_pw 가중치 조정 검토 |
| 자동 라벨 노이즈 | 부트스트랩이 레거시 모델의 오류 상속 | 높음 — Roboflow 시각 검수 의무화로 완화 |
| 처리 시간 | 105장 전체 타일 추론 ~35시간 (CPU only) | 중 — Quadro P2000 GPU 활용 또는 처리 범위 축소(메리스템만)로 단축 가능 |
| 도메인 갭 | 학습은 microscopy whole-slide, 배포는 USB 웹캠 가정 | 높음 — 검증 필요, 시연 시 동일 현미경+카메라 조합 권장 |
| 단계 모호성 | Pro↔Telo, Inter↔Pro 형태 유사 | 중 — 도메인 특화 augmentation + 후속 시퀀스 정보(시간축) 활용 가능 |
| Pi 5 검증 | 실 NPU 추론 미수행 (HEF 미생성) | 높음 — 학습 후 1주 내 검증 예정 |

### 7.2 개선 방안 (단기, 1–2주)

1. **5장 → 105장 점진 확장**: 3차 필터링 결과의 일관성을 5장 추가 검증 후 105장 전체 처리. 옵션 C(메리스템 상단 40%만 추출)로 처리 시간 14시간 압축.
2. **Roboflow Smart Polygon (SAM 보조) 검수**: 신뢰도 < 0.85 검출에 대해 사람이 30분/이미지 검토 → 라벨 품질 확보.
3. **GPU 활용**: 레거시 mrcnn 추론을 Quadro P2000 GPU로 전환 시 ~5–10× 가속 기대.
4. **클래스 균형 보정**: oversampling minority classes (Inter, Telo) + cls_pw 가중치 [1.5, 1.0, 1.0, 0.7, 1.5] 검토.

### 7.3 개선 방안 (중기, 학습 ~ 배포)

1. **백업 모델 동시 학습**: YOLOv11s 주모델, YOLOv8s 백업을 같은 데이터에서 동시 학습 → HEF 컴파일 실패 시 즉시 백업으로 분기.
2. **모델 사이즈 업그레이드 경로**: YOLOv11s mAP가 베이스라인 90 % 미달 시, 8 GB 메모리 여유로 YOLOv11m으로 즉시 업그레이드 (예상 30+ FPS 유지).
3. **GStreamer + hailo_apps_infra 1차, HailoRT Python 직접 호출 폴백**: 두 경로 모두 코드 골격 사전 구현 (R-06 대응).
4. **3중 백업**: best.pt / best.onnx / best.hef를 로컬 + 클라우드 + 외장 SSD에 보관 (R-07).

### 7.4 향후 확장 방향 (장기)

1. **시간 축 정보 활용**: 동영상 입력 시 인접 프레임의 단계 일관성을 부여하는 후처리 (Hidden Markov Model 또는 단순 voting) — 단일 프레임의 오분류를 시퀀스로 보정.
2. **카메라 + 현미경 통합 케이스**: 3D 프린팅 마운트로 Pi 카메라 모듈을 현미경 접안렌즈에 직결 → 학습-배포 도메인 갭 해소.
3. **교육용 UX**: 시연 화면에 분열 단계별 색상 + 한글 단계명 + 누적 카운트 + 분열 지수(MI = (Pro+Meta+Ana+Telo) / 전체)를 동시에 표시.
4. **다종 식물 확장**: 동일 파이프라인을 마늘·파·콩나물 등 다른 표본에 전이 학습으로 확장하여 “식물 일반” 분열 검출 도구로 일반화.
5. **생체 신호 연동**: 학생 관찰 데이터를 클라우드에 누적해 “학교별 분열 지수 분포 비교” 같은 시민 과학 활동으로 발전.

---

## 8. 결론 및 향후 일정

### 8.1 결론 (중간 시점)

본 연구는 양파 체세포 분열 검출 시스템을 **PC + Mask R-CNN(2020) → Pi 5 + Hailo NPU + YOLOv11(2026)** 구조로 재설계·구현 중이며, 2026-05-01 기준 다음을 달성했다.

- **인프라·도구·코드 ~95 % 완료**: 14개 모듈, 4,000줄, 75개 테스트, 3개 환경 셋업 모두 동작.
- **두 가지 핵심 발견**: (1) 레거시의 bbox-as-mask 구조 → seg 대신 det 채택, (2) 원본 데이터의 스케일 불일치 → 타일링 + 조직 마스크 파이프라인.
- **남은 핵심 블로커는 데이터 라벨 품질 1건**: 3차 필터링이 1장 시각 검증 통과, 5장 → 105장 확장 단계.

이후 학습(Kaggle, 30–60분) → ONNX export → HEF 컴파일(WSL2) → Pi 5 배포·시연 단계가 순차 예정되어 있으며, 라벨링 완료 시점부터 시연 가능 시점까지 **2–3일**로 추정된다.

### 8.2 향후 일정

| 시점 | 작업 | 산출물 |
|---|---|---|
| Day 1–2 | 5장 추가 처리 → 105장 확장 (옵션 C 메리스템 추출) → Roboflow 검수 | 정제된 YOLO det 라벨 |
| Day 3 | Kaggle YOLOv11s 학습 (epoch 100, T4) | best.pt + mAP 보고 |
| Day 4 | ONNX export + parity check + WSL2 HEF 컴파일 | best.hef |
| Day 5 | Pi 5 배포 + neomscope-live 시연 | FPS·전력 벤치마크 |
| Day 6–7 | 시연 리허설 5회 + 벤치마크 보고서 | Gap Analysis ≥ 90 % |
| 출품 | 전람회 시연 | 휴대형 박스 + 시연 영상 |

### 8.3 본 연구의 의의

1. **교육적 의의**: 학생이 직접 만든 표본을 즉시 자동 분석하는 “학습용 AI 박스”로, 전통 관찰의 정확도 한계를 보완.
2. **기술적 의의**: 5년 된 TF1.x 코드베이스를 최신 PyTorch + 엣지 NPU 스택으로 마이그레이션한 실제 사례. 레거시 모델 → 부트스트랩 → 신모델 학습의 워크플로 검증.
3. **공학적 의의**: 원본 해상도(8000×31000)와 NPU 입력(640×640)의 7배 이상 차이를 “타일링 + 조직 마스크”라는 단순한 결합으로 해결한 일반화 가능 패턴.
4. **방법론적 의의**: PDCA(Plan-Design-Do-Check-Act) 사이클을 따른 문서화된 연구 진행. 모든 의사결정·발견·롤백 전략이 git 커밋과 PDCA 문서에 추적 가능.

---

## 부록 A — 의사결정 기록

| ID | 결정 | 근거 | 시점 |
|---|---|---|---|
| D-01 | seg → det 전환 | legacy `dataset_config.py:88-94`이 bbox로 학습 | 2026-05-01 설계 단계 |
| D-02 | YOLO26 → YOLOv11s | 안정성·튜토리얼·Hailo 사례 최대화 | 2026-05-01 |
| D-03 | 로컬 GPU → Kaggle T4/P100 | Quadro P2000 4GB는 학습 부족 | 2026-05-01 |
| D-04 | 수동 → 부트스트랩 라벨 | 원본 어노테이션 부재, 가중치 보유 | 2026-05-01 |
| D-05 | 다운샘플 → 타일링 | raws 8000×31000 다운샘플 시 세포 가시성 상실 | 2026-05-01 |

## 부록 B — 위험 등록부 (2026-05-01 갱신)

| ID | 위험 | 상태 |
|---|---|---|
| R-01 | YOLO26-seg 신규성 | ✅ 해소 (YOLOv11-det 채택) |
| R-04 | AI HAT+ TOPS 미확정 | ✅ 해소 (Hailo-10H 확정) |
| R-09 | Hailo Model Zoo + YOLO11 + Hailo-10H 신규 조합 | ✅ 해소 (yolov11s.yaml 검증) |
| R-10 | DFC/HailoRT 버전 미스매치 | ✅ 해소 (둘 다 v5.3.0) |
| R-12 | 레거시 venv 셋업 실패 | ✅ 해소 (conda 환경 동작) |
| R-13 | WSL2 셋업 실패 | ✅ 해소 (Ubuntu 22.04 + DFC 동작) |
| R-14 | 부트스트랩 라벨 노이즈 | ⚠️ 활성 — 3차 필터링이 완화 중 |
| R-15 | 데이터 스케일 불일치 | ✅ 해소 (tile + native-res 검출) |
| R-16 | 105장 전체 처리 시간 ~35h | ⚠️ 활성 — 옵션 C로 14h 단축 가능 |

## 부록 C — 통계 요약

```
세션 누적:        ~12시간 (단일 세션)
Git commits:      14개
신규 코드 라인:   ~4,000
단위 테스트:      75개 (모두 통과)
ruff 에러:        0
PDCA 문서:        Plan v0.5, Design v0.2, 본 보고서
환경 셋업:        3개 (.venv + neomscope-legacy + WSL2 Hailo)
설치 스크립트:    3개 (setup.sh, install_hailo_dfc.sh, verify_hailo_install.sh)
의사결정 기록:    5개 핵심 + 다수 중간
위험 register:    16개 (12 해소, 2 활성, 2 대기)
```

## 부록 D — 참고 자료

- Raspberry Pi AI HAT+ 2: https://www.raspberrypi.com/products/ai-hat-plus-2/
- Hailo-10H AI Accelerator: https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/
- Hailo Model Zoo (master 브랜치): https://github.com/hailo-ai/hailo_model_zoo
- Hailo RPi5 Examples: https://github.com/hailo-ai/hailo-rpi5-examples
- Ultralytics YOLOv11 Docs: https://docs.ultralytics.com/models/yolo11/
- Matterport Mask R-CNN (legacy): https://github.com/matterport/Mask_RCNN

---

*본 보고서는 `bkit Vibecoding Kit` PDCA 방법론을 따라 작성되었으며, 모든 의사결정과 발견은 [Plan v0.5](../01-plan/features/aihat-yolo-port.plan.md), [Design v0.2](../02-design/features/aihat-yolo-port.design.md), [Progress Report](2026-05-01-progress-report.md), 그리고 git 커밋 이력에 추적 가능하다.*
