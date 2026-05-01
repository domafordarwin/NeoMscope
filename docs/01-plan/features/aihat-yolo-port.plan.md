---
template: plan
version: 1.2
feature: aihat-yolo-port
date: 2026-05-01
author: domafordarwin
project: NeoMscope
version: 1.0
---

# aihat-yolo-port Planning Document

> **Summary**: Mask R-CNN(TF1.x) 양파 세포 분열 검출기를 **YOLOv11-det(PyTorch/Ultralytics)** 기반으로 이식하고, **Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H, 40 TOPS INT4 / 20 TOPS INT8, 8GB LPDDR4X)** 위에서 실시간(≥30 FPS) 추론하도록 재구축한다. 학습은 **Kaggle T4/P100 16GB VRAM**, 라벨은 **기존 Mask R-CNN 부트스트랩** 방식으로 확보.
>
> **Project**: NeoMscope
> **Version**: 1.0 (전람회용)
> **Author**: domafordarwin
> **Date**: 2026-05-01
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

기존 PC + Matterport Mask R-CNN(2018) 파이프라인은 (1) TF1.x/Keras 의존으로 환경 재현이 어렵고, (2) 데스크톱 GPU가 있어야 실시간이 나오고, (3) 전람회 시연 시 PC를 끌고 다녀야 하는 문제가 있다.

본 작업은 동일 5-class 인스턴스 세그멘테이션 출력을 유지하면서:
- 모델을 **YOLO26-seg**로 교체 (단일 stage, NMS-free, 엣지 최적화)
- 추론 디바이스를 **Pi 5 + AI HAT+** 단독 박스로 이전
- 학습 파이프라인을 PyTorch/Ultralytics CLI 기반으로 단순화

### 1.2 Background

- **하드웨어 제약**: Hailo NPU(8/8L/10H 모두)는 RoIAlign/Two-stage 구조를 지원하지 않아 Mask R-CNN을 컴파일할 방법이 없다. YOLO 계열만 Hailo Model Zoo에 공식 등재됨.
- **소프트웨어 부채**: `mrcnn` 패키지는 5+년 미유지보수. TF2 호환성·Python 3.11 호환성 문제로 신규 환경 구축에 매번 막힘.
- **전람회 요건**: 휴대성·시연 안정성·"최신 기술 채택" 어필이 필요.
- **하드웨어 채택 근거 (AI HAT+ 2 / Hailo-10H, 2026-01-15 출시)**: 2세대 Hailo 신경 코어 + 8GB on-module LPDDR4X. 비전 성능은 Hailo-8(26 TOPS) 대비 동등하지만, on-module 메모리 덕에 더 큰 모델(YOLO `m`/`l`)을 여유롭게 구동 가능. 전력 3.5–4.5W로 저발열·저소음.
- **YOLOv11-det 채택 근거**: (1) 2024-09 출시 안정 버전, Hailo Model Zoo·Pi 5 사례 최다. (2) **설계 단계에서 발견 — 기존 모델은 polygon 마스크가 아닌 bbox 사각형으로 학습됨** ([dataset_config.py:88-94](../../../dataset_config.py#L88-L94)). 즉 현 모델 출력 = 본질적으로 bbox 검출. det로 가도 기능 동등하며, 학습·추론 모두 단순화. **백업: YOLOv8-det**.
- **데이터 확보 전략 — 수동 per-cell 라벨링** (v0.5 갱신): 원본 raws/JPEG_Export_Data 105장은 whole-mount root section이라 개별 분열 세포가 보이지 않음. 진짜 mitosis squash slide 데이터는 **`captured_raw_images/` 22장** (600×450, 이미지당 50-150 visible cells). Roboflow Smart Polygon (SAM 보조)로 SAM이 세포 경계 자동 추정 → 사용자가 5-class 분류만 결정. 예상 라벨링 시간 2-4시간 (이미지당 ~10분, 100+ cells/image). ❌ 부트스트랩 v1은 폐기 (잘못된 스케일).
- **학습 환경 — Kaggle**: 현 PC(Quadro P2000 4GB VRAM)는 학습 부적합. Kaggle T4/P100 16GB VRAM(주 30시간 무료) 사용. 현 PC는 코드 개발·ONNX export·WSL2 기반 HEF 컴파일·시연 준비용.
- **Hailo DFC 환경**: Hailo Dataflow Compiler는 Linux 전용 → Windows 11 기반 학습 PC에 **WSL2 + Ubuntu**로 별도 환경 구성 필요.

### 1.3 Related Documents

- 현 코드: [detect_onioncell.py](../../../detect_onioncell.py), [live_detect.py](../../../live_detect.py), [capture_and_detect.py](../../../capture_and_detect.py)
- 현 설정: [config_onion_20201022.py](../../../config_onion_20201022.py)
- 외부 레퍼런스:
  - Raspberry Pi AI HAT+ 2 제품 페이지: https://www.raspberrypi.com/products/ai-hat-plus-2/
  - Pi AI HAT+ 2 출시 공지: https://www.raspberrypi.com/news/introducing-the-raspberry-pi-ai-hat-plus-2-generative-ai-on-raspberry-pi-5/
  - Hailo-10H 공식 페이지: https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/
  - Hailo RPi5 Examples (주로 Hailo-8 기준 — 호환성 검증 필요): https://github.com/hailo-ai/hailo-rpi5-examples
  - Hailo Model Zoo (Hailo-10H는 **master 브랜치**): https://github.com/hailo-ai/hailo_model_zoo
  - Ultralytics YOLOv11 Docs: https://docs.ultralytics.com/models/yolo11/
  - Ultralytics YOLOv11 Segment 모델: https://docs.ultralytics.com/tasks/segment/

---

## 2. Scope

### 2.1 In Scope

- [ ] ~~부트스트랩 라벨링~~ ❌ **v0.5에서 폐기** (raws 데이터셋이 부적합).
- [ ] **수동 per-cell 라벨링**: [captured_raw_images/](../../../captured_raw_images/) 22장에 Roboflow Smart Polygon (SAM 보조)로 each cell → bbox + 5-class 분류
- [ ] Roboflow Augmentation 강화 (회전 360°, flip, crop, color jitter) — 22장 → 효과적 200-500장
- [ ] 데이터셋 무결성 검증(클래스 분포, 좌표 범위, bbox 유효성) 도구
- [ ] YOLOv11-det Kaggle 학습 노트북 (`training/notebooks/train-yolo11-det-kaggle.ipynb`)
- [x] WSL2 + Ubuntu 22.04 + Hailo DFC v5.3.0 환경 구성 ✅ 완료 (commit `690ba15`)
- [ ] PyTorch → ONNX → Hailo `.hef` 컴파일 파이프라인
- [ ] Pi 5 + AI HAT+ 실시간 추론 스크립트 (live_detect 대체)
- [ ] Pi 5 배치 추론 스크립트 (detect_onioncell 대체)
- [ ] Pi 5 캡처+추론 통합 스크립트 (capture_and_detect 대체)
- [ ] Python 3.11 + 가상환경 + pyproject.toml로 환경 표준화
- [ ] 새 디렉터리 구조 (`training/`, `inference/`, `tools/`, `models/`)
- [ ] mAP@0.5 / 클래스별 분류 정확도 / FPS 벤치마크 리포트
- [ ] 설치·시연 가이드(README.md 갱신)

### 2.2 Out of Scope

- 학습 데이터 추가 수집·새 현미경 촬영 (기존 105+22장 + 부트스트랩 라벨로 충분하다고 가정)
- **Polygon instance segmentation** (§1.3 발견에 따라 의미 없음 — 현 모델도 bbox만)
- 웹 UI / 원격 모니터링 / 클라우드 업로드
- 다중 카메라·다중 Pi 분산 추론
- 모델 양자화 외 추가 최적화(pruning, distillation)
- AI HAT+ 외 다른 가속기(Coral, Jetson) 포팅
- 학습용 GPU 머신 구축 (Colab/로컬 게이밍 PC 활용 가정)

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | **부트스트랩 라벨링**: 기존 Mask R-CNN으로 105장 자동 추론 → YOLO det `.txt` 생성 (CLI 스크립트) | High | Pending |
| FR-01b | 자동 라벨링 결과를 Roboflow/CVAT에 업로드해 시각 검수·보정 | High | Pending |
| FR-02 | 변환 후 무작위 샘플 시각화로 라벨 정합성 수동 검증 가능 | High | Pending |
| FR-03 | YOLOv11-det를 5-class(분열 단계)로 Kaggle에서 학습 가능 | High | Pending |
| FR-04 | 학습 결과 ONNX export 후 Hailo Dataflow Compiler로 `.hef` 생성 | High | Pending |
| FR-05 | Pi 5 + AI HAT+에서 USB/CSI 카메라 실시간 추론 (window 표시 + 콘솔 카운트) | High | Pending |
| FR-06 | Pi 5에서 `captured_raw_images/` 일괄 추론 및 결과 이미지/JSON 저장 | High | Pending |
| FR-07 | "캡처→즉시 추론→결과 저장" 통합 모드 (전람회 시연용) | Medium | Pending |
| FR-08 | 클래스별 검출 개수·신뢰도를 콘솔 출력 (현 동작 유지) | Medium | Pending |
| FR-09 | 5-class 색상 매핑 유지 ([colors.txt](../../../colors.txt) 호환) | Medium | Pending |
| FR-10 | 단일 명령으로 Pi 5에 의존성 설치 (`make install` 또는 `bootstrap.sh`) | Medium | Pending |
| FR-11 | 학습 실패·HEF 컴파일 실패 시 명확한 에러 메시지·복구 가이드 | Low | Pending |
| FR-12 | 학습 PC에 WSL2 + Ubuntu + Hailo DFC 4.x+ 환경 구성 (Linux 전용 도구) | High | Pending |
| FR-13 | Kaggle/Colab 노트북에서 학습→ONNX export까지 단일 셀로 실행 가능 | Medium | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| **Performance (실시간)** | AI HAT+ 2 (Hailo-10H, 20 TOPS INT8) + YOLOv11s-det @ 640×640: **≥40 FPS** (det는 seg보다 빠름), `m` 사이즈에서도 **≥30 FPS** | `live_detect.py` 내장 FPS 카운터 + 60초 평균 |
| **Performance (배치)** | Pi 5 + AI HAT+ 2: 1장당 <60ms (전처리·후처리 포함) | `time.perf_counter()` 측정 |
| **Power** | 추론 중 모듈 전력 < 5W, 패시브 쿨링으로 동작 | `vcgencmd measure_temp` 60초 평균 < 75°C |
| **Accuracy** | mAP@0.5 ≥ 기존 Mask R-CNN의 90% (절대치는 베이스라인 측정 후 확정) | Ultralytics `val` + 동일 holdout set |
| **Resource** | Pi 5 추론 중 RSS < 800MB, CPU < 50% (NPU 오프로드 검증) | `htop` / `psutil` 30초 샘플링 |
| **Reproducibility** | 신규 머신에서 README 따라 학습→배포 재현 가능 | 별도 머신·별도 사용자가 수행 검증 |
| **Compatibility** | Python 3.11, Ultralytics ≥ latest, HailoRT 4.20+, Pi OS Bookworm 64-bit | `pyproject.toml` 핀 + CI 체크 (옵션) |
| **Code quality** | Black + Ruff 통과, 핵심 모듈 100% type hint | `ruff check` / `black --check` |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] FR-01 ~ FR-11 모두 구현
- [ ] 모든 NFR 기준 충족 (벤치마크 결과 첨부)
- [ ] HEF 파일이 AI HAT+ 26 TOPS 환경에서 정상 동작 (TOPS 모델 확정 후 13 TOPS 대상도 검증)
- [ ] 시연 시나리오: "Pi 부팅 → USB 웹캠 연결 → 1개 명령 실행 → 실시간 검출 화면 표시"가 30초 내 완료
- [ ] README.md에 학습·배포·시연 절차 기록
- [ ] PDCA Gap Analysis Match Rate ≥ 90%

### 4.2 Quality Criteria

- [ ] 신규 코드 type hint 적용 + Ruff/Black 통과
- [ ] 라벨 변환 스크립트는 단위 테스트 포함 (좌표 정규화 round-trip)
- [ ] 추론 스크립트는 모델 파일 없이도 import 시 에러 없음 (lazy load)
- [ ] 가중치(.h5, .pt, .onnx, .hef)는 모두 gitignore 유지

---

## 5. Risks and Mitigation

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R-01 | ~~YOLO26-seg 신규성~~ → **해소**: 안정성 우선으로 YOLOv11-seg를 1차로 채택 (2026-05-01 결정). 백업 YOLOv8-seg로 충분. | — | — | ✅ 모델 선택으로 회피 |
| R-02 | 기존 Mask R-CNN 라벨 포맷이 다양해서 자동 변환이 모든 케이스 커버 못함 | High | Medium | 변환 후 무작위 50장 시각 검수(FR-02) 통과율 ≥98%만 학습 진행. 미달 시 라벨 패턴 추가 보정. |
| R-03 | Hailo Dataflow Compiler 라이선스/가입 절차로 일정 지연 | Medium | Medium | 작업 시작 전 Hailo Developer Zone 가입·DFC 다운로드 완료 (T0). |
| R-04 | ~~AI HAT+ TOPS 사양 미확정~~ → **해소**: AI HAT+ 2 (Hailo-10H) 확정 | — | — | ✅ 2026-05-01 확정 |
| R-05 | mAP가 Mask R-CNN 대비 크게 떨어짐 (도메인 갭, augmentation 부족) | High | Low | Ultralytics 기본 augmentation + onion-specific(회전·color jitter) 추가. 기준 미달 시 **8GB 메모리 여유로 `m`/`l` 사이즈로 즉시 업그레이드 가능**. |
| R-06 | Pi 5에서 카메라 → Hailo → 화면 출력 GStreamer 파이프라인 디버깅 난이도 | Medium | Medium | `hailo-rpi5-examples`는 Hailo-8 기준이라 Hailo-10H 호환성 검증 후 fork. 미동작 시 HailoRT Python API 직사용 경로 확보. |
| R-09 | Hailo-10H + Hailo Model Zoo master 브랜치는 비교적 새. **YOLOv11-seg 채택으로 위험 절반↓** (커뮤니티 사례 다수) | Medium | Medium | Phase 4에서 컴파일·런타임 검증 단계 명시적으로 분리. Hailo Community 포럼 사례 사전 수집. 백업으로 YOLOv8-seg. |
| R-10 | HailoRT / DFC 버전 미스매치로 컴파일 통과해도 런타임 실패 | Medium | Medium | Phase 0에서 학습 PC와 Pi 5의 HailoRT/DFC 버전을 단일 메이저로 고정. 패키지 버전 `pyproject.toml` + `setup.sh`에 명시. |
| R-11 | Kaggle 무료 한도(주 30시간) 초과 또는 세션 끊김 | Medium | Medium | 학습 1회는 1-2시간이라 여유 충분. 체크포인트 자동 저장 활성화. Colab을 백업 환경으로 둠. |
| R-12 | 기존 Mask R-CNN(TF1.x, mrcnn 패키지) 부트스트랩 추론이 모던 Python 환경에서 동작 안 함 | High | Medium | Phase 1에서 우선 Python 3.7-3.8 + TF 1.15 별도 venv로 동작 확인. 실패 시 (a) Roboflow 수작업 라벨링으로 폴백 (1-3일) (b) Colab의 TF1.x 노트북 사용. |
| R-13 | WSL2 + Hailo DFC 셋업이 Windows 환경에서 막힘 (네트워크·드라이버) | Medium | Medium | Phase 0에서 WSL2 셋업을 프로젝트 셋업과 분리해 사전 검증. 막힐 시 Hailo 공식 Docker 이미지 사용. |
| R-14 | 부트스트랩 라벨이 기존 Mask R-CNN의 오류를 그대로 상속 (가비지 인 가비지 아웃) | High | Medium | 자동 라벨 후 Roboflow 시각 검수 의무화 (FR-01b). 신뢰도 < 0.5 검출은 수작업 검토 대상으로 표시. |
| R-07 | 전람회 직전 모델 가중치 분실/HEF 손상 | High | Low | 모델 산출물(.pt, .onnx, .hef)을 로컬 + 클라우드 드라이브 + 외장 SSD 3중 백업. |
| R-08 | 한글 시각화 텍스트(클래스명) 폰트 누락으로 화면에 □□ | Low | Medium | 시연용 라벨은 영문/약어로 통일하거나 NanumGothic 폰트 동봉. |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| Starter | Simple structure | 정적 사이트 | ☐ |
| **Dynamic** | Feature-based modules, Python ML pipeline + edge runtime | **본 프로젝트** | ☑ |
| Enterprise | Layered, microservices | 대규모 시스템 | ☐ |

> ⚠️ bkit 표준 Dynamic은 Next.js/BaaS 프론트엔드를 가정하지만, 본 프로젝트는 **"Python ML + Edge"** 변형 Dynamic으로 운용한다.

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| ML Framework | TF1+Keras / TF2 / PyTorch | **PyTorch (Ultralytics)** | Ultralytics CLI 표준. Hailo Model Zoo 입력도 PyTorch/ONNX. |
| Detector | Mask R-CNN / YOLO seg / **YOLO det** | **YOLOv11-det** | §1.3 발견(bbox-as-mask) → seg 의미 없음. det로 충분 + 더 빠름. **백업: YOLOv8-det**. |
| Model Size | n / s / m / l | 1차 **YOLOv11s**, mAP 부족 시 **YOLOv11m**(8GB 메모리로 여유) | Hailo-10H 20 TOPS INT8 + 8GB → s 40+ FPS 안정, m 30+ FPS 가능 |
| 학습 환경 | 로컬 GPU / Colab / **Kaggle** | **Kaggle (T4/P100 16GB)** + Colab 백업 | 현 PC GPU(Quadro P2000 4GB) 부족. Kaggle 30h/wk 무료 충분. |
| 데이터 라벨링 | 수작업 / 부트스트랩 / 새 수집 | **부트스트랩 (기존 Mask R-CNN)** | 0.5–1일. R-14로 검수 강화. |
| 학습 PC OS 환경 | Windows 네이티브 / WSL2 / 듀얼부트 | **Windows 11 + WSL2 + Ubuntu** | Hailo DFC는 Linux 전용. WSL2가 가장 마찰 적음. |
| Hailo 도구체인 | Model Zoo v2.x + DFC 3.x (Hailo-8) / **master + DFC 4.x+ (Hailo-10H)** | **master + DFC 4.x+** | Hailo-10H는 Model Zoo master 브랜치 전용 |
| 추론 런타임 | HailoRT (Python 직접) / GStreamer + hailo_apps_infra | **GStreamer + hailo_apps_infra (1차)**, 미동작 시 HailoRT Python 직접 (백업) | 공식 예제 시작점 확보 + 폴백 경로 |
| 카메라 | USB UVC / Pi CSI camera | USB UVC 우선 (현 코드 호환), CSI 옵션 | 현 `imutils.video.VideoStream` 호환성 |
| 패키지 관리 | pip + requirements.txt / **pyproject.toml + uv** | pyproject.toml + uv | 재현성·속도. requirements.txt는 export로 호환 유지 |
| 라벨 변환 | 일회성 스크립트 / pip 도구(`globox`) | **자체 스크립트 + globox 검증** | 도메인-특화 보정 필요. globox로 round-trip 검증 |
| 코드 스타일 | Black + Ruff | Black + Ruff | Python 사실상 표준 |
| 테스트 | unittest / **pytest** | pytest | 변환 round-trip + 학습 산출물 smoke test |

### 6.3 Folder Structure

```
NeoMscope/
├── README.md
├── CLAUDE.md
├── pyproject.toml                          # 신규
├── .gitignore
├── docs/                                    # PDCA 문서
│   ├── 01-plan/features/aihat-yolo-port.plan.md
│   ├── 02-design/features/...
│   └── ...
├── training/                                # 학습 (PC/Colab)
│   ├── data.yaml                            # YOLO 데이터셋 설정
│   ├── train.py                             # Ultralytics 래퍼
│   ├── eval.py                              # mAP 측정
│   └── notebooks/
│       └── train-yolo26-seg.ipynb
├── tools/                                   # 변환·유틸
│   ├── convert_mrcnn_to_yolo.py             # 라벨 변환
│   ├── validate_dataset.py                  # 데이터셋 검증
│   ├── visualize_labels.py                  # 시각 검수
│   ├── export_onnx.py                       # PyTorch → ONNX
│   └── compile_hef.sh                       # ONNX → HEF (Hailo DFC 호출)
├── inference/                               # Pi 5 추론
│   ├── live_detect.py                       # FR-05
│   ├── batch_detect.py                      # FR-06
│   ├── capture_and_detect.py                # FR-07
│   ├── pipeline.py                          # GStreamer 파이프라인 정의
│   └── postprocess.py                       # 후처리·시각화
├── models/
│   ├── pt/        (gitignore)
│   ├── onnx/      (gitignore)
│   └── hef/       (gitignore)
├── captured_raw_images/                     # 추론 샘플(유지)
├── detection_results_*/                     # 결과 출력(gitkeep만)
└── legacy/                                  # 기존 Mask R-CNN 파일 보관(전람회 비교용)
    ├── detect_onioncell.py
    ├── live_detect.py
    ├── capture_and_detect.py
    ├── config_onion_20201022.py
    ├── dataset_config.py
    └── README.md  (legacy 사용 안내)
```

### 6.4 Data Flow

```
[Mask R-CNN labels (VOC/JSON)]
        |
        v
[tools/convert_mrcnn_to_yolo.py]   ← FR-01
        |
        v
[YOLO det .txt + data.yaml]
        |
        v
[training/train.py] ── Ultralytics CLI ──→ [.pt weights]   ← FR-03
        |
        v
[tools/export_onnx.py]                 ──→ [.onnx]
        |
        v
[tools/compile_hef.sh] ── Hailo DFC ──→ [.hef]            ← FR-04
        |
        v
[Pi 5 + AI HAT+]
   ├── inference/live_detect.py        (FR-05)
   ├── inference/batch_detect.py       (FR-06)
   └── inference/capture_and_detect.py (FR-07)
```

---

## 7. Convention Prerequisites

### 7.1 Existing Project Conventions

- [x] CLAUDE.md (코드베이스 컨텍스트만 — 코딩 컨벤션 없음)
- [ ] `docs/01-plan/conventions.md` 생성 필요
- [ ] `pyproject.toml` 생성 필요 (Black, Ruff, pytest 설정 포함)
- [ ] `.editorconfig` 생성 권장

### 7.2 Conventions to Define/Verify

| Category | Current | To Define | Priority |
|----------|---------|-----------|:--------:|
| Naming | snake_case (혼재) | snake_case 통일, 모듈은 단어 1~2개 | High |
| Folder | flat | training/inference/tools 분리 (6.3) | High |
| Import order | 자유 | isort 표준: stdlib → 3rd → local | Medium |
| Docstrings | 거의 없음 | Google-style, 공개 API 필수 | Medium |
| Type hints | 거의 없음 | 신규 코드 100%, 레거시는 점진 | Medium |
| 한글 주석 | 자유 | 도메인 설명에 한해 허용, 코드 식별자는 영문 | Low |
| 에러 처리 | 미흡 | 외부 경계(I/O, 카메라, 모델 로드) only, 내부는 fail-fast | Medium |

### 7.3 Environment Variables Needed

| Variable | Purpose | Scope | To Be Created |
|----------|---------|-------|:-------------:|
| `NEOMSCOPE_MODEL_PATH` | HEF 파일 위치 | Pi 추론 | ☐ |
| `NEOMSCOPE_CAMERA` | usb / csi / 파일 경로 | Pi 추론 | ☐ |
| `NEOMSCOPE_OUTPUT_DIR` | 결과 저장 경로 | Pi 추론 | ☐ |
| `NEOMSCOPE_CONF_THRESHOLD` | 신뢰도 임계값 (기본 0.25) | Pi 추론 | ☐ |
| `HAILO_DFC_LICENSE` | Hailo DFC 라이선스 (학습 PC) | 학습 | ☐ |

### 7.4 Pipeline Integration

bkit 9-phase Development Pipeline 매핑:

| Phase | Status | Document Location |
|-------|:------:|-------------------|
| Phase 1 (Schema) | ☐ | `docs/01-plan/schema.md` (5-class 분열 단계 정의) |
| Phase 2 (Convention) | ☐ | `docs/01-plan/conventions.md` |
| Phase 3 (Mockup) | ⏭ skip | UI 없음 (시각화 화면은 Phase 6에서) |
| Phase 4 (API) | ⏭ skip | 외부 API 없음 |
| Phase 5 (Design System) | ⏭ skip | UI 없음 |
| Phase 6 (UI Integration) | ☐ | live_detect 시각화 부분만 해당 |
| Phase 7 (SEO/Security) | ⏭ skip | 공개 서비스 아님 |
| Phase 8 (Review) | ☐ | Gap analysis 시 |
| Phase 9 (Deployment) | ☐ | Pi 이미지 SD카드 굽기 가이드 |

---

## 8. Development Phases (실행 단계)

> PDCA Plan 단계 내부의 **개발 phase 분해**. 각 게이트는 다음 단계로 넘어가기 위한 통과 조건.

### Phase 0 — 사전 준비 (Day 0-1)
- [x] AI HAT+ 2 / Hailo-10H 확정 (2026-05-01)
- [x] Hailo Developer Zone 가입 (사용자 완료)
- [ ] **WSL2 + Ubuntu 22.04** 학습 PC에 설치 (R-13)
- [ ] WSL2에 Hailo DFC 4.x+ 다운로드·설치
- [ ] Kaggle 계정 + Notebook 환경 점검 (`!nvidia-smi` 확인)
- [ ] Hailo Model Zoo **master 브랜치** clone + Hailo-10H 호환 모델 목록 확인
- [ ] Pi OS Bookworm 64-bit + AI HAT+ 2용 HailoRT 설치
- **Gate 0**: Pi 5 + AI HAT+ 2에서 Hailo Model Zoo의 stock **YOLOv11 detection** 데모가 정상 추론

### Phase 1 — 베이스라인 + 부트스트랩 라벨링 (Day 1-3)
- [ ] **레거시 환경 구성**: Python 3.7-3.8 venv + TF 1.15 + mrcnn 패키지 (R-12)
- [ ] 기존 Mask R-CNN 모델로 holdout 측정용 임의 30장 추론 → mAP·클래스별 precision/recall 기록 → `docs/03-analysis/baseline.md`
- [ ] **부트스트랩**: 105장 + 22장(captured) 모두 추론 → bbox + 클래스 + 신뢰도 추출
- [ ] `tools/bootstrap_labeling.py`로 결과를 YOLO det `.txt` 형식 + Roboflow 업로드용 COCO JSON으로 변환
- [ ] Roboflow에 업로드 → 시각 검수 + 신뢰도 < 0.5 케이스 수작업 보정 (FR-01b)
- [ ] Roboflow에서 보정 완료 데이터 export (YOLO det 형식)
- [ ] `tools/validate_dataset.py`: 좌표 범위·클래스 분포 검사
- [ ] `training/data.yaml` 작성 (train/val/test 70/20/10 split)
- **Gate 1**: 시각 검수 통과율 ≥ 98%, 클래스 분포 불균형 < 10:1

> ⚠️ R-12 폴백: 레거시 환경 구성이 막히면 **곧장 Roboflow 수작업 라벨링으로 전환** (1-3일 추가). Phase 1 일정에 1일 버퍼 포함.

### Phase 2 — Kaggle YOLOv11-det 학습 (Day 3-5)
- [ ] Kaggle 노트북 작성 (`training/notebooks/train-yolo11-det-kaggle.ipynb`)
- [ ] 데이터셋을 Kaggle Dataset으로 업로드 (또는 GitHub release를 wget)
- [ ] **YOLOv11s** 학습 (epoch 100, early stop, batch=16, imgsz=640)
- [ ] mAP가 베이스라인의 90% 미달 시 → YOLOv11m 또는 augmentation 강화로 재학습
- [ ] best.pt + last.pt 다운로드 (Kaggle output → 로컬)
- [ ] (선택) HEF 컴파일 차질 대비 백업 **YOLOv8s** 동시 학습
- **Gate 2**: YOLOv11s/m이 mAP@0.5 ≥ 베이스라인의 90%

### Phase 3 — 모델 변환 (Day 5-6)
- [ ] `.pt` → `.onnx` export (`tools/export_onnx.py`, opset=17, dynamic=False, nms=False)
- [ ] ONNX 검증 (`onnxruntime`로 PyTorch 출력 대비 max abs diff < 1e-3)
- [ ] **WSL2 Ubuntu**에서 Hailo DFC로 `.hef` 컴파일 (`tools/compile_hef.sh`, `--hw-arch hailo10h`)
- [ ] 컴파일 실패 시 백업 YOLOv8 모델로 분기
- **Gate 3**: Pi 5에서 `.hef` 로드 성공 + 단일 추론 성공

### Phase 4 — Pi 5 추론 포팅 (Day 6-8)
- [ ] `inference/pipeline.py` (GStreamer + hailo_apps_infra 래퍼)
- [ ] `inference/postprocess.py` (bbox NMS, 색상 매핑 — det이라 마스크 디코딩 없음, 단순)
- [ ] `inference/live_detect.py` (FR-05)
- [ ] `inference/batch_detect.py` (FR-06)
- [ ] `inference/capture_and_detect.py` (FR-07)
- **Gate 4**: 3개 시나리오 모두 정상 동작 + FPS NFR 충족 (≥40 FPS at s)

### Phase 5 — 통합 검증·시연 리허설 (Day 8-10)
- [ ] 신규 SD카드에서 README + setup.sh만으로 재현
- [ ] 30초 시연 시나리오 리허설 5회
- [ ] 벤치마크 보고서 (`docs/03-analysis/aihat-yolo-port.benchmark.md`)
- **Gate 5**: PDCA Gap Analysis Match Rate ≥ 90%

> 일정은 풀타임 기준 ~10일. 학교/방과 후 기준이면 ×2~3 가산.
> 단계 번호가 v0.3 대비 한 단계 줄었음 (det 채택으로 Phase 1 베이스라인과 Phase 2 변환을 Phase 1로 통합).

---

## 9. Next Steps

1. [x] ~~AI HAT+ TOPS 확정~~ → AI HAT+ 2 / Hailo-10H
2. [x] ~~Hailo Developer Zone 가입~~ (사용자 완료)
3. [x] ~~설계 문서 작성~~ → [aihat-yolo-port.design.md](../../02-design/features/aihat-yolo-port.design.md)
4. [ ] **사용자**: WSL2 + Ubuntu 22.04 설치 (Phase 0)
5. [ ] **사용자**: Kaggle 계정 점검 (있으면 그대로, 없으면 가입)
6. [ ] **다음 작업**: Phase 0/1 시작 — `/pdca do aihat-yolo-port` 또는 직접 Group 0(환경·이전) 코드 수정 시작

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-01 | Initial draft (Mask R-CNN → YOLO26-seg + Pi 5 AI HAT+ 이식 계획) | domafordarwin / Claude |
| 0.2 | 2026-05-01 | 하드웨어 확정: AI HAT+ 2 / Hailo-10H. Model Zoo master 브랜치 + DFC 4.x로 도구체인 갱신. R-04 해소, R-09/R-10 추가. 모델 사이즈 `s` 기본/`m` 업그레이드 가능. | domafordarwin / Claude |
| 0.3 | 2026-05-01 | **모델 선택 변경: YOLO26-seg → YOLOv11-seg** (안정성 우선). 백업 YOLOv11→YOLOv8. R-01 해소, R-09 위험도 ↓. | domafordarwin / Claude |
| 0.4 | 2026-05-01 | **3대 변경**: ① 설계 단계 발견(bbox-as-mask)로 **seg→det** 확정. ② 학습 PC GPU(P2000 4GB) 부족 확인 → **Kaggle 학습**. ③ 어노테이션 부재 확인 → **부트스트랩 라벨링**. R-11~14 추가, FR-12/13 추가, Phase 1↔2 통합으로 단계 6→5. | domafordarwin / Claude |
| 0.5 | 2026-05-01 | **데이터셋 근본 재정의**: Roboflow 검수 중 raws/JPEG_Export_Data(105장, 8000×31000)는 whole-mount root section으로 mitosis 관찰엔 부적합 발견. 실제 squash slide는 **captured_raw_images의 22장** (600×450, 이미지당 50-150 visible cells). 부트스트랩 라벨(399 detections)도 잘못된 스케일이라 폐기. **새 전략**: 22장 + 강한 augmentation + Roboflow Smart Polygon (SAM 보조) 수동 per-cell 라벨링. | domafordarwin / Claude |
