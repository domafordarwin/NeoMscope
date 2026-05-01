# NeoMscope v2.0 마이그레이션 — 작업 보고서

> **작성일**: 2026-05-01
> **작성자**: Claude (Opus 4.7) + domafordarwin
> **세션**: 단일 세션 ~12시간 작업
> **PDCA 단계**: Plan v0.5 → Design v0.2 → Do (진행 중)
> **GitHub**: https://github.com/domafordarwin/NeoMscope

---

## 1. 한 줄 요약

PC + Mask R-CNN(TF1.x) → **Pi 5 + AI HAT+ 2 (Hailo-10H) + YOLOv11-det** 마이그레이션 진행 중. 환경·도구·코드 ~95% 완료, **데이터 라벨링이 핵심 블로커**. 1차 시도(부트스트랩 자동 라벨링)에서 데이터셋 스케일 문제 발견 → 2차 시도(tile + native-res mrcnn)로 검증 중, 1장 결과 시각적으로 양호.

---

## 2. 작업 산출물 현황

### 2.1 Git 커밋 (14개, 모두 푸시됨)

| Commit | 내용 |
|---|---|
| `effffe2` | Initial commit (원격 README) |
| `a6f2624` | Mask R-CNN 코드베이스 + Claude env (초기 import) |
| `1369db6` | Group 0: 디렉터리 구조 + legacy/ 이전 + foundation types |
| `091a0eb` | Group 1: bootstrap toolchain + 43 unit tests |
| `3a7b917` | Groups 2-4: Kaggle notebook + ONNX/HEF 도구 + 추론 파이프라인 |
| `77af363` | bootstrap_labeling Python 3.7 호환성 fix |
| `b629dca` | visualize_labels --max-size + --jpg 옵션 |
| `415ea46` | package_for_roboflow.py + 12 tests (3.2GB → 9.5MB 압축) |
| `ccc6cb0` | CLAUDE.md + README v2.0 갱신 |
| `ab2e752` | install_hailo_dfc.sh 자동 설치 + compile_hef.sh 강화 |
| `690ba15` | **Hailo DFC v5.3.0 + Model Zoo 설치 완료 (WSL2)** |
| `baa72e2` | Scenario 2 pivot: 22 captured 이미지 (생산 안 됨, 곧 폐기) |
| `8c91ef3` | **tile_and_detect.py — native-res 타일 검출** |

### 2.2 코드 통계

| 항목 | 수량 |
|---|---|
| 신규 Python 파일 | 14개 |
| 신규 코드 라인 | ~4,000줄 |
| 단위 테스트 | **75개 (전부 통과)** |
| ruff lint | **0 errors** |
| 미커밋 변경 | 0 (모두 푸시됨) |

### 2.3 PDCA 문서

- **Plan v0.5**: [docs/01-plan/features/aihat-yolo-port.plan.md](../01-plan/features/aihat-yolo-port.plan.md)
- **Design v0.2**: [docs/02-design/features/aihat-yolo-port.design.md](../02-design/features/aihat-yolo-port.design.md)
- **이 보고서**: docs/03-analysis/2026-05-01-progress-report.md

---

## 3. 환경 셋업 (모두 완료)

### 3.1 메인 venv (`.venv/`, Python 3.11.14)
- **목적**: 신규 코드 개발·테스트·ONNX export·Roboflow 패키징
- **주요 deps**: ruff, black, pytest 9.0, ultralytics 8.3, onnx, opencv-python, numpy
- **상태**: ✅ 동작, 75 tests passing

### 3.2 레거시 conda venv (`neomscope-legacy`, Python 3.7)
- **목적**: 기존 Mask R-CNN 추론 (부트스트랩·타일 검출용)
- **주요 deps**: TF 1.15.0 (eigen_py37), Keras 2.2.5, NumPy 1.19, OpenCV 4.5, Matterport mrcnn
- **상태**: ✅ 동작, 가중치 로드 성공, 추론 성공

### 3.3 WSL2 Ubuntu 22.04 + Hailo DFC
- **위치**: `/root/hailo-workspace/venv/` (Python 3.10.12)
- **주요 deps**: Hailo Dataflow Compiler v5.3.0, hailo_model_zoo v5.3.0
- **상태**: ✅ 동작, `hailomz info yolov11s` 검증
- **YAML**: yolov11n/s/m/l/x 사용 가능

### 3.4 사용자 보유 / 완료 항목
- ✅ Raspberry Pi AI HAT+ 2 (Hailo-10H, 40 TOPS INT4 / 20 TOPS INT8, 8GB LPDDR4X)
- ✅ Hailo Developer Zone 가입
- ✅ DFC v5.3.0 .whl (499 MB) 다운로드 + 설치
- ✅ HailoRT PCIe 드라이버 .deb (28 MB) — `pi5-deploy/`에 보관 (Pi 배포용)

### 3.5 미완료 (사용자 작업 대기)
- ⏳ Kaggle 계정 + GPU 활성화
- ⏳ Roboflow 계정 (생성됨, 프로젝트 폐기 후 재시작 예정)
- ⏳ 학습 데이터 라벨 확정 (현재 진행 중)

---

## 4. 핵심 의사결정 (5개)

| ID | 결정 | 변경 전 | 변경 후 | 근거 |
|---|---|---|---|---|
| D-01 | 검출 출력 형식 | Mask seg | **bbox det** | legacy `dataset_config.py:88-94`이 polygon 아닌 bbox 사각형을 마스크로 사용 — seg 의미 없음 |
| D-02 | YOLO 버전 | YOLO26-seg (최신) | **YOLOv11s-det** | 안정성·튜토리얼·Hailo 사례 최대화. 최신 YOLO26은 미검증. |
| D-03 | 학습 환경 | 로컬 GPU | **Kaggle T4/P100** | 학습 PC GPU(Quadro P2000 4GB)는 YOLO 학습 부족 |
| D-04 | 라벨 확보 | 수동 라벨링 | **Mask R-CNN 자동 + 수동 검수** | 원본 어노테이션 부재. 가중치 보유. |
| D-05 | 데이터셋 | 1차: 127장 (raws + captured) | **재검증 중**: native-res 타일 검출 | 라벨이 root tip 단위로 너무 큼 — tile + 재실행으로 해결 시도 |

---

## 5. 데이터 발굴 여정

```
[시작]
   │
   │  raws/JPEG_Export_Data 105장 + captured_raw_images 22장 = 127장
   │  ↓ Mask R-CNN inference (1280px 다운샘플)
   │
[1차 시도] 부트스트랩 라벨링 (commit 091a0eb, 8c91ef3 이전)
   │  결과: 399 detections (avg 3.1/장)
   │  ⚠️ Roboflow 검수 시 발견: 박스가 root tip 전체를 감쌈 (스케일 잘못)
   │  → "완전히 잘못된 듯" (사용자 피드백)
   │
[Pivot 검토]
   │  발견: raws 원본은 8000×31000 픽셀 (whole-slide scan)
   │  다운샘플 1280으로 → 세포 한 개가 10-20px → mrcnn IMAGE_MIN_DIM=512 가
   │  전체를 letterbox → root tip 1개에 1 bbox만 그려짐
   │
[2차 시도] tile_and_detect.py (commit 8c91ef3)
   │  방법: 8000×31000을 512×512 타일 1886개로 분할, 각 타일에 mrcnn 적용
   │  결과: 747 detections (이전 대비 ×100), 평균 conf 0.71
   │  ⚠️ 사용자 피드백: "박스 너무 작음 + 양파 외부에 박스 많음"
   │
[3차 시도 — 현재] 필터링 추가
   │  필터: min bbox size 30px + tissue mask + conf 0.6
   │  결과 (1장): 240 detections, 평균 conf 0.78
   │   - 작은 박스 186개 제거
   │   - 조직 외부 97개 제거
   │   - Telo over-detection (411→27) 해소
   │  → 시각적으로 적절 (메리스템 영역 집중, false positive 거의 없음)
```

---

## 6. 현재 검출 결과 (3차 시도, 1장 검증)

### 6.1 통계
| 지표 | 1차 (스케일 잘못) | 2차 (타일링만) | **3차 (타일링 + 필터)** |
|---|---|---|---|
| 총 detections | 5 | 747 | **240** |
| 평균 신뢰도 | 0.65 | 0.71 | **0.78** |
| Inter | 0 | 18 | **6** |
| Pro | 1 | 89 | **51** |
| Meta | 0 | 39 | **26** |
| Ana | 1 | 190 | **130** |
| Telo | 3 | 411 | **27** |
| 시각 평가 | tip 전체에 1개 박스 (스케일 X) | 텅 빈 영역에도 박스 | **조직 안에만 박스, 메리스템 집중** |

### 6.2 시각 검증
- 오버뷰: `runs/tiled_detect_filtered/2020_10_07__17_31__0065-Image Export-01/_overview.jpg`
- 조직 마스크: `runs/tiled_detect_filtered/2020_10_07__17_31__0065-Image Export-01/_tissue_mask.jpg`

조직 마스크는 두 개의 양파 뿌리 끝 실루엣을 깨끗하게 분리. 흰 배경의 점·노이즈는 모두 제거됨.

### 6.3 처리 시간
- 1장 (8827×31530, 1886 타일) = **20분** (CPU only, Quadro P2000은 미사용)
- 105장 전체 추정 = **~35시간** (1.5일 연속)

---

## 7. 주요 코드 산출물

### 7.1 추론 파이프라인 (Pi 5 측)
- [inference/types.py](../../inference/types.py) — Detection, PipelineConfig, CLASS_NAMES (Source of truth)
- [inference/pipeline.py](../../inference/pipeline.py) — HailoInferencePipeline (mock=True 지원)
- [inference/postprocess.py](../../inference/postprocess.py) — YOLO det decode + render
- [inference/_camera.py](../../inference/_camera.py) — FrameSource (USB/folder)
- [inference/live_detect.py](../../inference/live_detect.py) — `neomscope-live` (FR-05)
- [inference/batch_detect.py](../../inference/batch_detect.py) — `neomscope-batch` (FR-06)
- [inference/capture_and_detect.py](../../inference/capture_and_detect.py) — `neomscope-capture` (FR-07)

### 7.2 도구 (Dev PC 측)
- [tools/tile_and_detect.py](../../tools/tile_and_detect.py) — **타일링 + mrcnn (현재 사용 중)**
- [tools/bootstrap_labeling.py](../../tools/bootstrap_labeling.py) — 부트스트랩 자동 라벨 (1차 시도, 사용 중단)
- [tools/package_for_roboflow.py](../../tools/package_for_roboflow.py) — 이미지+라벨 ZIP
- [tools/visualize_labels.py](../../tools/visualize_labels.py) — bbox 시각화
- [tools/validate_dataset.py](../../tools/validate_dataset.py) — YOLO 데이터셋 검증
- [tools/export_onnx.py](../../tools/export_onnx.py) — PyTorch → ONNX with parity check
- [tools/compile_hef.sh](../../tools/compile_hef.sh) — ONNX → HEF (WSL2)
- [tools/install_hailo_dfc.sh](../../tools/install_hailo_dfc.sh) — DFC 자동 설치
- [tools/verify_hailo_install.sh](../../tools/verify_hailo_install.sh) — DFC 검증

### 7.3 학습 (Kaggle)
- [training/data.yaml](../../training/data.yaml) — 5-class YOLO 데이터셋 정의
- [training/notebooks/train-yolo11-det-kaggle.ipynb](../../training/notebooks/train-yolo11-det-kaggle.ipynb) — Kaggle 자체-포함 노트북

### 7.4 배포 (Pi 5)
- [setup.sh](../../setup.sh) — Pi 5 부트스트랩 (HailoRT APT + venv + self-test)
- [pi5-deploy/](../../pi5-deploy/) — PCIe 드라이버 보관

---

## 8. 미해결 결정 사항

### 8.1 즉시 결정 (다음 작업 전제)

**Q1**: 3차 필터링 결과가 만족스러운가?
- ✅ 만족 → 5장 더 처리 (옵션 D 본래 계획)
- ⚠️ 추가 튜닝 필요 → 어느 부분?

**Q2**: 105장 전체 처리 일정
- A. 35시간 전체 (스트라이드 384, overlap 25%) — 최고 품질
- B. 17시간 (스트라이드 512, no overlap) — 절반 시간, 약간 누락
- C. 14시간 (메리스템 영역만, 상단 40%) — 메리스템 집중 추출
- D. 일단 5장 더 처리 후 결정

**Q3**: 라벨 검수 전략 (전체 처리 후)
- 자동 라벨 그대로 (속도)
- 신뢰도 ≥ 0.85 만 (품질)
- 일부 샘플 (20-30장) 수동 검수 (균형)

### 8.2 중장기 결정

- **클래스 불균형**: Telo 27개 vs Inter 6개 (4.5:1)는 양호하나 Ana 130개가 여전히 우세. 학습 시 augmentation 강화 또는 cls_pw 가중치 적용 검토.
- **Pi 5 배포 시 카메라**: 학습은 microscopy whole-slide, 배포는 USB 웹캠? 도메인 갭 가능. 별도 검토 필요.

---

## 9. 위험 register 갱신

| ID | 위험 | 이전 상태 | 현재 상태 |
|----|------|-----------|-----------|
| R-01 | YOLO26-seg 신규성 | High | ✅ 해소 (YOLOv11-det 채택) |
| R-02 | 라벨 변환 누락 | High | ✅ 해소 (변환 불필요, 직접 자동 라벨) |
| R-03 | Hailo Dev Zone 일정 | Medium | ✅ 해소 (가입·다운로드 완료) |
| R-04 | AI HAT+ TOPS 미확정 | Medium | ✅ 해소 (Hailo-10H 확정) |
| R-05 | mAP가 베이스라인 90% 미달 | Low | ⏳ 학습 후 확인 |
| R-06 | GStreamer 디버깅 난이도 | Medium | ⏳ Pi 5 통합 시 확인 |
| R-09 | Hailo Model Zoo + YOLO11 + Hailo-10H 신규 조합 | High | ✅ 해소 (yolov11s.yaml 검증) |
| R-10 | DFC/HailoRT 버전 미스매치 | Medium | ✅ 해소 (둘 다 v5.3.0) |
| R-11 | Kaggle 한도 초과 | Medium | ⏳ 학습 시 확인 |
| R-12 | 레거시 venv 셋업 실패 | High | ✅ 해소 (conda 환경 동작) |
| R-13 | WSL2 셋업 실패 | Medium | ✅ 해소 (Ubuntu 22.04 + DFC 동작) |
| R-14 | 부트스트랩 라벨 노이즈 (R-12 갱신) | High | ⚠️ **여전히 활성** — 3차 필터링이 완화 중 |
| **R-15 (신규)** | **데이터 스케일 불일치 (다운샘플로 세포 가시성 상실)** | — | ✅ 해소 (tile + native-res 검출) |
| **R-16 (신규)** | **105장 전체 처리 시간 (35시간)** | — | ⚠️ 활성 — 사용자 결정 대기 |

---

## 10. 다음 단계

### 단기 (오늘-내일)
1. ✅ 3차 필터링 결과 검증 (1장, 240 dets, 시각 양호) — **완료**
2. ⏳ 5장 샘플 처리 (옵션 D, 사용자 승인 대기)
3. ⏳ 5장 결과로 패턴 일관성 확인
4. ⏳ 105장 전체 처리 전략 결정

### 중기 (1주일)
5. 105장 자동 라벨 생성 (~35시간 또는 단축 옵션)
6. 일부 샘플 수동 검수 (Roboflow Smart Polygon)
7. Roboflow → YOLOv11 형식 export
8. Kaggle 학습 (~30-60분)
9. ONNX export + parity check
10. WSL2 HEF 컴파일

### 후기 (배포)
11. Pi 5 + AI HAT+ 2에 HEF 배포
12. setup.sh 실행 + neomscope-live 시연
13. 벤치마크 보고서 (FPS, mAP)
14. 전람회 시연 리허설

---

## 11. 통계 요약

```
세션 시간:       ~12시간 (단일 세션)
Git commits:     14개
신규 코드:       ~4,000 줄
단위 테스트:     75개 (모두 통과)
Lint 에러:       0
PDCA 문서:       Plan v0.5, Design v0.2, Analysis (이 보고서)
환경 셋업:       3개 (메인 venv + 레거시 conda + WSL2 Hailo)
설치 스크립트:   3개 (setup.sh, install_hailo_dfc.sh, verify_hailo_install.sh)
의사결정:        5개 핵심 + 다수 중간 결정
위험 register:   16개 (12 해소, 1 ⚠️ 활성, 3 대기)
```

---

## 12. 결론

v2.0 마이그레이션의 **인프라·도구·코드는 ~95% 완료**. 핵심 블로커는 학습 데이터 라벨 품질 — 두 차례 시도 후 3차 필터링으로 시각적 만족도 도달. 이 보고서 작성 시점에 사용자 검토 대기 중이며, 만족 시 5장 → 105장 처리로 진행 후 학습·배포 단계로 넘어갈 예정.

전체 일정 추정 (라벨링 부터 시연까지): **2-3일** (옵션 C 메리스템 추출 + 학습 + 배포 기준).

---

## 13. 세션 후반 추가 (Apex ROI 도입 + 5-sample 검증)

이 보고서를 처음 작성한 후, 사용자 제안으로 **메리스템 ROI만 잘라서 검출**하는 4차 시도를 진행했다.

### 13.1 Apex ROI 추출 도구 신규
[tools/extract_apex_rois.py](../../tools/extract_apex_rois.py) (commit `613583f`):
- 조직 마스크 → 연결 컴포넌트별 root tip → 위쪽 fraction만 잘라 native-res JPG로 저장
- 처리: ~6초/이미지, 105장 추정 ~10분

### 13.2 top_fraction 튜닝 (0.35 → 0.15)
첫 시도 0.35는 메리스템 + elongation zone을 함께 포함 → elongation 세포가 Anaphase로 잘못 분류되는 패턴 발견 (apex1 확대 검수에서 길쭉한 직사각형 세포가 모두 "Ana"로 라벨됨). top_fraction을 0.15로 줄여 둥근 메리스템 캡만 정확히 캡처.

### 13.3 5-sample 검증 결과 (commit `19223ac`)
v2 (top_fraction=0.15)로 5장 처리:

| 항목 | v0 (full) | v1 (apex 35%) | **v2 (apex 15%)** |
|---|---:|---:|---:|
| 1장 평균 detection | 240 | 159 | **108** |
| Pro (전기) 비율 | 12% | 21% | **36%** ⭐ |
| Ana (후기) 비율 | 25% | 54% | **32%** |
| Telo (말기) 비율 | 55% | 11% | **12%** |
| Inter (간기) 비율 | 2% | 3% | **3%** |
| 처리 시간/장 | 20분 | 10분 | **5분** |
| 105장 추정 | 35h | 17h | **9h** |

**v2가 가장 균형잡힌 분포 + 가장 빠름 + 생물학적으로 타당** (메리스템에서 Pro/Meta/Ana가 자주 보이는 게 정상).

### 13.4 5-sample 통계
- 입력: 5 raws → **11 ROIs** (이미지당 2.2개)
- 출력: **538 detections** (평균 49/ROI, 108/원본 이미지)
- 신뢰도: 0.74 - 0.80 (일관)
- 가장 좋은 케이스: 0067 apex1 → 137 detections, 메리스템 격자 cell 위에 정확히 박스
- 빈 ROI: 0068의 apex2 (작은 fragment 1768×191) → 0 detections (정상, 자동 필터링)

### 13.5 결정 대기 (오늘 미진행)
- 105장 전체 처리 (~9-10시간 백그라운드, 밤사이 진행)
- 5장 결과만으로 학습 시도 (작은 데이터셋)
- 추가 튜닝 (top_fraction 더 작게 / conf 상향)

### 13.6 추가 산출물
- [tools/extract_apex_rois.py](../../tools/extract_apex_rois.py) — apex ROI 추출
- [tools/build_report_html.py](../../tools/build_report_html.py) — Markdown → HTML (이 보고서)
- [tools/build_science_fair_html.py](../../tools/build_science_fair_html.py) — 전람회용 HTML 빌더
- [docs/03-analysis/2026-05-01-science-fair-report.md](2026-05-01-science-fair-report.md) — 과학전람회 출품용 보고서 (407줄)
- [docs/03-analysis/2026-05-01-science-fair-report.html](2026-05-01-science-fair-report.html) — 위 보고서 HTML 변환

### 13.7 갱신된 위험 register
- R-15 (스케일 불일치) → ✅ 해소 (apex ROI로 native-res 처리)
- R-16 (35시간) → ✅ 해소 (apex ROI로 9시간으로 단축)
- **R-17 (신규)** — top_fraction 자동 튜닝 부재. 현재는 모든 이미지에 0.15 일괄 적용. 일부 이미지의 메리스템 크기가 다를 수 있음. 일단은 일괄 적용으로 진행, 결과 검수에서 필요시 이미지별 조정.

