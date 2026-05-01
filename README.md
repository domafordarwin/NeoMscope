# NeoMscope

> 과학전람회용 양파 세포 분열 검출 시스템 — Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H NPU) 엣지 추론.

현미경/웹캠으로 촬영한 양파 표피세포 이미지에서 분열 단계 5종(**Inter, Pro, Meta, Ana, Telo**)을 실시간 자동 검출·계수합니다.

## v2.0 (진행 중)

| 항목 | v1.0 (legacy/) | **v2.0** |
|---|---|---|
| 모델 | Mask R-CNN (TF 1.15) | **YOLOv11-det** (Ultralytics) |
| 학습 | 로컬 GPU | **Kaggle T4/P100 16GB** |
| 추론 | PC + GPU | **Pi 5 + AI HAT+ 2** (Hailo-10H) |
| 입력 | 512×512 | 640×640 |

## 빠른 시작

### 1. 레포 클론 + venv
```powershell
git clone https://github.com/domafordarwin/NeoMscope.git
cd NeoMscope
uv venv --python 3.11 .venv
uv pip install -e ".[dev,tools]"
```

### 2. 테스트
```powershell
.venv\Scripts\python.exe -m pytest tests/unit/
```

### 3. 레거시 venv (부트스트랩 라벨링용, 1회 셋업)
```powershell
conda create -n neomscope-legacy python=3.7 "tensorflow=1.15.0=eigen_py37h9f89a44_0" "numpy=1.19" pillow -y
conda run -n neomscope-legacy pip install "keras==2.2.5" "scikit-image==0.17.2" "opencv-python==4.5.5.64" "imutils==0.5.4" "h5py<3.0"
conda run -n neomscope-legacy pip install "git+https://github.com/matterport/Mask_RCNN.git"
```

### 4. Pi 5 배포 (학습·HEF 컴파일 후)
```bash
bash setup.sh                          # HailoRT + venv + self-test
neomscope-live --camera /dev/video0    # 실시간 검출
```

## 워크플로 (Pipeline)

```
0. 환경 준비           ─── ✅ 완료 (Group 0)
1. 부트스트랩 라벨링    ─── ✅ 완료 (127장 → 399 detections, avg conf 0.72)
2. Roboflow 검수+Export ─── 🔵 사용자 작업 중 (uploads runs/roboflow-upload.zip)
3. Kaggle 학습          ─── ⏳ 다음 — training/notebooks/train-yolo11-det-kaggle.ipynb
4. ONNX export          ─── ⏳ tools/export_onnx.py
5. HEF 컴파일 (WSL2)    ─── ⏳ tools/compile_hef.sh
6. Pi 5 추론 데모        ─── ⏳ neomscope-live / batch / capture
```

## 핵심 문서

- 계획: [docs/01-plan/features/aihat-yolo-port.plan.md](docs/01-plan/features/aihat-yolo-port.plan.md)
- 설계: [docs/02-design/features/aihat-yolo-port.design.md](docs/02-design/features/aihat-yolo-port.design.md)
- AI 협업 가이드: [CLAUDE.md](CLAUDE.md)
- 레거시 코드: [legacy/README.md](legacy/README.md)

## 라이선스

MIT (자체 코드). YOLO/Ultralytics 가중치는 별도 라이선스 적용.
