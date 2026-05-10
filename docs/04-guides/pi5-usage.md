# Pi 5 사용 가이드 (192.168.123.110)

> NeoMscope를 Raspberry Pi 5 + AI HAT+ 2에 배포하고 두 GUI(기존 OpenCV / 신 PySide6 v2)를 실행하는 전체 절차.

---

## 0. 사전 준비 (한 번만)

### 0-1. Pi 측에서 확인할 것

Pi에 직접 접속(키보드/모니터 또는 SSH로 일단 한 번)해서:

```bash
# (a) IP 확인 — 우리는 192.168.123.110으로 가정
hostname -I

# (b) SSH 활성화 (대부분 Pi imager에서 켜짐, 안 됐으면)
sudo raspi-config nonint do_ssh 0

# (c) 사용자명 확인 (Bookworm은 imager에서 만든 이름)
whoami
```

`whoami`가 출력하는 이름이 **PI_USER** 입니다 (예: `pi`, `domam` 등).

### 0-2. 개발 PC 측에서 SSH 키 등록 (한 번만)

```powershell
# (a) SSH 키가 없으면 생성
ssh-keygen -t ed25519
# Enter 4번 (passphrase는 빈 값 또는 본인 선택)

# (b) Pi에 공개키 등록 — 이때 한 번만 비밀번호 입력
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pi@192.168.123.110 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# (c) 비밀번호 없이 접속 확인
ssh pi@192.168.123.110 "echo ✅ SSH OK; uname -a"
```

> 첫 접속 시 "The authenticity of host ... can't be established" 메시지가 뜨면 `yes` 입력.

---

## 1. 프로젝트 설정 (한 번만)

```powershell
cd c:\WorkSpace\Project\neoscope\deploy_detect_onioncell

# (a) Pi 연결 설정 파일 생성
copy tools\pi-config.example.env tools\pi-config.env
notepad tools\pi-config.env
```

`tools/pi-config.env` 내용 편집:
```env
PI_HOST=192.168.123.110
PI_USER=pi          ← 본인 Pi 사용자명으로
PI_PORT=22
PI_PROJECT_DIR=NeoMscope
GIT_REMOTE=https://github.com/domafordarwin/NeoMscope.git
GIT_BRANCH=main
SETUP_FLAGS=
```

저장 후 닫기.

### (선택) PowerShell 실행 정책 설정

처음 `.ps1` 실행 시 차단되면:
```powershell
# 현재 사용자만, 영구 적용
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 또는 일회용
PowerShell -ExecutionPolicy Bypass -File .\tools\deploy_to_pi.ps1
```

---

## 2. 연결 테스트

배포 전 연결만 확인:

```powershell
.\tools\deploy_to_pi.ps1 -Probe
```

성공하면:
```
[deploy] target:    pi@192.168.123.110 (port 22)
[deploy] ping 192.168.123.110...
[deploy]   ping OK
[deploy] ssh probe...
Linux raspberrypi 6.6.x ...
Raspberry Pi 5 Model B Rev 1.0
0001:01:00.0 Co-processor: Hailo Technologies Ltd. ...
[deploy]   ssh OK
[deploy] probe complete (no deploy).
```

문제 발생 시 → **"문제 해결" 섹션** 참조.

---

## 3. 첫 배포 — 모든 패키지 + 두 GUI 설치

```powershell
.\tools\deploy_to_pi.ps1
```

소요 시간: **15-25분** (인터넷 속도, APT 캐시 상태에 따라).

진행 단계 (스크립트가 자동):
1. `git clone` Pi에 NeoMscope 다운로드
2. APT: GStreamer + Python 3.11 + Qt runtime + Noto CJK 폰트
3. Hailo APT 리포 + HailoRT + TAPPAS 설치
4. Python venv 생성 + `pip install -e .[inference,gui]`
5. 자체 검증 (CLI/GUI import + `hailortcli`)

성공 시 마지막 줄:
```
[deploy] ✅ Done.
```

---

## 4. Pi에서 GUI 실행

### 4-1. SSH 접속 + venv 활성화

```powershell
ssh pi@192.168.123.110
```

Pi 안에서:
```bash
cd ~/NeoMscope
source .venv/bin/activate
```

### 4-2. 기존 GUI (OpenCV 단일 창)

```bash
# 실시간 검출 (USB 웹캠)
neomscope-live --camera /dev/video0

# 일괄 검출 (이미지 폴더)
neomscope-batch --input ~/test_images --output ~/results

# 캡처 + 즉시 검출 (스페이스바로 캡처)
neomscope-capture --camera 0 --output ~/captures
```

종료: 창에 포커스 → `q` 키 또는 Ctrl+C.

### 4-3. 신 GUI v2 (PySide6 4탭)

```bash
neomscope-gui
```

탭별 기능:
- **Live**: 실시간 카메라 + 검출 + chip 카운터 + FPS
- **Batch**: 폴더 선택 + 진행 막대 + 로그
- **Archive**: 저장된 결과 썸네일 브라우징
- **Settings**: HEF / 카메라 / 임계값 / 출력 디렉터리

### 4-4. 헤드리스 스크린샷 (v2만)

디스플레이 없는 환경에서도 GUI 렌더 가능:
```bash
# Pi 측
python -m inference.ui --screenshot /tmp/neomscope.png

# Windows 측에서 가져오기
exit  # SSH 빠져나옴
scp pi@192.168.123.110:/tmp/neomscope.png .\
```

---

## 5. 화면 표시 방법

두 GUI 모두 `$DISPLAY`가 필요합니다. 옵션:

### 옵션 A — Pi 직접 연결 (권장)
- HDMI 모니터 또는 7" DSI 터치스크린 직접 연결
- Pi 부팅 후 데스크톱 자동 로그인 → 터미널 열기 → 위 명령 실행

### 옵션 B — SSH X11 Forwarding
Windows에서 `-X` 또는 `-Y` 플래그:
```powershell
ssh -X pi@192.168.123.110
```
필요: Windows 측에 [VcXsrv](https://sourceforge.net/projects/vcxsrv/) 또는 [Xming](https://sourceforge.net/projects/xming/) 설치 후 실행 중이어야 함.

### 옵션 C — VNC (가장 간편)
Pi:
```bash
sudo raspi-config
# Interface Options → VNC → Enable
```
Windows:
- [RealVNC Viewer](https://www.realvnc.com/connect/download/viewer/) 또는 TigerVNC
- 주소: `192.168.123.110:5900`

VNC가 가장 안정적이고 한글 폰트가 깨끗하게 나옵니다.

### 옵션 D — 헤드리스 스크린샷
디스플레이 없이 v2 GUI를 PNG로:
```bash
python -m inference.ui --screenshot /tmp/v2.png
```
(기존 OpenCV 스크립트는 이 방식 지원 안 함)

---

## 6. 코드 수정 후 재배포

개발 PC에서 코드 수정 → push → Pi에 반영:

```powershell
# 개발 PC
git add . ; git commit -m "fix: ..." ; git push

# 같은 명령 한 번 더 — git pull + setup.sh 재실행 (idempotent)
.\tools\deploy_to_pi.ps1
```

`pip install -e .` (editable) 이므로 `import inference.*`가 자동으로 새 코드를 읽습니다.

---

## 7. 시연·전람회 시나리오

```bash
# Pi 부팅 후 자동 시작 (시스템 서비스로 등록 시 — 다음 절에서)
# 또는 수동:
ssh pi@192.168.123.110
cd ~/NeoMscope && source .venv/bin/activate

# 풀스크린 데모용 — v2 GUI 자동 풀스크린 (현재 구현 전: 추후 옵션 추가)
neomscope-gui

# 또는 라이트한 단일 창
neomscope-live --camera /dev/video0
```

자동 시작 systemd 서비스는 추후 추가 가능 (요청 시).

---

## 8. 부분 설치 옵션

### CLI만 (PySide6 미설치, 200MB 절약)
```powershell
.\tools\deploy_to_pi.ps1 -Flags "--no-gui"
```

### AI HAT+ 2 미연결 상태 (코드 검증만)
```powershell
.\tools\deploy_to_pi.ps1 -Flags "--no-hailo --no-self-test"
```

### Pi 호스트 임시 변경 (테스트용)
```powershell
.\tools\deploy_to_pi.ps1 -PiHost 192.168.1.50 -User domam
```

### dry-run (실제 실행 안 함, 명령만 미리보기)
```powershell
.\tools\deploy_to_pi.ps1 -DryRun
```

---

## 9. 문제 해결

### `ping` 실패
- Pi 전원 확인
- `192.168.123.110` 주소 정확한지 확인 (Pi에서 `hostname -I`)
- 같은 네트워크 (라우터 / Wi-Fi / 이더넷)에 있는지

### `ssh` 실패: "Connection refused"
Pi에 SSH 활성화:
```bash
sudo raspi-config nonint do_ssh 0
```

### `ssh` 실패: "Permission denied (publickey)"
공개키 등록 안 됨:
```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pi@192.168.123.110 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### `setup.sh` 실패: Hailo APT 리포 안 닿음
폴백: `pi5-deploy/hailort-pcie-driver_*.deb` 자동으로 설치됨. 만약 그것도 없으면:
```bash
# Pi에서
ls ~/NeoMscope/pi5-deploy/
# .deb 파일이 있으면:
sudo dpkg -i ~/NeoMscope/pi5-deploy/hailort-pcie-driver_*.deb
sudo reboot
```

### `hailortcli` 실패
```bash
lspci | grep -i hailo  # 디바이스 보이는지
ls /dev/hailo*         # 드라이버 노드 보이는지
sudo dmesg | grep -i hailo  # 커널 로그
```

대부분 PCIe 활성화 누락 — `/boot/firmware/config.txt`에 `dtparam=pciex1` 추가 후 재부팅.

### GUI 실행 시 "could not connect to display"
- HDMI 화면 없음 → 위 §5 옵션 B/C 참조
- SSH X11: `ssh -X` 사용 + Windows 측 VcXsrv 실행 확인
- VNC: `raspi-config`에서 활성화

### Qt가 렌더링 안 됨 / 깜빡임
필수 OpenGL 라이브러리 누락 가능성. 재시도:
```bash
sudo apt install libgl1-mesa-dri libgl1
```

---

## 10. 빠른 참조

| 작업 | 명령 |
|---|---|
| 연결만 확인 | `.\tools\deploy_to_pi.ps1 -Probe` |
| 첫 배포 (모든 것) | `.\tools\deploy_to_pi.ps1` |
| 코드만 새로 (구조 변경 없음) | `.\tools\deploy_to_pi.ps1` (재실행 OK) |
| Pi에 SSH | `ssh pi@192.168.123.110` |
| 기존 GUI 실행 | `neomscope-live --camera /dev/video0` |
| v2 GUI 실행 | `neomscope-gui` |
| 헤드리스 v2 스크린샷 | `python -m inference.ui --screenshot out.png` |
| Pi 측 git pull | `cd ~/NeoMscope && git pull` |
| venv 활성화 | `source ~/NeoMscope/.venv/bin/activate` |

---

## 관련 문서

- 전체 README: [../README.md](../../README.md)
- AI 협업 가이드: [../CLAUDE.md](../../CLAUDE.md)
- Pi 폴더 README: [../pi5-deploy/README.md](../../pi5-deploy/README.md)
- v2 GUI 코드: [../inference/ui/](../../inference/ui/)
- 진행 보고서: [../03-analysis/2026-05-01-progress-report.md](../03-analysis/2026-05-01-progress-report.md)
