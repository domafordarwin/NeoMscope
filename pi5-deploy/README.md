# Pi 5 Deployment (192.168.123.110)

How to install NeoMscope on the Raspberry Pi 5 + AI HAT+ 2 so that **both
GUIs are available**:

* **기존 GUI** — `live_detect.py` / `batch_detect.py` / `capture_and_detect.py`.
  Each opens a single OpenCV window (`cv2.imshow`) at full screen, launched
  from the terminal. Lightweight, fast, and the natural fit for a single-
  task workflow. Same Korean class labels and color scheme.
* **신 GUI (v2)** — the PySide6 desktop app from `inference/ui/`. Multi-tab
  (Live / Batch / Archive / Settings), polished, touch-friendly, fits the
  Pi 5's 1024×600 DSI screen. Same `inference.pipeline` underneath, so
  detection results are identical.

Pick whichever matches what you're doing — both run from the same venv.

## TL;DR

```powershell
# Once: edit tools/pi-config.env with your Pi's IP / user
cp tools/pi-config.example.env tools/pi-config.env
notepad tools/pi-config.env

# Verify reachability
.\tools\deploy_to_pi.ps1 -Probe

# Full install (CLI + GUI + Hailo runtime)
.\tools\deploy_to_pi.ps1
```

## What the script does

1. **Local**: reads `tools/pi-config.env` (or falls back to the example file).
2. **Probe**: pings the Pi, then runs an SSH probe to print kernel info and
   `lspci | grep hailo`.
3. **Deploy**: SSH'es in and either `git pull` (if a clone already exists) or
   `git clone https://github.com/domafordarwin/NeoMscope.git` (first run).
4. **Setup**: runs `bash setup.sh` on the Pi, which installs:
   - APT system packages (GStreamer, Python venv tools, Qt runtime libs,
     Noto CJK fonts for Korean)
   - Hailo APT repo + HailoRT (NPU runtime) + TAPPAS
   - Python 3.11 venv at `~/NeoMscope/.venv` with
     `pip install -e .[inference,gui]` — CLI scripts AND PySide6 GUI
5. **Self-test**: imports `inference.*` + `inference.ui.*`, runs
   `hailortcli fw-control identify`, and probes `models/hef/best.hef` if
   present.

## Two GUIs, both supported

After `setup.sh` finishes, on the Pi:

```bash
ssh pi@192.168.123.110
cd ~/NeoMscope && source .venv/bin/activate

# ─── 기존 GUI (single OpenCV window) ──────────────────────────
neomscope-live    --camera /dev/video0
neomscope-batch   --input captured/ --output results/
neomscope-capture --camera 0 --output detection_results_captured/

# ─── 신 GUI v2 (PySide6 multi-tab) ────────────────────────────
neomscope-gui

# Headless screenshot of the v2 GUI (e.g. SCP back to dev PC):
python -m inference.ui --screenshot /tmp/neomscope.png
scp pi@192.168.123.110:/tmp/neomscope.png ./
```

> Both GUIs need a display. Options on a Pi 5:
>
> * Local screen (HDMI or 7" DSI) — easiest, full-screen friendly
> * SSH with X11 forwarding: `ssh -X pi@192.168.123.110`
>   (works for both; cv2.imshow and Qt both honor `$DISPLAY`)
> * VNC: enable in `raspi-config`, then connect from Windows / phone
> * Headless: only the v2 GUI supports `--screenshot`; the OpenCV scripts
>   use `cv2.imshow` which always wants a display.

## Connecting to the Pi the first time (Windows)

```powershell
# 1. Generate an SSH key (if you don't have one yet)
ssh-keygen -t ed25519

# 2. Copy it to the Pi (one-time password prompt)
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh pi@192.168.123.110 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 3. From now on, no password needed
ssh pi@192.168.123.110
```

## Re-deploying after code changes

```powershell
# On dev PC: commit + push
git add . ; git commit -m "fix: ..." ; git push

# Pull + run setup on Pi
.\tools\deploy_to_pi.ps1
```

`setup.sh` is idempotent — re-running is a no-op if everything is already
installed. After a re-deploy, `pip install -e .` picks up code changes
because `-e` (editable) means `import` reads from the working tree.

## Skipping pieces

If you only want the existing OpenCV GUI (no PySide6, save ~200 MB):

```powershell
.\tools\deploy_to_pi.ps1 -Flags "--no-gui"
```

(`--no-gui` means "don't install PySide6". The OpenCV-based scripts are
always installed — they're part of the core `[inference]` extras.)

If your Pi isn't connected to the AI HAT+ 2 yet (testing on bare Pi 5):

```powershell
.\tools\deploy_to_pi.ps1 -Flags "--no-gui --no-hailo --no-self-test"
```

## Files in this folder

| File | Purpose |
|------|---------|
| `hailort-pcie-driver_*.deb` | Manual fallback PCIe driver if the Hailo APT repo can't reach the Pi (gitignored — download from Hailo Dev Zone) |
| `README.md` | This file |

## Pi 5 prerequisites checklist

- [ ] Pi OS Bookworm 64-bit installed and updated (`sudo apt update && sudo apt upgrade`)
- [ ] AI HAT+ 2 physically attached, ribbon cable seated, fan running
- [ ] `dtparam=pciex1` in `/boot/firmware/config.txt` (auto-set by the Pi imager since 2025)
- [ ] `lspci | grep -i hailo` shows the device
- [ ] User account created during Pi imager setup (no default `pi:raspberry` password anymore)
- [ ] SSH enabled
- [ ] (For GUI) HDMI display or 7" DSI touchscreen attached, OR VNC server set up
