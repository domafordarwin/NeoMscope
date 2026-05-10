#!/usr/bin/env bash
# NeoMscope Pi 5 + AI HAT+ 2 bootstrap script (FR-10).
#
# Run this on a fresh Raspberry Pi 5 with AI HAT+ 2 attached, after first boot.
# Prerequisites:
#   - Raspberry Pi OS Bookworm 64-bit
#   - AI HAT+ 2 (Hailo-10H) physically attached and recognized by lspci
#
# What this script does:
#   1. Apt update + install GStreamer + Python venv tools
#   2. Add Hailo APT repo and install HailoRT (NPU runtime) + hailo-tappas
#   3. Create Python 3.11 venv and install neomscope[inference,gui]
#      (CLI scripts AND PySide6 GUI both supported by default)
#   4. Verify HEF model loads (if models/hef/best.hef exists)
#
# Flags:
#   --no-gui       Skip PySide6 install (headless deploy / save 200 MB)
#   --no-hailo     Skip Hailo APT setup (useful when re-running on Pi 4 / x86 dev)
#   --no-self-test Skip the hailortcli + HEF probe at the end
#
# Re-runs are idempotent.

set -euo pipefail

# ---- Flags ----
INSTALL_GUI=1
INSTALL_HAILO=1
RUN_SELF_TEST=1

for arg in "$@"; do
    case "$arg" in
        --no-gui)       INSTALL_GUI=0 ;;
        --no-hailo)     INSTALL_HAILO=0 ;;
        --no-self-test) RUN_SELF_TEST=0 ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "[setup.sh] Unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "[setup.sh] Project root: $PROJECT_ROOT"
echo "[setup.sh] GUI: $([ $INSTALL_GUI = 1 ] && echo on || echo off) | Hailo: $([ $INSTALL_HAILO = 1 ] && echo on || echo off) | Self-test: $([ $RUN_SELF_TEST = 1 ] && echo on || echo off)"

# ---- Pre-flight checks ----
IS_PI=0
if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    IS_PI=1
    echo "[setup.sh] Detected: $(tr -d '\0' </proc/device-tree/model)"
else
    echo "[setup.sh] WARNING: not a Raspberry Pi. Continuing in dev/test mode."
fi

if [[ $INSTALL_HAILO = 1 ]] && ! lspci 2>/dev/null | grep -qi "hailo"; then
    echo "[setup.sh] WARNING: Hailo accelerator not detected via lspci."
    echo "           Check AI HAT+ 2 attachment + ribbon cable + dtparam=pciex1 in /boot/firmware/config.txt"
fi

# ---- 1. System packages ----
echo "[setup.sh] Installing system packages..."
APT_PKGS=(
    python3.11 python3.11-venv python3-pip
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    libgirepository1.0-dev libcairo2-dev pkg-config
    v4l-utils
)
if [[ $INSTALL_GUI = 1 ]]; then
    # PySide6 needs a few libs the wheel links against
    APT_PKGS+=(
        libxkbcommon0 libxkbcommon-x11-0
        libgl1 libglib2.0-0
        libfontconfig1 libdbus-1-3
        fonts-noto-cjk
    )
fi
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

# ---- 2. Hailo APT repo + runtime ----
if [[ $INSTALL_HAILO = 1 ]]; then
    echo "[setup.sh] Setting up Hailo APT repository..."
    if ! command -v hailortcli >/dev/null 2>&1; then
        curl -fsSL https://hailo.ai/developer-zone/sw-downloads/hailo.gpg \
            | sudo gpg --dearmor -o /usr/share/keyrings/hailo.gpg
        echo "deb [signed-by=/usr/share/keyrings/hailo.gpg] https://hailo.ai/apt stable main" \
            | sudo tee /etc/apt/sources.list.d/hailo.list
        sudo apt-get update -y
    fi

    echo "[setup.sh] Installing HailoRT + TAPPAS..."
    sudo apt-get install -y --no-install-recommends \
        hailort hailo-tappas-core python3-hailo-platform || {
        echo "[setup.sh] APT install failed. Falling back to local .deb if present..."
        if compgen -G "$PROJECT_ROOT/pi5-deploy/hailort-pcie-driver_*.deb" >/dev/null; then
            echo "[setup.sh] Installing $PROJECT_ROOT/pi5-deploy/hailort-pcie-driver_*.deb"
            sudo apt-get install -y "$PROJECT_ROOT"/pi5-deploy/hailort-pcie-driver_*.deb
        else
            echo "[setup.sh] No local .deb found in pi5-deploy/. Check Hailo Dev Zone." >&2
            exit 1
        fi
    }
fi

# ---- 3. Python venv ----
echo "[setup.sh] Creating Python 3.11 virtual environment at $VENV_DIR..."
if [[ ! -d "$VENV_DIR" ]]; then
    # --system-site-packages so the venv can import system python3-hailo-platform
    python3.11 -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Build the extras list dynamically
EXTRAS="inference"
if [[ $INSTALL_GUI = 1 ]]; then
    EXTRAS="${EXTRAS},gui"
fi
echo "[setup.sh] pip install -e .[${EXTRAS}]"
pip install -e "$PROJECT_ROOT[$EXTRAS]"

# ---- 4. Self-test ----
if [[ $RUN_SELF_TEST = 1 ]]; then
    echo "[setup.sh] Running self-test..."

    # CLI imports
    python - <<'PY'
import sys
print(f"  Python {sys.version.split()[0]}  OK")
import inference.types  # noqa: F401
import inference.pipeline  # noqa: F401
import inference.postprocess  # noqa: F401
import inference._camera  # noqa: F401
print("  inference.* imports  OK")
PY

    # GUI imports (only if GUI was installed)
    if [[ $INSTALL_GUI = 1 ]]; then
        python - <<'PY'
import inference.ui.main_window  # noqa: F401
import inference.ui.tabs.live_tab  # noqa: F401
print("  inference.ui.* imports  OK")
PY
    fi

    # Hailo runtime check
    if [[ $INSTALL_HAILO = 1 ]]; then
        if hailortcli fw-control identify 2>/dev/null; then
            echo "  hailortcli  OK"
        else
            echo "  WARNING: hailortcli could not talk to NPU. Check /dev/hailo0 permissions, lspci, dmesg."
        fi
    fi

    # HEF probe
    if [[ -f "$PROJECT_ROOT/models/hef/best.hef" ]]; then
        python - <<PY
from hailo_platform import HEF
hef = HEF("$PROJECT_ROOT/models/hef/best.hef")
print(f'  HEF loaded: {hef.get_network_group_names()}')
PY
    else
        echo "  (no HEF at models/hef/best.hef — copy after WSL2 compile)"
    fi
fi

cat <<EOF

[setup.sh] ✅ Done.

  Activate the venv:
      source $VENV_DIR/bin/activate

  CLI usage (legacy / scripts):
      neomscope-live    --camera /dev/video0
      neomscope-batch   --input captured/ --output results/
      neomscope-capture --camera 0 --output detection_results_captured/

EOF

if [[ $INSTALL_GUI = 1 ]]; then
    cat <<EOF
  GUI usage (PySide6, Pi 5 7" 1024x600 DSI):
      neomscope-gui

  Headless screenshot of the GUI (CI / docs):
      python -m inference.ui --screenshot /tmp/neomscope.png

EOF
fi
