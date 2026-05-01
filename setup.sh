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
#   3. Create Python 3.11 venv and install neomscope[inference]
#   4. Verify HEF model loads (if models/hef/best.hef exists)
#
# Re-runs are idempotent (apt install is no-op on already-installed packages).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# ---- Pre-flight checks ----
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "[setup.sh] WARNING: this does not look like a Raspberry Pi. Continuing anyway."
fi

if ! lspci 2>/dev/null | grep -qi "hailo"; then
    echo "[setup.sh] WARNING: Hailo accelerator not detected via lspci. Check AI HAT+ 2 connection."
fi

# ---- 1. System packages ----
echo "[setup.sh] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev libcairo2-dev pkg-config \
    v4l-utils

# ---- 2. Hailo APT repo + runtime ----
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
    echo "[setup.sh] WARNING: Hailo packages failed to install via APT."
    echo "[setup.sh] Fallback: download HailoRT .deb from https://hailo.ai/developer-zone/"
    echo "[setup.sh] and install manually with: sudo dpkg -i hailort_*.deb"
    exit 1
}

# ---- 3. Python venv ----
echo "[setup.sh] Creating Python 3.11 virtual environment at $VENV_DIR..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3.11 -m venv --system-site-packages "$VENV_DIR"
    # --system-site-packages so we can use system python3-hailo-platform
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -e "$PROJECT_ROOT[inference]"

# ---- 4. Self-test ----
echo "[setup.sh] Running self-test..."
hailortcli fw-control identify || {
    echo "[setup.sh] ERROR: hailortcli could not communicate with Hailo device."
    echo "[setup.sh] Check: lspci, sudo systemctl status hailo_pcie, /dev/hailo0 permissions."
    exit 1
}

if [[ -f "$PROJECT_ROOT/models/hef/best.hef" ]]; then
    echo "[setup.sh] Verifying HEF can be loaded..."
    python -c "
from hailo_platform import HEF
hef = HEF('$PROJECT_ROOT/models/hef/best.hef')
print(f'  ✓ HEF loaded: {hef.get_network_group_names()}')
"
else
    echo "[setup.sh] (No HEF found at models/hef/best.hef — skipping load test.)"
    echo "[setup.sh] After training, copy best.hef from dev PC and re-run this script."
fi

echo ""
echo "[setup.sh] ✅ Done. Activate the venv with:"
echo "           source $VENV_DIR/bin/activate"
echo ""
echo "[setup.sh] Try a live demo (USB webcam on /dev/video0):"
echo "           neomscope-live --camera /dev/video0"
