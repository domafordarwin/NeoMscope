#!/usr/bin/env bash
# Compile a YOLOv11-det ONNX model to Hailo .hef for AI HAT+ 2 (Hailo-10H).
#
# Run inside WSL2 Ubuntu (Windows host) or a native Linux machine. Requires
# Hailo Dataflow Compiler (DFC) 4.x+ and the Hailo Model Zoo `master` branch
# (Hailo-10H is NOT supported on the v2.x branch).
#
# Usage (from project root):
#   bash tools/compile_hef.sh \
#       models/onnx/best.onnx \
#       models/hef/best.hef \
#       datasets/onioncell/calib_imgs
#
# Or from WSL2 with Windows paths via /mnt/c:
#   ROOT="/mnt/c/WorkSpace/Project/neoscope/deploy_detect_onioncell"
#   bash "$ROOT/tools/compile_hef.sh" "$ROOT/models/onnx/best.onnx" \
#       "$ROOT/models/hef/best.hef" "$ROOT/datasets/onioncell/calib_imgs"

set -euo pipefail

ONNX="${1:?usage: $0 <onnx> <hef> <calib_dir>}"
HEF="${2:?usage: $0 <onnx> <hef> <calib_dir>}"
CALIB="${3:?usage: $0 <onnx> <hef> <calib_dir>}"
ARCH="${HAILO_ARCH:-hailo10h}"   # Override with HAILO_ARCH=hailo8 for Hailo-8

# --- Pre-flight ---
if ! command -v hailomz >/dev/null 2>&1; then
    # Try the standard WSL2 venv path
    VENV_HAILOMZ="/root/hailo-workspace/venv/bin/hailomz"
    if [[ -x "$VENV_HAILOMZ" ]]; then
        echo "[compile_hef.sh] Found hailomz at $VENV_HAILOMZ — re-exec with venv on PATH"
        export PATH="/root/hailo-workspace/venv/bin:$PATH"
    else
        echo "[compile_hef.sh] ERROR: 'hailomz' not found on PATH."
        echo "[compile_hef.sh] Run 'bash tools/install_hailo_dfc.sh' first, then either:"
        echo "[compile_hef.sh]   source /root/hailo-workspace/venv/bin/activate"
        echo "[compile_hef.sh] or re-run this script (it will auto-detect the venv)."
        exit 2
    fi
fi

if [[ ! -f "$ONNX" ]]; then
    echo "[compile_hef.sh] ERROR: ONNX file not found: $ONNX"
    exit 2
fi

if [[ ! -d "$CALIB" ]]; then
    echo "[compile_hef.sh] ERROR: Calibration directory not found: $CALIB"
    echo "[compile_hef.sh] Create it with at least 64 representative images:"
    echo "[compile_hef.sh]   mkdir -p $CALIB && cp datasets/onioncell/images/train/*.jpg $CALIB/"
    exit 2
fi

n_calib=$(find "$CALIB" -type f \( -iname "*.jpg" -o -iname "*.png" \) | wc -l)
if [[ "$n_calib" -lt 64 ]]; then
    echo "[compile_hef.sh] WARNING: only $n_calib calibration images. Recommend >= 64."
fi

mkdir -p "$(dirname "$HEF")"

# --- Compile ---
echo "[compile_hef.sh] Compiling: $ONNX -> $HEF"
echo "[compile_hef.sh]   arch:  $ARCH"
echo "[compile_hef.sh]   calib: $CALIB ($n_calib images)"

# yolov11s det config from the Model Zoo master branch.
# If your trained model uses a different size (m/l/n), change the YAML accordingly.
# Note: file is yolov11s.yaml (with 'v'), not yolo11s.yaml.
hailomz compile \
    --ckpt "$ONNX" \
    --calib-path "$CALIB" \
    --yaml "yolov11s.yaml" \
    --hw-arch "$ARCH" \
    --output-dir "$(dirname "$HEF")"

# hailomz writes the HEF using the network name; rename to the requested path.
generated_hef=$(find "$(dirname "$HEF")" -name "*.hef" -newer "$ONNX" | head -1)
if [[ -n "$generated_hef" && "$generated_hef" != "$HEF" ]]; then
    mv "$generated_hef" "$HEF"
fi

echo "[compile_hef.sh] ✅ HEF compiled: $HEF"
echo "[compile_hef.sh] Size: $(du -h "$HEF" | cut -f1)"
echo ""
echo "[compile_hef.sh] Next: copy to Pi 5 and verify with hailortcli:"
echo "[compile_hef.sh]   scp $HEF pi@raspberrypi.local:~/neomscope/models/hef/"
echo "[compile_hef.sh]   ssh pi@raspberrypi.local 'hailortcli run ~/neomscope/models/hef/best.hef'"
