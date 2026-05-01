#!/usr/bin/env bash
# Verify that the Hailo DFC + Model Zoo are correctly installed in the
# WSL2 Ubuntu venv at /root/hailo-workspace/venv.
#
# Run from inside WSL2 Ubuntu (or via `wsl -d Ubuntu-22.04 --user root`).
# Exits non-zero if any check fails.

set -e

VENV="/root/hailo-workspace/venv"
fail=0

echo "[verify] DFC version:"
"$VENV/bin/hailo" --version || { echo "  ❌ hailo CLI missing"; fail=1; }

echo ""
echo "[verify] hailomz CLI:"
"$VENV/bin/hailomz" --help > /dev/null && echo "  ✅ available" || { echo "  ❌ hailomz missing"; fail=1; }

echo ""
echo "[verify] yolov11s config (target model):"
"$VENV/bin/hailomz" info yolov11s 2>&1 | head -10 || { echo "  ❌ yolov11s config not found"; fail=1; }

echo ""
echo "[verify] yolov8s config (backup model):"
"$VENV/bin/hailomz" info yolov8s 2>&1 | head -3 || { echo "  ❌ yolov8s config not found"; fail=1; }

echo ""
if [[ $fail -eq 0 ]]; then
    echo "[verify] ✅ All checks passed."
else
    echo "[verify] ❌ One or more checks failed."
    exit 1
fi
