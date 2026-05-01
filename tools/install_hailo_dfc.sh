#!/usr/bin/env bash
# Install Hailo Dataflow Compiler into the WSL2 Ubuntu venv at
# /root/hailo-workspace/venv. Auto-detects the .whl in either:
#   1. /mnt/c/Users/<windows-user>/Downloads/  (default)
#   2. The first argument if given
#
# Run from inside WSL2 Ubuntu (or via `wsl -d Ubuntu-22.04 --user root`).
#
# Usage:
#   bash tools/install_hailo_dfc.sh
#   bash tools/install_hailo_dfc.sh /path/to/hailo_dataflow_compiler-X.Y.Z-py3-none-linux_x86_64.whl
#
# Idempotent — re-running with the same .whl does nothing.

set -euo pipefail

VENV="/root/hailo-workspace/venv"
WORKSPACE="/root/hailo-workspace"
WHL_ARG="${1:-}"

# 1. Sanity
if [[ ! -d "$VENV" ]]; then
    echo "[install_hailo_dfc.sh] ERROR: venv not found at $VENV"
    echo "  Run: python3 -m venv $VENV"
    exit 2
fi

# 2. Locate .whl
if [[ -n "$WHL_ARG" ]]; then
    WHL="$WHL_ARG"
else
    # Search Downloads of every Windows user, since we're running in WSL
    candidates=()
    for user_dir in /mnt/c/Users/*/Downloads/; do
        while IFS= read -r f; do
            candidates+=("$f")
        done < <(find "$user_dir" -maxdepth 1 -name "hailo_dataflow_compiler*.whl" 2>/dev/null)
    done

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "[install_hailo_dfc.sh] ERROR: no hailo_dataflow_compiler*.whl found in Downloads."
        echo "  Download from https://hailo.ai/developer-zone/sw-downloads/ first,"
        echo "  or pass the path explicitly: bash tools/install_hailo_dfc.sh /path/to/.whl"
        exit 2
    elif [[ ${#candidates[@]} -gt 1 ]]; then
        echo "[install_hailo_dfc.sh] Multiple .whl files found:"
        for c in "${candidates[@]}"; do echo "    $c"; done
        echo "  Pass one explicitly as the first argument."
        exit 2
    fi
    WHL="${candidates[0]}"
fi

if [[ ! -f "$WHL" ]]; then
    echo "[install_hailo_dfc.sh] ERROR: not a file: $WHL"
    exit 2
fi

echo "[install_hailo_dfc.sh] Installing: $(basename "$WHL")"

# 3. Idempotency: skip if already installed at this version
"$VENV/bin/pip" show hailo_dataflow_compiler 2>/dev/null | grep -E '^Version:' || echo "(not yet installed)"

# 4. Install
"$VENV/bin/pip" install --upgrade pip wheel setuptools --quiet
"$VENV/bin/pip" install "$WHL"

# 5. Verify
echo ""
echo "[install_hailo_dfc.sh] Verifying installation..."
"$VENV/bin/hailo" --version || {
    echo "[install_hailo_dfc.sh] ERROR: 'hailo' CLI not on venv PATH."
    exit 1
}

# 6. Install Hailo Model Zoo (master branch already cloned)
if [[ -d "$WORKSPACE/hailo_model_zoo" ]]; then
    echo ""
    echo "[install_hailo_dfc.sh] Installing Hailo Model Zoo (editable, from clone)..."

    # Hailo's setup.py expects a versions.py file 3 directories above setup.py
    # — meant for installation from inside their internal monorepo. Create a
    # shim so pip install -e . succeeds outside that monorepo.
    DFC_VER="$("$VENV/bin/pip" show hailo_dataflow_compiler | awk '/^Version:/ {print $2}')"
    SHIM="/root/versions.py"
    if [[ ! -f "$SHIM" ]] || ! grep -q "$DFC_VER" "$SHIM" 2>/dev/null; then
        printf 'DFC_VERSION = "%s"\nMZ_VERSION = "%s"\n' "$DFC_VER" "$DFC_VER" > "$SHIM"
        echo "  Wrote $SHIM (DFC_VERSION=$DFC_VER)"
    fi

    # Must `cd` into the package dir for the editable install to find files
    (cd "$WORKSPACE/hailo_model_zoo" && "$VENV/bin/pip" install --quiet -e .)
    "$VENV/bin/hailomz" --help > /dev/null && echo "  hailomz CLI OK"
else
    echo "[install_hailo_dfc.sh] Skipping model zoo install (clone not found at $WORKSPACE/hailo_model_zoo)"
fi

echo ""
echo "[install_hailo_dfc.sh] ✅ Done."
echo ""
echo "Activate the venv with:"
echo "  source /root/hailo-workspace/venv/bin/activate"
echo ""
echo "Compile a HEF with:"
echo "  bash /mnt/c/WorkSpace/Project/neoscope/deploy_detect_onioncell/tools/compile_hef.sh \\"
echo "      /mnt/c/.../models/onnx/best.onnx \\"
echo "      /mnt/c/.../models/hef/best.hef \\"
echo "      /mnt/c/.../datasets/onioncell/calib_imgs"
