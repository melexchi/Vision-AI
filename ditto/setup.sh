#!/usr/bin/env bash
# Idempotent setup script for Ditto Avatar Server on RunPod
#
# Usage:
#   cd /workspace/ditto && bash setup.sh
#
# What it does:
#   1. Installs uv if missing
#   2. uv sync (all deps) + TRT from NVIDIA index
#   3. Downloads model weights (idempotent)
#   4. Builds TRT engines from ONNX
#   5. Builds GridSample3D TRT plugin
#   6. Compiles Cython blend extension
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINTS_DIR="$SCRIPT_DIR/ditto-talkinghead/checkpoints"
ONNX_DIR="$CHECKPOINTS_DIR/ditto_onnx"
TRT_DIR="$CHECKPOINTS_DIR/ditto_trt_Ampere_Plus"
PLUGIN_SO="$ONNX_DIR/libgrid_sample_3d_plugin.so"

echo "=== Ditto Avatar Server Setup ==="
echo "Directory: $SCRIPT_DIR"
echo ""

# ── 1. Install uv ──────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "▸ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ uv $(uv --version)"

# ── 2. Install Python deps ─────────────────────────────────────
echo ""
echo "▸ Running uv sync..."
uv sync
echo "✓ Python deps installed"

# ── 3. Download model weights ──────────────────────────────────
if [ ! -d "$CHECKPOINTS_DIR" ] || [ -z "$(ls -A "$CHECKPOINTS_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "▸ Downloading model weights..."
    bash "$SCRIPT_DIR/download_weights.sh"
else
    echo "✓ Model weights already present at $CHECKPOINTS_DIR"
fi

# ── 4. Build TRT engines from ONNX ────────────────────────────
if [ -d "$ONNX_DIR" ] && [ ! -d "$TRT_DIR" ]; then
    echo ""
    echo "▸ Building TRT engines from ONNX models..."
    uv run "$SCRIPT_DIR/ditto-talkinghead/scripts/cvt_onnx_to_trt.py" \
        --onnx-dir "$ONNX_DIR" \
        --trt-dir "$TRT_DIR"
    echo "✓ TRT engines built"
elif [ -d "$TRT_DIR" ]; then
    echo "✓ TRT engines already present at $TRT_DIR"
else
    echo "⚠ No ONNX dir found — skipping TRT build (run download_weights.sh first)"
fi

# ── 5. Build GridSample3D TRT plugin ──────────────────────────
if [ ! -f "$PLUGIN_SO" ]; then
    echo ""
    echo "▸ Building GridSample3D TRT plugin..."
    PLUGIN_SRC_DIR="$ONNX_DIR/GridSample3D_plugin_src"
    if [ -d "$PLUGIN_SRC_DIR" ]; then
        pushd "$PLUGIN_SRC_DIR" >/dev/null
        make -j"$(nproc)"
        cp libgrid_sample_3d_plugin.so "$PLUGIN_SO"
        popd >/dev/null
        echo "✓ GridSample3D plugin built"
    else
        echo "⚠ Plugin source not found at $PLUGIN_SRC_DIR — skipping"
    fi
else
    echo "✓ GridSample3D plugin already built"
fi

# ── 6. Compile Cython blend extension ─────────────────────────
BLEND_DIR="$SCRIPT_DIR/ditto-talkinghead/core/utils/blend"
BLEND_SO=$(find "$BLEND_DIR" -name "blend*.so" 2>/dev/null | head -1)
if [ -z "$BLEND_SO" ]; then
    echo ""
    echo "▸ Compiling Cython blend extension..."
    uv run python -c "
import pyximport
pyximport.install()
import sys
sys.path.insert(0, '$BLEND_DIR')
import blend
print('Blend extension compiled successfully')
" 2>/dev/null || echo "⚠ Blend compilation failed (non-critical, will use fallback)"
else
    echo "✓ Cython blend extension already compiled"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo "  Python deps:    ✓ (uv sync)"
echo "  Weights:        $([ -d "$CHECKPOINTS_DIR" ] && echo '✓' || echo '✗')"
echo "  TRT engines:    $([ -d "$TRT_DIR" ] && echo '✓' || echo '✗ (need ONNX models)')"
echo "  GridSample3D:   $([ -f "$PLUGIN_SO" ] && echo '✓' || echo '✗')"
echo "  Blend ext:      $([ -n "$(find "$BLEND_DIR" -name 'blend*.so' 2>/dev/null)" ] && echo '✓' || echo '✗')"
echo ""
echo "Start the server:"
echo "  uv run ditto_api.py"
