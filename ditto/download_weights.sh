
#!/usr/bin/env bash
# Download Ditto TalkingHead model weights from HuggingFace
# Source: https://huggingface.co/digital-avatar/ditto-talkinghead
#
# Usage:
#   ./download_weights.sh [--checkpoints-dir <path>] [--skip-trt] [--skip-onnx] [--token <hf_token>]
#
# Default: downloads to ./ditto-talkinghead/checkpoints/
# Requires: huggingface_hub (pip install huggingface_hub[cli])

set -euo pipefail

REPO_ID="digital-avatar/ditto-talkinghead"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="${SCRIPT_DIR}/ditto-talkinghead/checkpoints"
SKIP_TRT=false
SKIP_ONNX=false
HF_TOKEN="${HF_TOKEN:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints-dir) CHECKPOINTS_DIR="$2"; shift 2 ;;
        --skip-trt) SKIP_TRT=true; shift ;;
        --skip-onnx) SKIP_ONNX=true; shift ;;
        --token) HF_TOKEN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Downloads Ditto TalkingHead model weights from HuggingFace."
            echo ""
            echo "Options:"
            echo "  --checkpoints-dir <path>  Target directory (default: ./ditto-talkinghead/checkpoints/)"
            echo "  --skip-trt               Skip TensorRT Ampere+ engines (~2GB)"
            echo "  --skip-onnx              Skip ONNX models (~2.4GB)"
            echo "  --token <hf_token>       HuggingFace token (or set HF_TOKEN env var)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Ditto TalkingHead Weight Downloader ==="
echo "Repository: ${REPO_ID}"
echo "Target:     ${CHECKPOINTS_DIR}"
echo ""

# Ensure huggingface-hub is available (v1.x uses 'hf' command)
if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface-hub..."
    pip install -q "huggingface_hub[hf_transfer]" 2>/dev/null \
        || pip install -q --break-system-packages "huggingface_hub[hf_transfer]" 2>/dev/null \
        || { echo "ERROR: Cannot install huggingface_hub. Create a venv or activate one first."; exit 1; }
fi

# Resolve CLI command (v1.x: 'hf', older: 'huggingface-cli')
if command -v hf &>/dev/null; then
    HF_CMD="hf"
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD="huggingface-cli"
else
    echo "ERROR: Neither 'hf' nor 'huggingface-cli' found after install."
    exit 1
fi

mkdir -p "${CHECKPOINTS_DIR}"

# Build exclude patterns for huggingface-cli download
EXCLUDE_ARGS=()
EXCLUDE_ARGS+=(--exclude ".gitattributes" --exclude ".gitignore" --exclude "LICENSE" --exclude "README.md")
if [ "$SKIP_TRT" = true ]; then
    echo "Skipping TensorRT engines (--skip-trt)"
    EXCLUDE_ARGS+=(--exclude "ditto_trt_Ampere_Plus/*")
fi
if [ "$SKIP_ONNX" = true ]; then
    echo "Skipping ONNX models (--skip-onnx)"
    EXCLUDE_ARGS+=(--exclude "ditto_onnx/*")
fi

# Token args
TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ]; then
    TOKEN_ARGS+=(--token "$HF_TOKEN")
fi

echo ""
echo "Downloading with hf_transfer (parallel, fast)..."

# Enable hf_transfer for maximum download speed
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use Python API for reliable downloads (CLI arg handling varies between versions)
python3 -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
ignore = ['*.md', 'LICENSE', '.gitattributes', '.gitignore']
skip_trt = '${SKIP_TRT}' == 'true'
skip_onnx = '${SKIP_ONNX}' == 'true'
if skip_trt:
    ignore.append('ditto_trt_Ampere_Plus/*')
if skip_onnx:
    ignore.append('ditto_onnx/*')
token = '${HF_TOKEN}' or None
print(f'Ignore patterns: {ignore}')
path = snapshot_download(
    repo_id='${REPO_ID}',
    local_dir='${CHECKPOINTS_DIR}',
    token=token,
    ignore_patterns=ignore,
)
print(f'Downloaded to: {path}')
"

echo ""
echo "=== Download complete ==="
echo ""

# Show what we got
echo "Checkpoint structure:"
find "${CHECKPOINTS_DIR}" -maxdepth 3 -type f -not -path '*/.cache/*' | sort | while read f; do
    size=$(du -h "$f" | cut -f1)
    rel="${f#${CHECKPOINTS_DIR}/}"
    echo "  ${rel}  (${size})"
done

echo ""
echo "Total size: $(du -sh "${CHECKPOINTS_DIR}" | cut -f1)"
