#!/bin/bash
# Setup script for SkyReels-A1 on RunPod (A100 80GB)
# Downloads pretrained models and creates a Python venv with all dependencies.
# Idempotent: skips steps that are already done. Creates .setup_done on completion.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$SCRIPT_DIR/pretrained_models"
MARKER="$SCRIPT_DIR/.setup_done"

if [ -f "$MARKER" ]; then
    echo "[setup] Already complete (.setup_done exists). Remove it to re-run."
    exit 0
fi

echo "============================================"
echo " SkyReels-A1 Setup (one-time, ~15-20 min)"
echo "============================================"

# ---------- 1. Python venv ----------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[setup] Creating Python 3.12 venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "[setup] Venv already exists."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ---------- 2. PyTorch (CUDA 12.1 — compatible with RunPod CUDA 12.x) ----------
echo "[setup] Installing PyTorch..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# ---------- 3. Requirements ----------
echo "[setup] Installing requirements.txt..."
# chumpy has a broken build that fails in isolation — install it first with --no-build-isolation
pip install chumpy==0.70 --no-build-isolation 2>/dev/null || pip install chumpy --no-build-isolation
# Exclude pytorch3d (step 4) and torch/torchvision/torchaudio (step 2) from requirements
grep -v -E 'pytorch3d|^torch' requirements.txt | pip install -r /dev/stdin

# ---------- 4. pytorch3d (build from source for CUDA compat) ----------
if ! python -c "import pytorch3d" 2>/dev/null; then
    echo "[setup] Installing pytorch3d..."
    # Try prebuilt wheel first (much faster), fall back to source build with CUDA
    CUDA_HOME=/usr/local/cuda pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
else
    echo "[setup] pytorch3d already installed."
fi

# ---------- 5. gdown for Google Drive downloads ----------
pip install -q "huggingface_hub[cli]" gdown

# ---------- 6. Download pretrained models ----------
mkdir -p "$MODELS_DIR"

# 6a. SkyReels-A1-5B main model (includes FLAME, mediapipe, smirk in extra_models/)
SKYREELS_MODEL_DIR="$MODELS_DIR/SkyReels-A1-5B"
if [ ! -d "$SKYREELS_MODEL_DIR/transformer" ]; then
    echo "[setup] Downloading SkyReels-A1-5B model (~15GB)..."
    huggingface-cli download Skywork/SkyReels-A1 \
        --local-dir "$SKYREELS_MODEL_DIR" \
        --local-dir-use-symlinks False
else
    echo "[setup] SkyReels-A1-5B already downloaded."
fi

# 6b. Symlink FLAME, mediapipe, smirk from the HF download to pretrained_models/ root
EXTRA="$SKYREELS_MODEL_DIR/extra_models"
for subdir in FLAME mediapipe smirk; do
    target="$MODELS_DIR/$subdir"
    if [ ! -e "$target" ] && [ -d "$EXTRA/$subdir" ]; then
        echo "[setup] Linking $subdir..."
        ln -sf "$EXTRA/$subdir" "$target"
    fi
done

# 6c. DiffPoseTalk weights (not in HF repo — separate downloads required)
DPT_DIR="$MODELS_DIR/diffposetalk"
mkdir -p "$DPT_DIR"

# iter_0110000.pt — DiffPoseTalk checkpoint from their HF repo
if [ ! -f "$DPT_DIR/iter_0110000.pt" ]; then
    echo "[setup] Downloading DiffPoseTalk checkpoint..."
    huggingface-cli download multimodalart/diffposetalk \
        --local-dir "$MODELS_DIR/_tmp_dpt" \
        --local-dir-use-symlinks False \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
    # copy checkpoint
    find "$MODELS_DIR/_tmp_dpt" -name "iter_0110000.pt" -exec cp {} "$DPT_DIR/" \;
    # copy stats_train.npz
    find "$MODELS_DIR/_tmp_dpt" -name "stats_train.npz" -exec cp {} "$DPT_DIR/" \;
    # copy style dir if present
    [ -d "$MODELS_DIR/_tmp_dpt/style" ] && cp -r "$MODELS_DIR/_tmp_dpt/style" "$DPT_DIR/"
    rm -rf "$MODELS_DIR/_tmp_dpt"
fi

# 6d. FILM frame interpolation model (from GitHub releases)
FILM_DIR="$MODELS_DIR/film_net"
if [ ! -f "$FILM_DIR/film_net_fp16.pt" ]; then
    echo "[setup] Downloading FILM interpolation model..."
    mkdir -p "$FILM_DIR"
    wget -q -O "$FILM_DIR/film_net_fp16.pt" \
        "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp16.pt"
fi

# ---------- 7. Marker ----------
echo "[setup] All done!"
date > "$MARKER"
echo "Setup completed at $(cat "$MARKER")"
