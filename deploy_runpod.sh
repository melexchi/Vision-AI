#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Vision-AI One-Click RunPod Deployment
# ═══════════════════════════════════════════════════════════════════
#
#  Paste this ENTIRE script into a RunPod terminal and it will:
#    1. Clone the repo
#    2. Install all dependencies
#    3. Download all model weights (~20GB)
#    4. Start all services
#    5. Print URLs to access the avatar
#
#  Requirements:
#    - RunPod GPU pod with A100 40GB+ (or A40 48GB)
#    - Template: any CUDA 12.x / PyTorch template
#    - Disk: 100GB minimum
#    - Exposed ports: 8080, 8181, 8282
#
#  Time: ~15-25 minutes (mostly downloading models)
#
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Vision-AI Avatar Stack — RunPod Deployer            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Step 0: System check ─────────────────────────────────────────

echo "[0/7] Checking system..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU pod?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME ($GPU_MEM)"
echo "  Python: $(python3 --version 2>&1)"
echo "  Disk free: $(df -h /workspace | tail -1 | awk '{print $4}')"
echo ""

# ─── Step 1: Clone repo ───────────────────────────────────────────

echo "[1/7] Cloning Vision-AI repository..."
cd /workspace
if [ -d "Vision-AI" ]; then
    echo "  Already exists, pulling latest..."
    cd Vision-AI && git pull && cd ..
else
    git clone https://github.com/melexchi/Vision-AI.git
fi
cd Vision-AI
echo "  Done."
echo ""

# ─── Step 2: Install system deps ──────────────────────────────────

echo "[2/7] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1 || true
echo "  Done."
echo ""

# ─── Step 3: Setup Ditto (main avatar service) ────────────────────

echo "[3/7] Setting up Ditto (lip-sync engine)..."
echo "  This downloads ~4GB of model weights and builds TensorRT engines."
echo "  ETA: 10-15 minutes on first run."
echo ""
cd ditto
bash setup.sh 2>&1 | tail -20
cd ..
echo ""
echo "  Ditto setup complete."
echo ""

# ─── Step 4: Setup Chatterbox (TTS) ───────────────────────────────

echo "[4/7] Setting up Chatterbox (text-to-speech)..."
cd chatterbox
pip install -q -r requirements.txt 2>&1 | tail -3
cd ..
echo "  Done."
echo ""

# ─── Step 5: Setup SmolVLM (vision) ───────────────────────────────

echo "[5/7] Setting up SmolVLM (camera vision)..."
cd smolvlm
pip install -q -r requirements.txt 2>&1 | tail -3
cd ..
echo "  Done."
echo ""

# ─── Step 6: Create workspace directories ─────────────────────────

echo "[6/7] Creating workspace directories..."
mkdir -p /workspace/avatar_cache /workspace/avatar_clips /workspace/avatar_images
echo "  Done."
echo ""

# ─── Step 7: Start all services ───────────────────────────────────

echo "[7/7] Starting all services..."
echo ""

# Kill any existing instances
pkill -f "uvicorn api_server:app" 2>/dev/null || true
pkill -f "uvicorn ditto_api:app" 2>/dev/null || true
pkill -f "uvicorn smolvlm_server:app" 2>/dev/null || true
sleep 2

# Start Chatterbox TTS (port 8080)
cd /workspace/Vision-AI/chatterbox
nohup python -m uvicorn api_server:app --host 0.0.0.0 --port 8080 > /workspace/chatterbox.log 2>&1 &
CHATTERBOX_PID=$!
echo "  Chatterbox TTS started (PID $CHATTERBOX_PID, port 8080)"

# Start SmolVLM (port 8282)
cd /workspace/Vision-AI/smolvlm
nohup python -m uvicorn smolvlm_server:app --host 0.0.0.0 --port 8282 > /workspace/smolvlm.log 2>&1 &
SMOLVLM_PID=$!
echo "  SmolVLM started (PID $SMOLVLM_PID, port 8282)"

# Start Ditto (port 8181) — this one takes ~30s to load models
cd /workspace/Vision-AI/ditto
export DITTO_BACKEND=${DITTO_BACKEND:-onnx}
export TTS_URL=http://localhost:8080/api/tts
export DITTO_DEBUG_ENDPOINTS=1
nohup uv run uvicorn ditto_api:app --host 0.0.0.0 --port 8181 > /workspace/ditto.log 2>&1 &
DITTO_PID=$!
echo "  Ditto Avatar started (PID $DITTO_PID, port 8181)"
echo ""

# Wait for Ditto to load
echo "  Waiting for Ditto to load models (30-60s)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8181/health > /dev/null 2>&1; then
        echo "  Ditto is ready!"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# ─── Step 8: Register a demo avatar ───────────────────────────────

echo ""
echo "Registering demo avatar from example image..."
cd /workspace/Vision-AI

# Use the first reference image
DEMO_IMAGE="skyreels/assets/ref_images/1.png"
if [ -f "$DEMO_IMAGE" ]; then
    B64=$(base64 -w0 "$DEMO_IMAGE")
    RESULT=$(curl -s -X POST http://localhost:8181/register \
        -H "Content-Type: application/json" \
        -d "{\"avatar_id\": \"demo\", \"image_base64\": \"$B64\", \"prerender_clips\": false}")
    echo "  Registration result: $RESULT"
else
    echo "  Demo image not found, skipping auto-registration."
    echo "  You can register manually via the API."
fi

# ─── Done! Print access info ──────────────────────────────────────

# Get the pod's public URL
POD_ID=${RUNPOD_POD_ID:-$(hostname)}
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "<your-pod-ip>")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ALL SERVICES RUNNING                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Ditto Avatar API:  http://${PUBLIC_IP}:8181                ║"
echo "║  Chatterbox TTS:    http://${PUBLIC_IP}:8080                ║"
echo "║  SmolVLM Vision:    http://${PUBLIC_IP}:8282                ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  OPEN IN YOUR BROWSER:                                       ║"
echo "║                                                              ║"
echo "║  Interactive API Docs:                                       ║"
echo "║    http://${PUBLIC_IP}:8181/docs                             ║"
echo "║                                                              ║"
echo "║  Avatar Test Viewer (see the avatar talking!):               ║"
echo "║    http://${PUBLIC_IP}:8181/test_viewer                     ║"
echo "║                                                              ║"
echo "║  Health Check:                                               ║"
echo "║    http://${PUBLIC_IP}:8181/health                          ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  QUICK TEST COMMANDS:                                        ║"
echo "║                                                              ║"
echo "║  # See all registered avatars                                ║"
echo "║  curl http://localhost:8181/avatars                          ║"
echo "║                                                              ║"
echo "║  # Generate a talking video (if avatar 'demo' registered)   ║"
echo "║  # First generate speech:                                    ║"
echo "║  curl -X POST http://localhost:8080/api/tts \\               ║"
echo "║    -F 'text=Hello! I am your AI avatar.' \\                  ║"
echo "║    -F 'language=en' -o speech.wav                            ║"
echo "║                                                              ║"
echo "║  # Then make the avatar say it:                              ║"
echo "║  curl -X POST http://localhost:8181/generate \\              ║"
echo "║    -H 'Content-Type: application/json' \\                    ║"
echo "║    -d '{\"avatar_id\":\"demo\",                              ║"
echo "║         \"audio_base64\":\"'$(base64 -w0 speech.wav)'\"}' \\ ║"
echo "║    -o avatar_talking.mp4                                     ║"
echo "║                                                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  LOGS:                                                       ║"
echo "║    tail -f /workspace/ditto.log                              ║"
echo "║    tail -f /workspace/chatterbox.log                         ║"
echo "║    tail -f /workspace/smolvlm.log                            ║"
echo "║                                                              ║"
echo "║  STOP ALL:                                                   ║"
echo "║    pkill -f uvicorn                                          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
