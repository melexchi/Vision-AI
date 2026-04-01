#!/usr/bin/env python3
"""
Chatterbox TTS WebSocket API Server
Generate high-quality speech with voice cloning and emotion control
"""

import os
import sys
import json
import base64
import asyncio
import threading
import tempfile
import time
import uuid
import io
from pathlib import Path
from typing import Optional

import torch
import torchaudio as ta
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chatterbox TTS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (lazy loaded)
multilingual_model = None
turbo_model = None
_model_lock = threading.Lock()

TEMP_DIR = Path(tempfile.gettempdir()) / "chatterbox_api"
TEMP_DIR.mkdir(exist_ok=True)


def get_multilingual_model():
    """Lazy load Chatterbox Multilingual model (thread-safe)."""
    global multilingual_model
    if multilingual_model is None:
        with _model_lock:
            if multilingual_model is None:  # double-check after acquiring lock
                print("Loading Chatterbox Multilingual model...")
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
                print("Chatterbox Multilingual model loaded!")
    return multilingual_model


def get_turbo_model():
    """Lazy load Chatterbox Turbo model (thread-safe)."""
    global turbo_model
    if turbo_model is None:
        with _model_lock:
            if turbo_model is None:  # double-check after acquiring lock
                print("Loading Chatterbox Turbo model...")
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                turbo_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
                print("Chatterbox Turbo model loaded!")
    return turbo_model


@app.on_event("startup")
async def startup():
    """Pre-load model on startup"""
    print("Starting Chatterbox TTS API server...")
    print("Loading Chatterbox Multilingual model...")
    get_multilingual_model()
    print("Loading Chatterbox Turbo model...")
    get_turbo_model()
    print("All models loaded successfully!")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "multilingual_loaded": multilingual_model is not None,
        "turbo_loaded": turbo_model is not None,
    }


@app.post("/api/tts")
async def generate_tts(
    text: str = Form(...),
    language: str = Form("en"),
    voice_prompt: Optional[UploadFile] = File(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    model_type: str = Form("multilingual"),  # "multilingual" or "turbo"
):
    """
    REST API endpoint for TTS generation
    Returns audio as WAV file
    """
    request_id = str(uuid.uuid4())[:8]

    try:
        # Save voice prompt if provided
        voice_prompt_path = None
        if voice_prompt:
            voice_prompt_path = TEMP_DIR / f"{request_id}_voice.wav"
            with open(voice_prompt_path, "wb") as f:
                f.write(await voice_prompt.read())

        # Generate audio
        if model_type == "turbo":
            model = get_turbo_model()
            if voice_prompt_path:
                wav = model.generate(text, audio_prompt_path=str(voice_prompt_path))
            else:
                raise HTTPException(status_code=400, detail="Turbo model requires voice_prompt")
        else:
            model = get_multilingual_model()
            if voice_prompt_path:
                wav = model.generate(
                    text,
                    language_id=language,
                    audio_prompt_path=str(voice_prompt_path),
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
            else:
                wav = model.generate(
                    text,
                    language_id=language,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)

        # Cleanup
        if voice_prompt_path and voice_prompt_path.exists():
            voice_prompt_path.unlink()

        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=tts_{request_id}.wav"}
        )

    except Exception as e:
        print(f"[error] TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for TTS generation (Turbo model with paralinguistic tags)

    Protocol:
    1. Client sends: {"type": "init", "voice_prompt": "<base64 wav>" (REQUIRED for Turbo)}
    2. Server responds: {"type": "ready", "session_id": "..."}
    3. Client sends: {"type": "tts", "text": "...", "temperature": 0.8, "top_p": 0.95}
       - Supports tags: [laugh], [chuckle], [cough], [sigh], [gasp], [groan], [sniff], [clear throat]
    4. Server responds: {"type": "processing"}
    5. Server responds: {"type": "audio", "data": "<base64 wav>", "duration": X.X}
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    session_dir = TEMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    voice_prompt_path: Optional[Path] = None
    model = get_turbo_model()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type")

            if msg_type == "init":
                # Voice prompt is REQUIRED for Turbo model
                if "voice_prompt" not in msg:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Turbo model requires voice_prompt for voice cloning"
                    })
                    continue

                voice_b64 = msg["voice_prompt"]
                if len(voice_b64) > 50 * 1024 * 1024:
                    await websocket.send_json({"type": "error", "message": "Voice prompt too large (max 50MB)"})
                    continue
                voice_data = base64.b64decode(voice_b64)
                voice_prompt_path = session_dir / "voice_prompt.wav"
                with open(voice_prompt_path, "wb") as f:
                    f.write(voice_data)

                # Pre-load conditionals for faster generation
                model.prepare_conditionals(str(voice_prompt_path))

                await websocket.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "message": "Turbo TTS ready. Supports [laugh], [chuckle], [cough], etc.",
                    "voice_cloning": True,
                    "model": "turbo"
                })

            elif msg_type == "tts":
                text = msg.get("text", "")
                if not text:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No text provided"
                    })
                    continue

                if voice_prompt_path is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Please send init with voice_prompt first"
                    })
                    continue

                # Turbo model parameters
                temperature = msg.get("temperature", 0.8)
                top_p = msg.get("top_p", 0.95)
                top_k = msg.get("top_k", 1000)
                repetition_penalty = msg.get("repetition_penalty", 1.2)

                await websocket.send_json({"type": "processing"})

                # Generate in thread pool
                loop = asyncio.get_event_loop()
                start_time = time.time()

                def generate():
                    return model.generate(
                        text,
                        audio_prompt_path=str(voice_prompt_path),
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )

                wav = await loop.run_in_executor(None, generate)
                process_time = time.time() - start_time

                # Convert to bytes
                buffer = io.BytesIO()
                ta.save(buffer, wav, model.sr, format="wav")
                buffer.seek(0)
                audio_bytes = buffer.read()

                # Calculate duration
                duration = wav.shape[1] / model.sr

                # Send as base64
                audio_b64 = base64.b64encode(audio_bytes).decode()

                await websocket.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "duration": round(duration, 2),
                    "process_time": round(process_time, 2),
                    "size_kb": round(len(audio_bytes) / 1024, 1),
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "close":
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"Session {session_id} error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass  # Client already disconnected, nothing to send to
    finally:
        # Cleanup session files
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    print(f"Starting Chatterbox TTS API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
