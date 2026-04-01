# Chatterbox — Neural Text-to-Speech Service

## Purpose

Neural TTS service with voice cloning support, serving on port 8080. Provides both REST and WebSocket endpoints for speech synthesis. Supports two model variants: Multilingual (higher quality, more languages) and Turbo (faster, English-focused). Uses Resemble AI's Chatterbox library.

## Key Files

- `api_server.py` — Complete FastAPI server (304 lines). Contains:
  - `POST /api/tts` — REST endpoint: text + optional voice prompt → WAV file
  - `WS /ws` — WebSocket endpoint: session-based streaming TTS with voice cloning
  - `GET /health` — Model load status
  - Lazy model loading (`get_multilingual_model()`, `get_turbo_model()`)
  - WebSocket session management with temp directory cleanup
- `requirements.txt` — Dependencies: `chatterbox-tts`, `torch`, `torchaudio`, `fastapi`, `uvicorn`

## Data Flow

```
REST: POST /api/tts {text, language, voice_prompt} → model.generate() → WAV bytes → Response
WS:   "init" {voice_prompt: base64} → prepare_conditionals()
      "tts" {text} → model.generate() → base64 WAV → {"type": "audio", "data": ...}
```

WebSocket session lifecycle: connect → init (voice) → tts (repeated) → close → cleanup temp dir.

## Dependencies

- **Depends on:** `chatterbox-tts` pip package (Resemble AI), PyTorch, torchaudio, CUDA GPU
- **Depended on by:** Ditto (`/generate_from_text` calls `TTS_URL` via HTTP)

## Conventions

- Models are global variables, lazy-loaded on first request
- WebSocket messages are JSON with `{"type": "...", ...}` envelope
- Paralinguistic tags supported: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[groan]`, `[sniff]`, `[clear throat]`
- Voice prompt stored as temp WAV file per session, cleaned up on disconnect
- TTS inference runs in `run_in_executor()` to avoid blocking the event loop

## Common Commands

```bash
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8080
```
