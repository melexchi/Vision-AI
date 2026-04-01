# SmolVLM — Camera Vision Service

## Purpose

Lightweight vision-language model service for understanding camera frames from users. Uses HuggingFace SmolVLM2-2.2B-Instruct to answer questions about images. Serves on port 8282 with REST endpoints for image queries via base64, URL, or file upload.

## Key Files

- `smolvlm_server.py` — Complete FastAPI server (195 lines). Contains:
  - `POST /v1/query` — Query with image (base64 or URL) + text prompt → text response
  - `POST /v1/query/upload` — Query with uploaded image file + prompt
  - `POST /v1/unload` — Free GPU VRAM (deletes model, calls `torch.cuda.empty_cache()`)
  - `GET /health` — Service status
  - `get_model()` — Lazy loader for SmolVLM2-2.2B-Instruct
- `pyproject.toml` — Project config
- `requirements.txt` — Dependencies: `transformers>=4.49.0`, `torch`, `fastapi`, `uvicorn`, `Pillow`, `httpx`

## Data Flow

```
POST /v1/query {image_base64 or image_url, prompt}
  → get_model() (lazy load SmolVLM2-2.2B-Instruct)
  → processor(text=prompt, images=[image]) → input tensors
  → model.generate(max_new_tokens=512) → output tokens
  → decode → {"success": true, "response": "..."}
```

## Dependencies

- **Depends on:** HuggingFace `transformers` (auto-downloads SmolVLM2-2.2B-Instruct on first use), PyTorch, CUDA GPU (16GB+ VRAM)
- **Depended on by:** Client applications (REST API)

## Conventions

- Unlike Ditto/Chatterbox, uses `JSONResponse({"success": bool, ...})` pattern instead of `HTTPException`
- Model and processor are global variables, lazy-loaded via `get_model()`
- `/v1/` prefix on API paths (only service using versioned paths)
- Image fetched via async `httpx` when URL provided
- `/v1/unload` allows explicit VRAM release when not in use

## Common Commands

```bash
pip install -r requirements.txt
uvicorn smolvlm_server:app --host 0.0.0.0 --port 8282
```
