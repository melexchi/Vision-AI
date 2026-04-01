# Access & Credentials Guide

How to get access to everything needed for this project.

## GitHub Repository

- Repo: https://github.com/melexchi/Vision-AI
- Upstream: https://github.com/ageofai-llc/Avatar-AI
- Access: Ask [FILL IN tech lead name] to add you as collaborator
- Setup SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- After access granted, run:

```bash
git clone https://github.com/melexchi/Vision-AI.git
git config user.name "Your Name"
git config user.email "your@email.com"
```

## Environment Variables

- No `.env` file exists yet — all config uses `os.environ.get()` with hardcoded defaults
- Key variables (see `docs/ARCHITECTURE.md` Section 10 for full list):
  - `DITTO_BACKEND` — `trt` or `onnx` (default: `onnx`)
  - `AVATAR_CACHE_DIR` — Avatar feature cache (default: `/workspace/avatar_cache`)
  - `AVATAR_CLIPS_DIR` — Pre-rendered clips (default: `/workspace/avatar_clips`)
  - `TTS_URL` — Chatterbox endpoint (default: `http://localhost:8282/tts/stream`)
  - `HF_TOKEN` — HuggingFace token for private model downloads
- NEVER share credentials via email, Slack, or any unencrypted channel

## Deployment / Server Access

- Deployment platform: **RunPod** (GPU cloud)
- Dashboard: https://www.runpod.io/console/pods
- Access: Request RunPod account from tech lead
- Docker image: Built from `ditto/Dockerfile`

### Server

| Server | Host/IP | SSH User | Purpose | Who Can Access |
|--------|---------|----------|---------|----------------|
| RunPod Pod | [FILL IN — dynamic per pod] | root | GPU inference (Ditto + SkyReels) | [FILL IN] |
| [FILL IN] | [FILL IN] | claude-server | LiveKit Agent server (Hetzner) | [FILL IN] |

### SSH Setup for Claude Code

See `docs/SSH_CONFIG.md` for full setup instructions.

1. Generate SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/claude-server -C "claude-code-access"`
2. Send your PUBLIC key (`~/.ssh/claude-server.pub`) to tech lead
3. Tech lead adds it to the server's authorized_keys
4. Test connection: `ssh vision-ai-server "echo connected"`

## Third-Party Services (API Keys in .env)

| Service | Purpose | How to Get Access |
|---------|---------|-------------------|
| HuggingFace | Model weight downloads | Create account at huggingface.co, generate token |
| RunPod | GPU cloud deployment | Request account from tech lead |
| LiveKit | WebRTC streaming server | [FILL IN — self-hosted or cloud?] |
| Google Chirp3 HD | Cloud TTS fallback | [FILL IN — API key from Google Cloud] |

## Database Access

No SQL database. All state is in-memory dicts + pickle files on disk.
- Avatar cache: `{AVATAR_CACHE_DIR}/*.pkl`
- Clips: `{AVATAR_CLIPS_DIR}/*.mp4`
- No migrations, no backup commands needed for database

## Who to Contact

- Tech Lead: [FILL IN name + contact]
- DevOps / Server: [FILL IN name + contact]
- Project Manager: [FILL IN name + contact]

## Security Reminders

- Each developer uses their OWN SSH key — never share keys
- Rotate any credential immediately if you suspect it's compromised
- Never commit .env, private keys, or secrets to git
- Use a password manager (1Password, Bitwarden) for shared team credentials
- When migrating to production server, generate ALL new credentials — never reuse dev credentials
