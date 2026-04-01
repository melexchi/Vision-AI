# SSH Config for Claude Code

Add to your `~/.ssh/config` on the machine running Claude Code.
Replace `[FILL IN]` with actual values.

```
Host vision-ai-server
    HostName [FILL IN server IP]
    User claude-server
    IdentityFile ~/.ssh/claude-server
```

## Server User Setup (run once as root)

```bash
adduser claude-server --disabled-password
usermod -aG docker claude-server
usermod -aG www-data claude-server
```

## Generate Claude's SSH key

```bash
ssh-keygen -t ed25519 -f ~/.ssh/claude-server -C "claude-code-access"
ssh-copy-id -i ~/.ssh/claude-server.pub claude-server@[server-ip]
```

## Notes

- Current deployment target is **RunPod GPU cloud** (managed via RunPod portal, not direct SSH)
- If migrating to a persistent server (Hetzner, AWS, etc.), fill in the config above
- Each developer uses their OWN SSH key — never share keys
