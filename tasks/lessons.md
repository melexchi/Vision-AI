# Lessons Learned

Rules added here prevent repeated mistakes. Each rule was born from an actual error.

## Code Patterns

(none yet)

## Frontend Design

- All AI models default to generic templates (Inter font, purple gradients, cards-in-cards). Always challenge the first design output with /critique before accepting.
- Animations should have purpose. Never animate just because you can. Every motion must communicate state change, guide attention, or provide feedback.
- Dark mode is not "invert colors". It requires separate consideration for contrast, shadows, and surface hierarchy.

## Common Pitfalls

(none yet)

## Testing

(none yet)

## Bulk Operations

- NEVER run sed -i on files without checking the file list first
- NEVER run find -exec on directories without excluding: .claude/skills/, node_modules/, .git/, dist/, build/, checkpoints/, pretrained_models/
- ALWAYS show the exact file list before any operation touching 5+ files
- ALWAYS exclude lock files (package-lock.json, bun.lock, yarn.lock, uv.lock) from bulk modifications
