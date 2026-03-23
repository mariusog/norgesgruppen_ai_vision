# Claude Code via Vertex AI — Devcontainer Setup

Drop-in config to use Claude Code (Opus 4.6) through your GCP lab account.
No Anthropic API key needed — bills through GCP.

## Prerequisites

1. GCP project with Vertex AI API enabled
2. Claude Opus enabled in Model Garden:
   https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-opus-4?project=YOUR_GCP_PROJECT_ID
3. `gcloud auth application-default login` (prompted on first use)

## Option A: Add to your devcontainer.json

Add these two blocks to your existing `.devcontainer/devcontainer.json`:

```jsonc
{
  // Add Claude Code feature (if not already present)
  "features": {
    "ghcr.io/anthropics/devcontainer-features/claude-code:1": {}
  },

  // Route Claude Code through Vertex AI
  "containerEnv": {
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLOUD_ML_REGION": "us-east5",
    "ANTHROPIC_VERTEX_PROJECT_ID": "YOUR_GCP_PROJECT_ID"
  }
}
```

Then rebuild the devcontainer.

## Option B: Apply the patch file

From your repo root:

```bash
git apply patches/claude-vertex.patch
```

## Option C: Just set env vars (any machine)

```bash
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=YOUR_GCP_PROJECT_ID
```

## Recommended .claude/settings.json

For maximum capability (free credits = go all out):

```json
{
  "model": "opus[1m]",
  "effortLevel": "max"
}
```

- `opus[1m]` — Opus 4.6 with 1M token context window (sees entire codebase)
- `effortLevel: "max"` — unlimited reasoning budget, deepest thinking (Opus only)
- Prompt caching is automatic, no config needed

## Verify

After setup, start Claude Code and check the model shows as Opus 4.6 via Vertex:

```bash
claude
```
