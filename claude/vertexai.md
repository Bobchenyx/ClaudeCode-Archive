# Claude Code + Google Vertex AI Setup

## Prerequisites

- macOS 10.15+ or Linux
- Node.js 18+
- Vertex AI User role on the `neu-research` GCP project (ask admin if needed)

## Setup

### 1. Install gcloud CLI

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### 3. Authenticate

```bash
gcloud auth login
# gcloud auth login --no-browser
gcloud auth application-default login
gcloud config set project neu-research
gcloud auth application-default set-quota-project neu-research
```

### 4. Configure environment

Add to `~/.zshrc` (or `~/.bashrc`):

```bash
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=neu-research
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'
```

Then reload:

```bash
source ~/.zshrc
```

### 5. Verify

```bash
claude
```

Type `/status` — you should see `API provider: Google Vertex AI`.

## Models

| Model | Best for |
|-------|----------|
| **Sonnet 4.6** (default) | Daily coding |
| **Opus 4.6** | Complex reasoning |
| **Haiku 4.5** | Quick, lightweight tasks |

Switch models in Claude Code with `/model`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `gcloud: command not found` | `exec -l $SHELL` or reinstall gcloud |
| Model not available | Enable it in [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) |
| Permission denied | Ask admin: `gcloud projects add-iam-policy-binding neu-research --member="user:EMAIL" --role="roles/aiplatform.user"` |
| Wrong API provider | Check `echo $CLAUDE_CODE_USE_VERTEX` returns `1`, then `source ~/.zshrc` |
