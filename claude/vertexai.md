# Claude Code & Google Vertex AI Setup

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

On **macOS** (browser opens automatically):

```bash
gcloud auth login
gcloud auth application-default login
```

On **remote server** (no browser):

```bash
gcloud auth login --no-browser
gcloud auth application-default login --no-browser
```

Then set project and quota:

```bash
gcloud config set project neu-research
gcloud auth application-default set-quota-project neu-research
```

### 4. Configure environment

**macOS (zsh):**

```bash
cat >> ~/.zshrc << 'EOF'
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=neu-research
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='claude-haiku-4-5-20251001'
EOF
source ~/.zshrc
```

**Linux server (bash):**

```bash
cat >> ~/.bashrc << 'EOF'
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=us-east5
export ANTHROPIC_VERTEX_PROJECT_ID=neu-research
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='claude-haiku-4-5-20251001'
EOF
source ~/.bashrc
```

### 5. Verify

```bash
claude
```

Type `/status` — you should see:

```
API provider: Google Vertex AI
GCP project: neu-research
Default region: us-east5
```

---

## Models

| Model | Best for |
|-------|----------|
| **Sonnet 4.6** (default) | Daily coding |
| **Opus 4.6** | Complex reasoning |
| **Haiku 4.5** | Quick, lightweight tasks |

Switch models in Claude Code with `/model`.

---
