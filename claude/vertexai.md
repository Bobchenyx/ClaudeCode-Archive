# Claude Code + Google Vertex AI Setup Guide

> EmbodyX Team 内部文档 | Last updated: March 2026

## Overview

本指南帮助团队成员配置 Claude Code，通过 Google Cloud Vertex AI 使用 Claude 模型，费用从 GCP Partner Credit 中扣除。

## 模型选择

| Model | 速度 | 成本 | 适用场景 |
|-------|------|------|----------|
| **Haiku 4.5** | 最快 | 最低 | 快速问答、代码补全、简单任务 |
| **Sonnet 4.6** ⭐ | 快 | 中等 | **日常编码（推荐默认）** |
| **Opus 4.6** | 最慢 | 最高 | 复杂架构设计、深度推理 |

> 💡 建议日常使用 Sonnet 4.6，遇到复杂任务时在 Claude Code 中用 `/model` 切换到 Opus 4.6。

---

## Prerequisites

- macOS 10.15+ 或 Linux
- Node.js 18+
- 你的 `@embodyx.io` 账号已被授予 `neu-research` 项目的 **Vertex AI User** 权限

> 如果没有权限，请联系项目管理员执行：
> ```bash
> gcloud projects add-iam-policy-binding neu-research \
>   --member="user:你的邮箱@embodyx.io" \
>   --role="roles/aiplatform.user"
> ```

---

## Step 1: 安装 gcloud CLI

```bash
curl https://sdk.cloud.google.com | bash
```

安装过程中所有选项按 **回车** 使用默认值。如果提示输入密码，是在安装 Python 3.13，输入你的 Mac 登录密码即可。

安装完成后重载 shell：

```bash
exec -l $SHELL
```

验证安装：

```bash
gcloud --version
```

---

## Step 2: 安装 Claude Code

如果还没装过 Claude Code：

```bash
npm install -g @anthropic-ai/claude-code
```

验证：

```bash
claude --version
```

---

## Step 3: 认证 Google Cloud

使用你的 `@embodyx.io` 账号登录（浏览器会自动弹出）：

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project neu-research
gcloud auth application-default set-quota-project neu-research
```

---

## Step 4: 配置环境变量

将以下内容追加到 `~/.zshrc`（如果用 Bash 则追加到 `~/.bashrc`）：

```bash
echo 'export CLAUDE_CODE_USE_VERTEX=1' >> ~/.zshrc
echo 'export CLOUD_ML_REGION=us-east5' >> ~/.zshrc
echo 'export ANTHROPIC_VERTEX_PROJECT_ID=neu-research' >> ~/.zshrc
echo "export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'" >> ~/.zshrc
echo "export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'" >> ~/.zshrc
```

重载配置：

```bash
source ~/.zshrc
```

---

## Step 5: 启动 Claude Code

```bash
cd ~/your-project
claude
```

输入 `/status` 验证，应看到：

```
API provider: Google Vertex AI
GCP project: neu-research
Default region: us-east5
Model: Default (claude-sonnet-4-6)
```

✅ 配置完成！

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 切换模型 | `/model` |
| 查看状态 | `/status` |
| 退出 Claude Code | `/exit` |
| 重新认证 GCP | `gcloud auth application-default login` |
| 查看当前项目 | `gcloud config get-value project` |

---

## Troubleshooting

### "Model not available" error

模型未在 Model Garden 中启用。联系项目管理员在 [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) 中启用对应模型，或用 `/model` 切换到已启用的模型。

### "Permission denied" error

你的账号缺少 Vertex AI User 权限。联系项目管理员添加：

```bash
gcloud projects add-iam-policy-binding neu-research \
  --member="user:你的邮箱@embodyx.io" \
  --role="roles/aiplatform.user"
```

### gcloud: command not found

gcloud 路径未加载。尝试：

```bash
exec -l $SHELL
```

如果仍然不行，重新安装：

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### Claude Code 仍然显示旧的 API provider

确认环境变量是否生效：

```bash
echo $CLAUDE_CODE_USE_VERTEX
```

如果返回空值，重载 shell：

```bash
source ~/.zshrc
```

---

## Notes

- 所有用量通过 `neu-research` 计费账号的 GCP Partner Credit 支付
- 使用 Vertex AI 时 `/login` 和 `/logout` 命令会被禁用，认证完全通过 gcloud 处理
- 账单和 credit 余额可在 [GCP Billing Console](https://console.cloud.google.com/billing) 查看
