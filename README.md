# Qwen2.5-1.5B Telegram bot on Render

这套方案用你的 `train.jsonl` 重新训练一个较小的 Qwen2.5-1.5B 风格模型，转换为 GGUF Q4_K_M，然后由 Render CPU Worker 本地加载。它不调用 Hugging Face Inference Endpoint。

## 1. 在 Colab 训练并量化

打开 `colab_train_qwen15b.ipynb`，按顺序运行所有单元格：

1. 上传 `C:\Users\ASUS\Downloads\train.jsonl`。
2. 在 Colab 的输入框中粘贴一个有写入权限的 Hugging Face token。Token 不要发到聊天里。
3. 训练 Qwen2.5-1.5B-Instruct 的 LoRA 适配器。
4. 合并为 16-bit 模型并量化成 `qwen15b-style-Q4_K_M.gguf`。
5. Notebook 会把 GGUF 上传到 `fengminqi/my-tg-qwen1.5b-style`。仓库可以设为 Private，但 Render 的 `HF_TOKEN` 必须有读取权限。

训练完成后，在 Hugging Face 仓库 Files 页面确认文件名和大小。

## 2. 部署 Render

把本目录文件放在 GitHub 仓库根目录，在 Render 选择 **New → Blueprint** 并连接仓库。`render.yaml` 使用 `1c-2g` Background Worker，当前约 `$25/月`。

在 Render Secret Environment Variables 中填写：

- `TELEGRAM_BOT_TOKEN`
- `HF_TOKEN`

如果你把 GGUF 仓库公开，`HF_TOKEN` 可以省略；私有仓库则需要一个有读取权限的 token。首次启动会从 Hugging Face 下载 GGUF，模型之后保存在当前实例文件系统中。

## 3. 参数调整

- `MAX_OUTPUT_TOKENS=256` 控制回答长度。
- `N_CTX=2048` 控制上下文长度；内存不足时降到 `1536`。
- `N_THREADS=2` 适合 1 CPU Worker。
- `ALLOWED_CHAT_IDS` 可填 Telegram chat ID，多个 ID 用逗号分隔。

Render Worker 是持续运行的服务，因此即使没有消息也会按月计费。若要停止计费，需要在 Render 中手动 Suspend 服务。
