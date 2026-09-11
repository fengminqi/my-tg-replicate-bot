import os
import replicate
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 1. 统一从环境变量获取密钥（不要在代码里写明文 Token）
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")  # 安全读取 HF Token
MY_HF_MODEL = "fengminqi/my-tg-qwen2.5-7b"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        output = replicate.run(
            "replicate/hf-inference",
            input={
                "model": MY_HF_MODEL,
                "prompt": f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n",
                "hf_token": os.getenv("HF_TOKEN"),  # 从环境变量读取
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )

        reply_text = "".join([str(item) for item in output]).strip()
        reply_text = reply_text.replace("<|im_end|>", "").strip()

        if reply_text:
            await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text("走神了，没想好怎么回。")

    except Exception as e:
        print(f"❌ 模型推理出错: {e}")
        await update.message.reply_text("出了一点小故障，稍后再试试看！")
