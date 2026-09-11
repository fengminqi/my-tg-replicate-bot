import os
import replicate
from telegram import Update
from telegram.ext import ContextTypes

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"收到用户消息: {user_text}")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 使用 Replicate 官方 Qwen2.5-7B 镜像，并通过 lora_weights 挂载你的 HF 专属模型
        output = replicate.run(
            "qwen/qwen-2.5-7b-instruct",
            input={
                "prompt": user_text,
                "lora_weights": "fengminqi/my-tg-qwen2.5-7b",  # 填入你的 HF 模型路径
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )

        reply_text = "".join([str(item) for item in output]).strip()

        if reply_text:
            await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text("走神了，没想好怎么回。")

    except Exception as e:
        print(f"❌ 模型调用出错: {e}")
        await update.message.reply_text("出了一点小故障，稍后再试试看！")
