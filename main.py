import os
import replicate
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 读取环境变量
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
MY_HF_MODEL = "fengminqi/my-tg-qwen2.5-7b"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"收到用户消息: {user_text}")
    
    # 触发 Telegram 的 typing 状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 使用 Replicate 官方 Qwen2.5-7B 镜像 + 动态挂载你的 HF 专属模型
        output = replicate.run(
            "qwen/qwen-2.5-7b-instruct",
            input={
                "prompt": user_text,
                "lora_weights": MY_HF_MODEL,
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

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("未找到 TELEGRAM_BOT_TOKEN 环境变量")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Telegram Bot 启动监听中...")
    app.run_polling()

if __name__ == "__main__":
    main()
