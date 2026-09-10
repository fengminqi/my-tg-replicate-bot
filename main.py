import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import replicate

logging.basicConfig(level=logging.INFO)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # 异步调用你的 Replicate 模型
    try:
        output = replicate.run(
            os.environ["REPLICATE_MODEL"],
            input={"prompt": user_text}
        )
        reply = "".join(output) if isinstance(output, list) else str(output)
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text(f"模型调用出错: {e}")

if __name__ == "__main__":
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot 启动成功，正在监听 Telegram 消息...")
    app.run_polling()
