import os
import traceback
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到 Telegram 消息: {user_text}")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if not API_URL:
        await update.message.reply_text("未配置 API_URL 环境变量！")
        return

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(API_URL, json={"prompt": user_text})
            reply_text = resp.json().get("response", "走神了，没想好怎么回。")

        print(f"🤖 机器人回复: {reply_text}")
        await update.message.reply_text(reply_text)

    except Exception as e:
        print("❌ 调用发生异常:")
        traceback.print_exc()
        await update.message.reply_text("出了一点小故障，稍后再试试看！")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("❌ 未找到 TELEGRAM_BOT_TOKEN 环境变量！")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Telegram Bot 监听启动中...")
    app.run_polling()

if __name__ == "__main__":
    main()
