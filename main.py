import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import replicate

# 开启基础日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    logger.info(f"收到用户消息: {user_text}")
    try:
        # 调用 Replicate 部署模型
        output = replicate.run(
            os.environ["REPLICATE_MODEL"],
            input={"prompt": user_text}
        )
        reply = "".join(output) if isinstance(output, list) else str(output)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"模型调用出错: {e}")
        await update.message.reply_text(f"模型调用出错: {e}")

if __name__ == "__main__":
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("未找到 TELEGRAM_BOT_TOKEN 环境变量")
    
    # 构建应用
    app = ApplicationBuilder().token(bot_token).build()
    
    # 注册消息处理器（只处理文本消息）
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("Bot 正在安全启动，准备接管 Telegram 监听...")
    
    # drop_pending_updates=True 可以自动踢掉所有残留在后台的冲突连接
    app.run_polling(drop_pending_updates=True)
