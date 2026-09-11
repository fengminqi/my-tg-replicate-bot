import os
import traceback
from gradio_client import Client
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 1. 获取环境变量
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# 你的 Hugging Face Space 名称 (格式: 用户名/Space名)
HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "fengminqi/my-qwen-api")

print(f"🚀 初始化 Gradio Client，连接后端 Space: {HF_SPACE_NAME}")

# 初始化连接到 HF Space
try:
    hf_client = Client(HF_SPACE_NAME, hf_token=HF_TOKEN)
    print("✅ 成功建立与 Space 推理后端的 Client 连接！")
except Exception as e:
    print(f"⚠️ 初始化 Gradio Client 失败: {e}")
    hf_client = None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到 Telegram 消息: '{user_text}' (来自用户: {update.effective_user.id})")
    
    # 触发 Telegram 的 typing 正在输入状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if not hf_client:
        await update.message.reply_text("后端推理服务连接失败，请检查 HF_SPACE_NAME 是否正确或 Space 是否处于 Running 状态！")
        return

    try:
        print("⏳ 正在请求 Hugging Face Space 生成回复...")
        
        # 使用 gradio_client 官方 predict 方法，自动处理协议与 JSON 解析
        reply_result = hf_client.predict(
            user_text,
            api_name="/predict"
        )
        
        reply_str = str(reply_result).strip() if reply_result else ""

        if reply_str:
            print(f"🤖 成功获得模型回复: {reply_str}")
            await update.message.reply_text(reply_str)
        else:
            print("⚠️ Space 返回内容为空")
            await update.message.reply_text("走神了，没想好怎么回。")

    except Exception as e:
        print("\n" + "="*50)
        print("❌ 调用 Space 推理接口发生异常:")
        traceback.print_exc()
        print("="*50 + "\n")
        
        await update.message.reply_text("出了一点小故障，稍后再试试看！")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("❌ 错误：未找到 TELEGRAM_BOT_TOKEN 环境变量！")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Telegram Bot 监听启动中...")
    app.run_polling()

if __name__ == "__main__":
    main()
