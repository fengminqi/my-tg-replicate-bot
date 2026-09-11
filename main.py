import os
import traceback
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 1. 获取环境变量
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MY_HF_MODEL = "fengminqi/my-tg-qwen2.5-7b"

# 初始化 Hugging Face 推理客户端
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else InferenceClient()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到用户消息: {user_text}")
    
    # 触发 Telegram 正在输入状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 使用 Hugging Face 官方 Client 直接调用你的微调模型
        response = hf_client.chat_completion(
            model=MY_HF_MODEL,
            messages=[
                {"role": "user", "content": user_text}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )

        reply_text = response.choices[0].message.content.strip()

        if reply_text:
            print(f"🤖 机器人回复: {reply_text}")
            await update.message.reply_text(reply_text)
        else:
            print("⚠️ 模型返回内容为空")
            await update.message.reply_text("走神了，没想好怎么回。")

    except Exception as e:
        print("\n" + "="*50)
        print("❌ 模型调用过程发生异常，具体报错堆栈如下：")
        traceback.print_exc()
        print("="*50 + "\n")
        
        await update.message.reply_text("出了一点小故障，稍后再试试看！")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("❌ 错误：未找到 TELEGRAM_BOT_TOKEN 环境变量！")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Telegram Bot 启动监听中...")
    app.run_polling()

if __name__ == "__main__":
    main()
