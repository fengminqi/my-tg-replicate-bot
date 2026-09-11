import os
import traceback
from huggingface_hub import InferenceClient
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 1. 安全获取环境变量
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MY_HF_MODEL = "fengminqi/my-tg-qwen2.5-7b"

# 校验 HF_TOKEN 是否正常获取
if HF_TOKEN:
    print(f"🔑 已成功加载 HF_TOKEN (前缀: {HF_TOKEN[:5]}...)")
else:
    print("⚠️ 未找到 HF_TOKEN 环境变量！")

# 初始化 InferenceClient
client = InferenceClient(token=HF_TOKEN)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到用户消息: {user_text}")
    
    # 触发 Telegram 的 typing（正在输入...）状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 使用格式化 Prompt 进行文本生成
        prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
        
        response = client.text_generation(
            prompt,
            model=MY_HF_MODEL,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )

        reply_text = response.strip()

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
