import os
import logging
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# 配置日志输出
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# ---------------- 从 Render 环境变量获取配置 ----------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID", "fengminqi/my-tg-qwen2.5-7b")
# -----------------------------------------------------------

def query_huggingface_model(prompt: str) -> str:
    """调用 Hugging Face 云端 Serverless Inference API"""
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # 构建 Qwen2.5 对话格式
    formatted_prompt = f"<|im_start|>system\n你是一个随意的助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "error" in result:
            error_msg = result["error"]
            if "currently loading" in error_msg:
                return "⏳ 模型正在云端冷启动中（需要 20-30 秒），请稍等片刻再发一条消息！"
            return f"⚠️ HF API 提示: {error_msg}"
        return "抱歉，云端模型未返回有效回复。"
    except requests.exceptions.Timeout:
        return "⏳ 请求超时，云端模型正在冷启动或响应较慢，请稍后重试。"
    except Exception as e:
        return f"❌ 出现异常: {str(e)}"

# 机器人 /start 命令回复
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我已经成功连接到云端微调模型，随时可以聊天！")

# 处理文本消息
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # 显示正在输入状态
    await update.message.chat.send_action(action="typing")
    
    # 请求 HF 模型并发送回复
    bot_reply = query_huggingface_model(user_text)
    await update.message.reply_text(bot_reply)

if __name__ == "__main__":
    if not BOT_TOKEN or not HF_TOKEN:
        print("❌ 错误：未检测到 BOT_TOKEN 或 HF_TOKEN 环境变量！")
        exit(1)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("🚀 Telegram Bot 启动中，开始监听消息...")
    app.run_polling()
