import os
import traceback
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# 你的 Hugging Face Space 名称，例如 "fengminqi/my-qwen-api"
HF_SPACE_NAME = os.getenv("HF_SPACE_NAME", "fengminqi/my-qwen-api")

print(f"🚀 Render 转接节点启动！对接 Space: {HF_SPACE_NAME}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到 Telegram 消息: {user_text}")
    
    # 触发 Telegram 的 typing 正在输入状态
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 使用轻量级 HTTP 请求调用 Hugging Face Space
        url = f"https://{HF_SPACE_NAME.replace('/', '-')}.hf.space/call/predict"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            # 1. 提交推理任务
            response = await http_client.post(url, json={"data": [user_text]}, headers=headers)
            response_json = response.json()
            
            if "event_id" in response_json:
                event_id = response_json["event_id"]
                result_url = f"https://{HF_SPACE_NAME.replace('/', '-')}.hf.space/call/predict/{event_id}"
                
                # 2. 获取推理结果
                result_resp = await http_client.get(result_url, headers=headers)
                result_text = result_resp.text
                
                # 解析返回的 SSE 流数据
                lines = result_text.strip().split("\n")
                reply = ""
                for line in lines:
                    if line.startswith("data:"):
                        reply = line.replace("data:", "").strip()
                        # 去除 JSON 数组外壳
                        if reply.startswith("[") and reply.endswith("]"):
                            import json
                            reply = json.loads(reply)[0]
                
                if reply:
                    print(f"🤖 成功回复用户: {reply}")
                    await update.message.reply_text(str(reply))
                else:
                    await update.message.reply_text("思考超时了，没想好怎么回。")
            else:
                print(f"⚠️ Space 返回异常: {response_json}")
                await update.message.reply_text("后端的 Space 正在唤醒中，请稍等十几秒再发一次试试！")

    except Exception as e:
        print("\n" + "="*50)
        print("❌ 调用过程发生异常:")
        traceback.print_exc()
        print("="*50 + "\n")
        await update.message.reply_text("出了一点小故障，稍后再试试看！")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("❌ 未找到 TELEGRAM_BOT_TOKEN 环境变量！")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🚀 Telegram Bot 轮询监听中...")
    app.run_polling()

if __name__ == "__main__":
    main()
