import os
import torch
import traceback
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# 1. 环境变量配置
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_MODEL = "fengminqi/my-tg-qwen2.5-7b"

print("⏳ 正在加载 Tokenizer 和模型，这可能需要 1~2 分钟...")

# 2. 初始化加载基座模型与你的私有 LoRA 适配器
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN, trust_remote_code=True)

# 针对 CPU / 轻量环境优化加载
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="auto",
    token=HF_TOKEN,
    trust_remote_code=True
)

# 动态挂载你的私有 LoRA 权重
model = PeftModel.from_pretrained(base_model, LORA_MODEL, token=HF_TOKEN)
model.eval()

print("✅ 模型与 LoRA 权重加载完成！")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    print(f"📩 收到用户消息: {user_text}")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # 构建对话格式
        messages = [
            {"role": "system", "content": "你是一个幽默、接地气且富有表达力的助手，用日常随意的口吻跟用户对话。"},
            {"role": "user", "content": user_text}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if response_text:
            print(f"🤖 机器人回复: {response_text}")
            await update.message.reply_text(response_text)
        else:
            await update.message.reply_text("走神了，没想好怎么回。")

    except Exception as e:
        print("\n" + "="*50)
        print("❌ 推理过程发生异常，具体报错堆栈如下：")
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
