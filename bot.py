import asyncio
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "fengminqi/my-tg-qwen1.5b-style")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "qwen15b-style-Q4_K_M.gguf")
MODEL_PATH = os.getenv("MODEL_PATH")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "你正在模仿用户本人的聊天风格。使用自然、简洁、口语化的中文，保持用户常用的语气和用词。不要声称自己是真实的用户；如果不确定，就直接说明。",
)
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "2"))
ALLOWED_CHAT_IDS = {
    int(chat_id.strip())
    for chat_id in os.getenv("ALLOWED_CHAT_IDS", "").split(",")
    if chat_id.strip()
}


def resolve_model_path() -> str:
    if MODEL_PATH:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"MODEL_PATH does not exist: {path}")
        return str(path)
    return hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)


logger.info("Loading GGUF model %s/%s", MODEL_REPO_ID, MODEL_FILENAME)
llm = Llama(
    model_path=resolve_model_path(),
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=128,
    verbose=False,
)
histories: Dict[int, Deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
generation_lock = asyncio.Lock()


def generate_sync(messages: List[dict]) -> str:
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
    )
    return (result["choices"][0]["message"].get("content") or "").strip()


async def generate_reply(chat_id: int, text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *histories[chat_id], {"role": "user", "content": text}]
    async with generation_lock:
        reply = await asyncio.to_thread(generate_sync, messages)
    if not reply:
        raise RuntimeError("模型返回了空消息")
    histories[chat_id].append({"role": "user", "content": text})
    histories[chat_id].append({"role": "assistant", "content": reply})
    return reply


def is_allowed(update: Update) -> bool:
    return not ALLOWED_CHAT_IDS or update.effective_chat.id in ALLOWED_CHAT_IDS


def split_for_telegram(text: str, limit: int = 4096):
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if is_allowed(update):
        await update.message.reply_text("你好，发消息给我吧。发送 /reset 可以清空当前对话记忆。")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if is_allowed(update):
        histories.pop(update.effective_chat.id, None)
        await update.message.reply_text("已清空当前对话记忆。")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text or not is_allowed(update):
        return
    text = update.message.text.strip()
    if not text:
        return
    if len(text) > MAX_INPUT_CHARS:
        await update.message.reply_text(f"消息太长了，请控制在 {MAX_INPUT_CHARS} 个字符以内。")
        return
    try:
        chat_id = update.effective_chat.id
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        reply = await generate_reply(chat_id, text)
        for chunk in split_for_telegram(reply):
            await update.message.reply_text(chunk)
    except Exception:
        logger.exception("生成回复失败")
        await update.message.reply_text("模型暂时不可用，请稍后再试。")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Telegram update error: %s", context.error)


def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    logger.info("Starting local GGUF bot")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
