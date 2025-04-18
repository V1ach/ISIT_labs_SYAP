import os
import re
import logging
import pytesseract
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
from PIL import Image, ImageEnhance, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ai-forever/rugpt3large_based_on_gpt2"
model_path = "./rugpt3large_model"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    print(f"PyTorch установлен: {torch.__version__}")
except ImportError:
    raise RuntimeError("Для работы с моделями требуется установить PyTorch.")

if os.path.exists(model_path):
    logger.info("Загрузка модели и токенизатора с диска...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
else:
    logger.info("Загрузка модели и токенизатора с Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("Сохранение модели и токенизатора на диск...")
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

def save_user_id(user_id):
    if not os.path.exists("users.txt"):
        open("users.txt", "w").close()

    with open("users.txt", "r+") as file:
        users = file.readlines()
        if str(user_id) + "\n" not in users:
            file.write(f"{user_id}\n")

def get_all_user_ids():
    if not os.path.exists("users.txt"):
        return []

    with open("users.txt", "r") as file:
        return [int(line.strip()) for line in file.readlines()]

def clean_response(text):
    text = text.replace("'", "").replace('"', "")
    text = re.sub(r"(.+?)\1+", r"\1", text)
    return text.strip()

async def generate_response_with_ai(user_message):
    try:
        prompt = (
            f"Ты — тролль с 4chan. Твой стиль общения — саркастичный, токсичный и провокационный. "
            r"Ты остроумный и всегда стараешься задеть собеседника, но при этом твои ответы релевантны и соответствуют контексту. "
            r"Ты не повторяешь одни и те же фразы и не пишешь слишком длинные сообщения.\n\n"
            f"Пользователь написал: '{user_message}'\n"
            f"Ответь на это сообщение в своём стиле, но не продолжай фразу пользователя."
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
            #num_beams=3,
            #early_stopping=True
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        generated_text = clean_response(generated_text)
        return generated_text if generated_text else None

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа с помощью нейросети: {e}")
        return None

async def handle_message(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    save_user_id(user_id)

    user_message = update.message.text
    logger.info(f"Пользователь {user_id} написал: {user_message}")

    await update.message.reply_text("Бот думает...")
    ai_response = await generate_response_with_ai(user_message)

    if ai_response and ai_response.strip():
        logger.info(f"Бот ответил пользователю {user_id}: {ai_response}")
        await update.message.reply_text(ai_response)
    else:
        logger.warning(f"Бот не смог сгенерировать ответ для пользователя {user_id}.")
        await update.message.reply_text("Извини, я не смог придумать ответ. Попробуй спросить что-нибудь ещё! (╯°□°）╯︵ ┻━┻")

async def handle_image(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    try:
        file = await update.message.photo[-1].get_file()
        file_path = f"temp_{update.message.message_id}.jpg"
        await file.download_to_drive(file_path)

        if not os.path.exists(file_path):
            logger.error(f"Не удалось сохранить изображение от пользователя {user_id}.")
            await update.message.reply_text("Не удалось сохранить изображение.")
            return

        try:
            image = Image.open(file_path)
        except UnidentifiedImageError:
            logger.error(f"Не удалось открыть изображение от пользователя {user_id}.")
            await update.message.reply_text("Не удалось открыть изображение. Проверьте формат.")
            return

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        text = pytesseract.image_to_string(image, lang='eng+rus', config='--psm 6')

        logger.info(f"Распознанный текст от пользователя {user_id}: {text}")

        os.remove(file_path)

        if text and text.strip():
            await update.message.reply_text(f"Распознанный текст: {text}")
            await handle_message(update, context)
        else:
            logger.warning(f"Не удалось распознать текст на изображении от пользователя {user_id}.")
            await update.message.reply_text("Не удалось распознать текст на изображении.")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения от пользователя {user_id}: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения.")

async def broadcast_message(update: Update, context: CallbackContext):
    admin_id = 868313781
    user_id = update.message.from_user.id

    if user_id != admin_id:
        logger.warning(f"Пользователь {user_id} попытался использовать команду /text без прав.")
        await update.message.reply_text("У вас нет прав для выполнения этой команды.")
        return

    if not context.args:
        logger.warning(f"Пользователь {user_id} использовал команду /text без аргументов.")
        await update.message.reply_text("Использование: /text ТЕКСТ СООБЩЕНИЯ")
        return

    message_text = " ".join(context.args)
    user_ids = get_all_user_ids()

    for user_id in user_ids:
        try:
            await context.bot.send_message(chat_id=user_id, text=message_text)
            logger.info(f"Сообщение отправлено пользователю {user_id}.")
        except Exception as e:
            logger.error(f"Не удалось отправить сообщение пользователю {user_id}: {e}")

    await update.message.reply_text(f"Сообщение отправлено {len(user_ids)} пользователям.")

async def start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    logger.info(f"Пользователь {user_id} запустил бота.")
    await update.message.reply_text("Привет! Я твой чат-бот.")

def main():
    application = ApplicationBuilder().token("8167413265:AAHTu91e1aVSaVBRhvwGCB7iFH9dk7nvw9Q").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("text", broadcast_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()