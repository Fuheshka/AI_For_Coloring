import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Настройка кэша Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ["HF_HOME"] = "D:/huggingface_cache"  # Замените на ваш путь с достаточным местом

# Инициализация Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True
)
pipe.enable_attention_slicing()
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация переводчика
translator = GoogleTranslator()

def to_coloring_book(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.bitwise_not(edges)
    return Image.fromarray(edges)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напиши, что хочешь увидеть на раскраске, например: 'кот в шляпе'.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    logging.info(f"Получен запрос: {user_text}")

    try:
        translated_text = translator.translate(user_text, dest="en")
    except Exception as e:
        logging.error(f"Ошибка перевода: {e}")
        await update.message.reply_text("Ошибка перевода. Попробуй написать на английском!")
        return

    prompt = f"{translated_text}, black and white line art, coloring book style, clean outlines, simple design, highly detailed, no shading, no patterns, no abstract elements"
    negative_prompt = "abstract, patterns, noise, blurry, low detail, grayscale, shadows, textures"
    await update.message.reply_text("Генерирую раскраску, это займёт 10–60 секунд...")

    try:
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        logging.info("Изображение сгенерировано")

        coloring_image = to_coloring_book(image)
        coloring_image.save("temp_coloring.png", optimize=True, quality=85)
        logging.info("Изображение обработано и сохранено")

        with open("temp_coloring.png", "rb") as photo:
            await update.message.reply_photo(photo)
        logging.info("Изображение отправлено пользователю")
    except Exception as e:
        logging.error(f"Ошибка генерации: {e}")
        await update.message.reply_text("Ошибка при генерации. Попробуй другой запрос!")

def main():
    app = Application.builder().token("ваш_токен").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling()

if __name__ == "__main__":
    main()