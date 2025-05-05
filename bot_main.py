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
import urllib.request

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

def to_coloring_book(input_data):
    if isinstance(input_data, str):  # Если передан путь к файлу
        img = cv2.imread(input_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # Если передан PIL-объект
        img = cv2.cvtColor(np.array(input_data), cv2.COLOR_RGB2GRAY)

    # Сглаживание для уменьшения мелких деталей
    img = cv2.GaussianBlur(img, (9, 9), 0)
    # Извлечение контуров с Canny
    edges = cv2.Canny(img, 50, 150)
    # Очистка мелких деталей
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    # Инверсия (чёрные линии на белом фоне)
    edges = cv2.bitwise_not(edges)
    # Сохранение результата
    output_path = "temp_coloring.png"
    cv2.imwrite(output_path, edges)
    return output_path

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для создания детских раскрасок. Напиши промпт (например, 'кот в шляпе') или отправь изображение, и я превращу это в раскраску!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    logging.info(f"Получен текстовый запрос: {user_text}")

    try:
        translated_text = translator.translate(user_text, dest="en")
        prompt = f"{translated_text}, black and white line art, coloring book style, clean outlines, simple design, highly detailed, no shading, no patterns, no abstract elements"
        negative_prompt = "abstract, patterns, noise, blurry, low detail, grayscale, shadows, textures"
        await update.message.reply_text("Генерирую раскраску по промпту, это займёт 10–60 секунд...")

        # Генерация изображения
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        logging.info("Изображение сгенерировано")

        # Преобразование в раскраску
        output_path = to_coloring_book(image)
        logging.info("Изображение обработано")

        # Отправка результата
        with open(output_path, "rb") as photo:
            await update.message.reply_photo(photo)
        logging.info("Раскраска отправлена пользователю")

        # Очистка
        os.remove(output_path)
    except Exception as e:
        logging.error(f"Ошибка генерации: {e}")
        await update.message.reply_text("Ошибка при генерации. Попробуй другой промпт или отправь изображение!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получение самого большого изображения
    photo = update.message.photo[-1]
    logging.info("Получено изображение от пользователя")

    # Скачивание изображения
    file = await photo.get_file()
    file_url = file.file_path
    temp_image_path = "temp_image.jpg"
    urllib.request.urlretrieve(file_url, temp_image_path)
    logging.info(f"Изображение скачано: {temp_image_path}")

    try:
        # Преобразование в раскраску
        output_path = to_coloring_book(temp_image_path)
        logging.info("Изображение обработано")

        # Отправка результата
        with open(output_path, "rb") as photo:
            await update.message.reply_photo(photo)
        logging.info("Раскраска отправлена пользователю")

        # Очистка
        os.remove(temp_image_path)
        os.remove(output_path)
    except Exception as e:
        logging.error(f"Ошибка обработки изображения: {e}")
        await update.message.reply_text("Ошибка при обработке изображения. Попробуй отправить другое!")

def main():
    app = Application.builder().token("ваш_токен").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()