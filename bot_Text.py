import logging
import os
import cv2
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import urllib.request

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def to_coloring_book(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    # Сглаживание для уменьшения мелких деталей
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # Извлечение контуров с помощью Canny
    edges = cv2.Canny(img, 50, 150)

    # Дополнительная очистка мелких деталей
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
    await update.message.reply_text("Привет! Я бот для создания детских раскрасок. Отправь мне изображение, и я превращу его в раскраску!")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Получение самого большого изображения (highest resolution)
    photo = update.message.photo[-1]
    logging.info("Получено изображение от пользователя")

    # Скачивание изображения
    file = await photo.get_file()
    file_url = file.file_path
    temp_image_path = "temp_image.jpg"
    urllib.request.urlretrieve(file_url, temp_image_path)
    logging.info(f"Изображение скачано: {temp_image_path}")

    try:
        # Преобразование в стиль раскраски
        output_path = to_coloring_book(temp_image_path)
        logging.info("Изображение обработано")

        # Отправка результата
        with open(output_path, "rb") as photo:
            await update.message.reply_photo(photo)
        logging.info("Раскраска отправлена пользователю")

        # Очистка временных файлов
        os.remove(temp_image_path)
        os.remove(output_path)
    except Exception as e:
        logging.error(f"Ошибка обработки изображения: {e}")
        await update.message.reply_text("Ошибка при обработке изображения. Попробуй отправить другое!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, отправь мне изображение, чтобы я мог сделать из него раскраску!")

def main():
    app = Application.builder().token("ваш_токен").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling()

if __name__ == "__main__":
    main()