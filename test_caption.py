from semantic_image_search.models.caption import ImageCaption
import os

# Укажите устройство
device = "cpu"  # Используйте "cuda", если есть GPU

# Создайте экземпляр модели
captioner = ImageCaption(device=device)

# Путь к одному изображению
image_path = "img/cat.jpg"
caption = captioner.caption(image_path)
print(f"Подпись к изображению [cat]: {caption}")

image_path = "img/rose.jpg"
caption = captioner.caption(image_path)
print(f"Подпись к изображению [rose]: {caption}")

# Опционально: тестирование на папке с изображениями
# image_folder = "path/to/your/image/folder"  # Замените на актуальный путь
# for image_file in os.listdir(image_folder):
#     if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
#         image_path = os.path.join(image_folder, image_file)
#         caption = captioner.caption(image_path)
#         print(f"Изображение: {image_file}, Подпись: {caption}")
