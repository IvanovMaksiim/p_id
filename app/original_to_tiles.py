from PIL import Image
import os
"""
Разрезка изображений на тайлы с перекрытием (overlap).
Позволяет подготовить изображения для обучения моделей на больших изображениях.
"""
def crop_image(image_path, output_dir, tile_size=1024, overlap=150):
    """
    Разрезает одно изображение на квадраты заданного размера с перекрытием.

    :param image_path: путь к исходному изображению
    :param output_dir: директория для сохранения тайлов
    :param tile_size: размер тайла (по умолчанию 1024)
    :param overlap: перекрытие между тайлами (по умолчанию 150)
    :return: None
    """
    img = Image.open(image_path)
    w, h = img.size
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = img.crop(box)
            tile.save(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{count}.png"))
            count += 1

    print(f"{os.path.basename(image_path)} → {count} тайлов")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(BASE_DIR, '..', 'data', 'pages')
output_dir = os.path.join(BASE_DIR, '..', 'data', 'tiles')

for file in os.listdir(input_dir):
    if file.endswith(".png"):
        crop_image(os.path.join(input_dir, file), output_dir)
