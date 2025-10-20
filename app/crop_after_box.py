import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
"""
Разрезание изображений на тайлы с сохранением YOLO-аннотаций.
Особенности:
- Основная сетка тайлов с перекрытием
- Object-centered тайлы для маленьких или больших объектов
- Тайлы подстраиваются под размер объектов
- Подсчет статистики размеров боксов относительно тайла
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


images_dir = os.path.join(BASE_DIR, '..', 'dataset_original_box', 'images', 'all')
labels_dir = os.path.join(BASE_DIR, '..', 'dataset_original_box', 'labels', 'all')

tiles_dir = os.path.join(BASE_DIR, '..', 'dataset_original_box', 'tiles')

tile_size = 1024
MIN_TILE = 512
MAX_TILE = 2048
pad = 0.25
overlap = 0.2

os.makedirs(os.path.join(tiles_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(tiles_dir, "labels"), exist_ok=True)



def read_yolo_label(path):
    """
    Чтение YOLO-файла в список боксов
    :param path: путь к .txt файлу
    :return: список [cls, cx, cy, w, h]
    """
    if not os.path.exists(path):
        return []
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            boxes.append([int(cls), cx, cy, w, h])
    return boxes


def save_yolo_label(path, boxes):
    """
    Сохранение боксов в YOLO-формате
    :param path: путь к файлу
    :param boxes: список боксов [cls, cx, cy, w, h]
    :return: None
    """
    with open(path, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(box, W, H):
    """
    Конвертация YOLO-бокса в координаты пикселей: [cls, x_min, y_min, x_max, y_max, bx, by, bw, bh]
    :param box: [cls, cx, cy, w, h]
    :param W: ширина изображения
    :param H: высота изображения
    :return: tuple с координатами
    """
    cls, cx, cy, w, h = box
    bx, by, bw, bh = cx * W, cy * H, w * W, h * H
    return cls, bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2, bx, by, bw, bh


def pixels_to_yolo(cls, bx, by, bw, bh, tile_w, tile_h, x0, y0):
    """
    Конвертация пиксельных координат бокса в YOLO-формат относительно тайла
    :param cls: класс объекта
    :param bx, by: центр объекта
    :param bw, bh: ширина/высота объекта
    :param tile_w, tile_h: размеры тайла
    :param x0, y0: координаты верхнего левого угла тайла в оригинальном изображении
    :return: [cls, cx, cy, w, h] в координатах 0-1
    """
    cx = (bx - x0) / tile_w
    cy = (by - y0) / tile_h
    w = bw / tile_w
    h = bh / tile_h

    cx, cy, w, h = [max(0, min(1, v)) for v in (cx, cy, w, h)]
    return [cls, cx, cy, w, h]


def extract_tile(img, boxes, x0, y0, tile_w, tile_h, base_name, suffix):
    """
    Извлечение тайла и сохранение только боксов, полностью попавших внутрь
    :param img: numpy array изображения
    :param boxes: список YOLO-боксов
    :param x0, y0: верхний левый угол тайла
    :param tile_w, tile_h: размеры тайла
    :param base_name: имя исходного файла
    :param suffix: суффикс тайла
    """
    tile_boxes = []
    H, W = img.shape[:2]
    x1, y1 = x0 + tile_w, y0 + tile_h

    for b in boxes:
        cls, l, t, r, btm, bx, by, bw, bh = yolo_to_xyxy(b, W, H)
        if l >= x0 and r <= x1 and t >= y0 and btm <= y1:
            tile_boxes.append(pixels_to_yolo(cls, bx, by, bw, bh, tile_w, tile_h, x0, y0))

    if tile_boxes:
        tile = img[y0:y1, x0:x1]
        img_name = f"{base_name}_{suffix}.png"
        label_name = f"{base_name}_{suffix}.txt"
        cv2.imwrite(os.path.join(tiles_dir, "images", img_name), tile)
        save_yolo_label(os.path.join(tiles_dir, "labels", label_name), tile_boxes)


def process_image(img_path, label_path):
    """
    Основная логика разрезания одного изображения
    :param img_path: путь к изображению
    :param label_path: путь к YOLO-разметке
    """
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось прочитать {img_path}")
        return
    H, W = img.shape[:2]
    boxes = read_yolo_label(label_path)
    if not boxes:
        return

    stride = int(tile_size * (1 - overlap))
    created = []


    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1, y1 = min(x0 + tile_size, W), min(y0 + tile_size, H)
            partial = False
            for b in boxes:
                _, l, t, r, btm, *_ = yolo_to_xyxy(b, W, H)

                if l < x0 or r > x1 or t < y0 or btm > y1:
                    if not (r <= x0 or l >= x1 or btm <= y0 or t >= y1):
                        partial = True
                        break
            if not partial:
                extract_tile(img, boxes, x0, y0, x1 - x0, y1 - y0, base_name, f"{x0}_{y0}")
                created.append((x0, y0, x1, y1))


    for b in boxes:
        cls, l, t, r, btm, bx, by, bw, bh = yolo_to_xyxy(b, W, H)


        if any(l >= tx0 and r <= tx1 and t >= ty0 and btm <= ty1 for tx0, ty0, tx1, ty1 in created):
            continue

        obj_size = int(max(bw, bh))


        if obj_size <= tile_size:
            size = max(MIN_TILE, int(obj_size * (1 + pad)))
            size = min(size, tile_size)

        elif obj_size > tile_size and obj_size <= MAX_TILE:
            size = int(obj_size * (1 + pad))
        else:
            size = int(obj_size * (1 + pad))
            size = min(size, max(W, H))


        x0 = max(0, min(int(bx - size / 2), W - size))
        y0 = max(0, min(int(by - size / 2), H - size))

        extract_tile(img, boxes, x0, y0, size, size, base_name, f"obj_{x0}_{y0}")


for img_name in os.listdir(images_dir):
    if img_name.lower().endswith((".png")):
        process_image(
            os.path.join(images_dir, img_name),
            os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
        )

print(" Разделение завершено.")


sizes = []
for lbl in glob.glob(os.path.join(tiles_dir, "labels", "*.txt")):
    with open(lbl, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, _, _, w, h = map(float, parts)
                sizes.append(max(w, h))

if sizes:
    sizes = np.array(sizes)
    print("\n Статистика размеров боксов (относительно тайла):")
    print(f"  Средний размер: {np.mean(sizes):.3f}")
    print(f"  Медиана:        {np.median(sizes):.3f}")
    print(f"  >0.5 (очень крупные): {np.mean(sizes > 0.5) * 100:.1f}%")
    print(f"  <0.05 (очень мелкие): {np.mean(sizes < 0.05) * 100:.1f}%")
else:
    print("\n Не найдено аннотаций для анализа.")
