from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from pathlib import Path
import json

# Базовая директория проекта (на уровень выше app/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути
MODEL_PATH = BASE_DIR / "runs" / "project_pid_run_aug" / "weights" / "best.pt"
SOURCE_DIR = BASE_DIR / "tmp"
EXPORT_DIR = SOURCE_DIR / "labels"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Проверка наличия модели
assert MODEL_PATH.exists(), f"Файл модели не найден: {MODEL_PATH}"

# Загрузка модели
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(MODEL_PATH),
    confidence_threshold=0.4,
    device="cpu"  # можно "cuda:0"
)

# Список изображений
image_paths = list(SOURCE_DIR.glob("*.png"))
print(f"Найдено изображений: {len(image_paths)}")

# Словарь для сопоставления категорий
category_set = {}

for img_path in image_paths:
    image_np = read_image(str(img_path))

    # Получаем предсказания с SAHI
    result = get_sliced_prediction(
        image_np,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    # Имя txt-файла
    label_path = EXPORT_DIR / (img_path.stem + ".txt")

    # Собираем строки для этого изображения
    lines = []
    for obj in result.object_prediction_list:
        cat_name = obj.category.name

        # Назначаем id категории
        if cat_name not in category_set:
            category_set[cat_name] = len(category_set)
        class_id = category_set[cat_name]

        # Получаем bbox [x_min, y_min, x_max, y_max]
        bbox = obj.bbox.to_xyxy()
        x_min, y_min, x_max, y_max = bbox

        # Преобразуем в YOLO-формат (x_center, y_center, w, h)
        img_h, img_w = image_np.shape[:2]
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Сохраняем результат
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("✅ Обработка завершена!")
print("TXT-аннотации сохранены в:", EXPORT_DIR)

# Вывод списка категорий
print("\nКатегории (class_id -> name):")
for name, cid in category_set.items():
    print(f"{cid}: {name}")
