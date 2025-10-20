from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from pathlib import Path
import json
"""
Предсказания на оригинале с помощью SAHI
сохранение COCO для визуализации в CVAT
"""
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "runs" / "project_pid_run_aug" / "weights" / "best.pt"
SOURCE_DIR = BASE_DIR / "tmp"
EXPORT_DIR = SOURCE_DIR / "results"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(MODEL_PATH),
    confidence_threshold=0.4,
    device="cpu",
)


image_paths = list(SOURCE_DIR.glob("*.png"))

coco_json = {
    "images": [],
    "annotations": [],
    "categories": []
}
category_set = {}
annotation_id = 1
image_id = 1

for img_path in image_paths:
    image_np = read_image(str(img_path))

    result = get_sliced_prediction(
        image_np,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    coco_json["images"].append({
        "id": image_id,
        "file_name": img_path.name,
        "width": image_np.shape[1],
        "height": image_np.shape[0]
    })

    for obj in result.object_prediction_list:
        cat_name = obj.category.name
        if cat_name not in category_set:
            category_id = len(category_set) + 1
            category_set[cat_name] = category_id
            coco_json["categories"].append({
                "id": category_id,
                "name": cat_name
            })
        else:
            category_id = category_set[cat_name]

        bbox = obj.bbox.to_xywh()

        coco_json["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })
        annotation_id += 1

    image_id += 1

CVAT_JSON_PATH = EXPORT_DIR / "cvat_annotations.json"
with open(CVAT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(coco_json, f, indent=4, ensure_ascii=False)

print("Обработка завершена!")
print("COCO JSON сохранён в:", CVAT_JSON_PATH)
