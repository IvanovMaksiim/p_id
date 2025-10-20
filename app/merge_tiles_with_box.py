import os
import json
from PIL import Image
import numpy as np
"""
Возврат к оригинальному изображения после предварительного разреза
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

images_dir = os.path.join(BASE_DIR, 'data', 'pages')
tiles_dir = os.path.join(BASE_DIR, 'dataset', 'images', 'all')
preds_dir = os.path.join(BASE_DIR, 'dataset', 'labels', 'all')

output_dir = os.path.join(BASE_DIR, 'dataset_original_box', 'COCO_after_merge')
os.makedirs(output_dir, exist_ok=True)

tile_size = 1024
overlap = 150
step = tile_size - overlap

classes = [
    "armatura_ruchn", "klapan_obratn", "regulator_ruchn", "armatura_electro",
    "regulator_electro", "drossel", "perehod", "klapan_obratn_seroprivod",
    "armatura_seroprivod", "regulator_seroprivod", "armatura_membr_electro",
    "nasos", "ventilaytor", "predohran", "condensatootvod", "rashodomernaya_shaiba",
    "vodostruiniy_nasos", "teploobmen", "zaglushka", "gidrozatvor", "bak", "voronka",
    "filtr_meh", "separator", "kapleulov", "celindr_turb", "redukcion_ustr",
    "bistro_redukc_ustr", "separator_paro", "dearator", "silfonnii_kompensator",
    "electronagrevat", "smotrowoe_steclo", "datchik", "annotation", "output"
]

def nms(boxes, scores, iou_threshold=0.5):
    """
    убираем дублирующие боксы
    :param boxes: список боксов в формате [x, y, w, h]
    :param scores: список confidence (уверенности модели) для каждого бокса
    :param iou_threshold: порог IoU, выше которого боксы считаются дубликатами
    :return: список индексов оставшихся боксов после NMS
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]; y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def generate_tile_grid_positions(w_img, h_img, tile_size, overlap):
    """
    Создаёт сетку позиций тайлов для исходного изображения.

    :param w_img: ширина исходного изображения
    :param h_img: высота исходного изображения
    :param tile_size: размер тайла в пикселях
    :param overlap: перекрытие между тайлами
    :return: список словарей с координатами тайлов:
             [{"count": 0, "x": ..., "y": ..., "w": ..., "h": ...}, ...]
    """
    positions = []
    step = tile_size - overlap
    count = 0
    for y in range(0, h_img, step):
        for x in range(0, w_img, step):
            tw = min(tile_size, w_img - x)
            th = min(tile_size, h_img - y)
            positions.append({"count": count, "x": x, "y": y, "w": tw, "h": th})
            count += 1
    return positions


coco = {
    "info": {"description": "merged", "version": "1.0"},
    "licenses": [{"id":1,"name":"none"}],
    "images": [],
    "annotations": [],
    "categories": [{"id": i+1, "name": classes[i] if i < len(classes) else f"class_{i+1}"} for i in range(len(classes))]
}

ann_id = 1
image_id = 1

for img_file in sorted(os.listdir(images_dir)):
    if not img_file.lower().endswith((".png")):
        continue
    img_path = os.path.join(images_dir, img_file)
    base_name = os.path.splitext(img_file)[0]
    img = Image.open(img_path)
    w_img, h_img = img.size

    coco["images"].append({
        "id": image_id,
        "file_name": img_file,
        "width": w_img,
        "height": h_img
    })

    grid = generate_tile_grid_positions(w_img, h_img, tile_size, overlap)
    for g in grid:
        g["tile_filename"] = f"{base_name}_{g['count']}.png"

    boxes_all, scores_all, labels_all = [], [], []

    for g in grid:
        tile_name = g["tile_filename"]
        tile_path = os.path.join(tiles_dir, tile_name)
        pred_path = os.path.join(preds_dir, os.path.splitext(tile_name)[0] + ".txt")

        if not os.path.exists(tile_path) or not os.path.exists(pred_path):
            continue

        with Image.open(tile_path) as t:
            tw, th = t.size

        with open(pred_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(float(parts[0]))
                x_c, y_c, bw, bh = map(float, parts[1:5])
                x_min_local = x_c * tw - (bw * tw)/2
                y_min_local = y_c * th - (bh * th)/2
                x_min_global = max(0.0, x_min_local + g["x"])
                y_min_global = max(0.0, y_min_local + g["y"])
                w_abs = min(bw * tw, w_img - x_min_global)
                h_abs = min(bh * th, h_img - y_min_global)
                if w_abs <=0 or h_abs <=0: continue
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                boxes_all.append([x_min_global, y_min_global, w_abs, h_abs])
                scores_all.append(score)
                labels_all.append(cls)

    if len(boxes_all) == 0:
        print(f"{img_file}: нет боксов")
        image_id += 1
        continue

    boxes_arr = np.array(boxes_all)
    scores_arr = np.array(scores_all)
    labels_arr = np.array(labels_all)

    keep_indices = []
    for cls in np.unique(labels_arr):
        idxs = np.where(labels_arr == cls)[0]
        cls_boxes = boxes_arr[idxs]
        cls_scores = scores_arr[idxs]
        keep = nms(cls_boxes, cls_scores, iou_threshold=0.5)
        keep_indices.extend(idxs[keep])

    keep_indices = sorted(keep_indices, key=lambda i: scores_arr[i], reverse=True)

    for i in keep_indices:
        x_min, y_min, w_box, h_box = boxes_arr[i]
        lab = labels_arr[i]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": int(lab) + 1,
            "bbox": [float(x_min), float(y_min), float(w_box), float(h_box)],
            "area": float(w_box * h_box),
            "iscrowd": 0
        })
        ann_id += 1

    image_id += 1

out_json = os.path.join(output_dir, "merged_annotations.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print(f"COCO JSON сохранён: {out_json}")
