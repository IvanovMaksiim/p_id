from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.ops import xywh2xyxy
import numpy as np
from pathlib import Path

# --- Пути ---
gt_file = Path(r"C:\project_pid\pythonProject2\470a1ccb-1a8e-49c4-bbe4-219f9c78a68f-70.txt")       # оригинальная разметка
pred_file = Path(r"C:\project_pid\pythonProject2\tmp\labels\470a1ccb-1a8e-49c4-bbe4-219f9c78a68f-70.txt")  # предсказания

# --- Инициализация ---
metrics = DetMetrics()

# --- Загрузка данных ---
gt = np.loadtxt(gt_file).reshape(-1, 5) if gt_file.stat().st_size else np.empty((0, 5))
pred_raw = np.loadtxt(pred_file)

if pred_raw.ndim == 1:
    pred_raw = pred_raw.reshape(1, -1)

# Если в предсказаниях нет confidence, добавляем фиктивный (1.0)
if pred_raw.shape[1] == 5:
    pred = np.hstack([pred_raw, np.ones((pred_raw.shape[0], 1))])
else:
    pred = pred_raw

# --- Конвертация YOLO → xyxy ---
gt[:, 1:5] = xywh2xyxy(gt[:, 1:5])
pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])

# --- Добавляем фиктивный image_id ---
img_id = np.zeros((len(gt), 1))
gt = np.concatenate([img_id, gt], 1)    # [img_id, cls, x1, y1, x2, y2]
pred = np.concatenate([img_id, pred], 1)  # [img_id, cls, x1, y1, x2, y2, conf]

# --- Обработка одной пары ---
metrics.process_batch(pred, gt)

# --- Результаты ---
results = metrics.results_dict
print(f"Precision: {results['metrics/precision(B)']:.4f}")
print(f"Recall: {results['metrics/recall(B)']:.4f}")
print(f"mAP@0.5: {results['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {results['metrics/mAP50-95(B)']:.4f}")
