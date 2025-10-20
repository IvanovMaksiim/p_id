import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Анализ обучения модели на основе CSV-логов (Ultralytics / YOLO).
Строятся графики потерь (loss) и метрик (precision, recall, mAP) по эпохам.
"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, '..', 'runs', 'project_pid_run_aug', 'results.csv')

df = pd.read_csv(csv_path)

print(df.columns)


plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='box_loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='cls_loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='dfl_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Validation Metrics')
plt.legend()
plt.grid(True)
plt.show()
