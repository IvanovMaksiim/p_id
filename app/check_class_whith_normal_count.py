import os
from collections import defaultdict
import yaml
"""
Подсчёт количества объектов каждого класса в разметке YOLO.
Позволяет выявить редкие классы и фильтровать их по порогу для обучения.
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_all = os.path.join(BASE_DIR, '..', 'dataset', 'labels', 'all')

yaml_path = os.path.join(BASE_DIR, '..', 'dataset', 'data.yaml')

with open(yaml_path, "r", encoding="utf-8") as f:
    data_yaml = yaml.safe_load(f)
all_classes = data_yaml["names"]


class_counts = defaultdict(int)
for cls_id in all_classes.keys():
    class_counts[int(cls_id)] = 0

def count_labels(label_dir):
    """
    Подсчитывает количество объектов каждого класса в папке с YOLO-текстовыми файлами.

    :param label_dir: путь к папке с .txt файлами разметки
    :return: None (обновляет глобальный словарь class_counts)
    """
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(label_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1

count_labels(labels_all)

threshold = 10
# те классы которые больше порога, идут в обучающую модель для авторазметки, чтобы не быол проблем с единичными классами
enough_classes = [cls_id for cls_id, count in class_counts.items() if count >= threshold]

print(f"Классы с достаточным количеством примеров {enough_classes}")