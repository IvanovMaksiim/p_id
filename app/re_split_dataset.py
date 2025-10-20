import os
import shutil
import random
"""
Разбивка датасета изображений и меток на train, val и test наборы.
Поддерживаются режимы: 
 - train_val_test (70/20/10 по умолчанию)
 - train_val (например, 80/20)
 - all_in_train (всё идёт в train)
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
images_all = os.path.join(BASE_DIR, '..', 'dataset', 'images', 'all')
labels_all = os.path.join(BASE_DIR, '..', 'dataset', 'labels', 'all')

dataset_dir = os.path.join(BASE_DIR, '..', 'dataset')


# Доступные варианты: "train_val_test", "train_val", "all_in_train"
split_mode = "train_val"

# Проценты для разбивки (для train_val_test и train_val)
split_ratios = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

def make_dirs(splits):
    """
    Создаёт директории для train/val/test под images и labels.

    :param splits: список сплитов (например: ["train", "val", "test"])
    :return: None
    """
    for split in splits:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)


def copy_split(files, split_name):
    """
    Копирует изображения и соответствующие лейблы в указанную директорию сплита.

    :param files: список имён файлов изображений (.png)
    :param split_name: название сплита ("train", "val" или "test")
    :return: None
    """
    dst_img_dir = os.path.join(dataset_dir, "images", split_name)
    dst_lbl_dir = os.path.join(dataset_dir, "labels", split_name)

    for img_file in files:
        src_img = os.path.join(images_all, img_file)
        dst_img = os.path.join(dst_img_dir, img_file)
        shutil.copy(src_img, dst_img)


        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_lbl = os.path.join(labels_all, label_file)
        shutil.copy(src_lbl, os.path.join(dst_lbl_dir, label_file))


image_files = [f for f in os.listdir(images_all) if f.lower().endswith((".png", '.jpg'))]
image_files.sort()
random.shuffle(image_files)

n_total = len(image_files)

if split_mode == "train_val_test":
    make_dirs(["train", "val", "test"])

    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["val"])
    n_test = n_total - n_train - n_val

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")

    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")
    print(f"  Test:  {len(test_files)}")

elif split_mode == "train_val":
    make_dirs(["train", "val"])

    n_train = int(n_total * (split_ratios["train"] / (split_ratios["train"] + split_ratios["val"])))
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]

    copy_split(train_files, "train")
    copy_split(val_files, "val")

    print(f"  Train: {len(train_files)}")
    print(f"  Val:   {len(val_files)}")

elif split_mode == "all_in_train":
    make_dirs(["train"])
    copy_split(image_files, "train")

    print(f"  Train: {len(image_files)}")

