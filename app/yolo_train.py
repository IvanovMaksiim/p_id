import torch
from ultralytics import YOLO
import multiprocessing
import os

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("CUDA available:", torch.cuda.is_available())

    model_path = os.path.join(BASE_DIR, "yolov8n.pt")
    model = YOLO(model_path)

    data_path = os.path.join(BASE_DIR, "dataset", "data.yaml")
    project_path = os.path.join(BASE_DIR, "runs")

    model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        project=project_path,
        batch=16,
        name="project_pid_run_aug",
        augment=True,
        auto_augment="RandAugment",
        # classes= enough_classes,
        patience=5,
        workers=0,
        verbose=True
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
