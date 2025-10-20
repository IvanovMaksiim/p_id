from ultralytics import YOLO
import torch
import multiprocessing
import os

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model_path = os.path.join(BASE_DIR, "runs", "project_pid_run_aug", "weights", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = YOLO(model_path)


    data_path = os.path.join(BASE_DIR, "dataset", "data.yaml")


    results = model.val(
        data=data_path,
        split='test',
        imgsz=640,
        batch=16,
        save_json=True,
        save_hybrid=True,
        conf=0.25,
        project=os.path.join(BASE_DIR, "runs_val_test"),
        name="project_val"
    )

    print(results)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
