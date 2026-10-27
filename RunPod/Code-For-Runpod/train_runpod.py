import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. Dataset configuration path on the cloud server
    data_yaml_path = Path("/workspace/data/pcb-defect-dataset/data.yaml")

    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found. Check the path.")
        return

    # 2. Initialize the YOLO model architecture
    model_name = "yolo26m.pt"
    print(f"[Runpod] Loading base architecture: {model_name}...")
    model = YOLO(model_name)

    # 3. High-performance cloud training settings for RTX A5000 (24GB VRAM)
    print("[Runpod] Starting training on Cloud GPU...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=100,
        batch=32,                         # Increased batch size to leverage 24GB VRAM
        imgsz=640,
        workers=8,                        # Multi-threaded dataloader using cloud vCPUs
        device=0,                         # First available cloud GPU
        amp=True,                         # Mixed precision FP16 for faster execution
        patience=15,                      # Early stopping patience
        save=True,
        project="/workspace/runs/pcb_train",
        name="runpod_pcb_exp",
        exist_ok=True
    )

    best_model_path = Path("/workspace/runs/pcb_train/runpod_pcb_exp/weights/best.pt")
    print(f"[Runpod] Training completed! Best weights: {best_model_path}")

if __name__ == "__main__":
    main()