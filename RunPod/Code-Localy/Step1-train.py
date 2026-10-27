import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. נתיב לקובץ ה-data.yaml במחשב המקומי
    data_yaml_path = Path(r"GPU - On-line Websites/RunPod/Code-Localy/data.yaml")

   
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found. Check the path.")
        return

    # 2. אתחול המודל (yolo11s או yolo11m - לוקלית yolo11s/m מצוינים לבדיקות)
    model_name = "yolo26m.pt"
    print(f"[Local] Loading base architecture: {model_name}...")
    model = YOLO(model_name)

    # 3. אימון מותאם לחומרה מקומית (8GB VRAM)
    print("[Local] Starting training on local GPU...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=100,               # פחות epochs לבדיקה ראשונית של התכנסות
        batch=8,                 # גודל בטוח למניעת שגיאות CUDA Out of Memory
        imgsz=640,
        workers=2,               # יציב עבור Windows
        device=0,                # GPU מקומי
        amp=True,                # ניצול FP16 לחיסכון בזיכרון
        patience=10,
        save=True,
        project=r"D:/temp/models/pcb_train",
        name="local_pcb_exp",
        exist_ok=True
    )

    best_model_path = Path(r"D:/temp/models/pcb_train/local_pcb_exp/weights/best.pt")
    print(f"[Local] Training completed! Best weights: {best_model_path}")

if __name__ == "__main__":
    main()