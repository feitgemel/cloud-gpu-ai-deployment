import glob
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==========================================
# 1. Configuration & Paths
# ==========================================
WEIGHTS_PATH = Path(r"D:/Temp/Models/runpod/best.pt")
TEST_IMAGES_DIR = Path(r"cloud-gpu-ai-deployment/RunPod/Code-Localy/test_images")

CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"]

def run_pcb_inference():
    if not WEIGHTS_PATH.exists():
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    print(f"Loading trained model from: {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    # Collect images
    test_images = []
    for ext in IMAGE_EXTENSIONS:
        test_images.extend(glob.glob(str(TEST_IMAGES_DIR / ext)))
        test_images.extend(glob.glob(str(TEST_IMAGES_DIR / ext.upper())))
        test_images.extend(glob.glob(str(TEST_IMAGES_DIR / "images" / ext)))
        test_images.extend(glob.glob(str(TEST_IMAGES_DIR / "images" / ext.upper())))

    test_images = sorted(list(set(test_images)))

    if not test_images:
        print(f"No valid image files found in: {TEST_IMAGES_DIR}")
        return

    print(f"Found {len(test_images)} test images. Starting evaluation...")
    print("Controls:")
    print(" - Press any key (except 'q' or ESC) to display next image")
    print(" - Press 'q' or 'ESC' to exit viewer\n")

    # Create a single persistent window to avoid OpenCV GUI crashes on Windows
    window_name = "PCB Defect Detection Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)

    for idx, img_path in enumerate(test_images, start=1):
        try:
            results = model.predict(
                source=img_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=640,
                device=0,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            print(f"[{idx}/{len(test_images)}] File: {Path(img_path).name} | Detections: {len(boxes)}")
            if len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(b.conf[0].item())
                    xyxy = [round(x, 1) for x in b.xyxy[0].tolist()]
                    print(f"   -> [{cls_name}] (Confidence: {conf:.2%}) BBox: {xyxy}")
            else:
                print("   -> Clean board / No defects identified above threshold.")

            # Render output
            annotated_frame = result.plot(line_width=2, font_size=0.5)

            if annotated_frame is None or annotated_frame.size == 0:
                print(f"   [Warning] Frame rendering failed for {Path(img_path).name}. Skipping display.")
                continue

            # Display on the existing window
            cv2.imshow(window_name, annotated_frame)
            cv2.setWindowTitle(window_name, f"PCB Defect Detection - [{idx}/{len(test_images)}] {Path(img_path).name}")

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                print("\nEvaluation loop interrupted by user.")
                break

        except Exception as e:
            print(f"   [Error processing image {Path(img_path).name}]: {e}")
            continue

    cv2.destroyAllWindows()
    print("Testing session finished.")

if __name__ == "__main__":
    run_pcb_inference()