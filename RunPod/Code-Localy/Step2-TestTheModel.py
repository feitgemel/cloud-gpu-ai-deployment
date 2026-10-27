import glob
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==========================================
# 1. Configuration & Paths
# ==========================================
# Path to the trained checkpoint
WEIGHTS_PATH = Path(r"D:/Temp/Models/pcb_train/local_pcb_exp/weights/best.pt")

# Directory containing test images (supports jpg, png, webp)
TEST_IMAGES_DIR = Path(r"GPU - On-line Websites/RunPod/Code-Localy/test_images")

# Minimum detection confidence threshold
CONF_THRESHOLD = 0.40

# Non-Maximum Suppression (NMS) IoU threshold
IOU_THRESHOLD = 0.45

# Supported image file extensions
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"]


def run_pcb_inference():
    # Verify model weights file existence
    if not WEIGHTS_PATH.exists():
        print(f"Error: Weights file not found at {WEIGHTS_PATH}")
        return

    # Load the fine-tuned YOLO model
    print(f"Loading trained model from: {WEIGHTS_PATH}...")
    model = YOLO(str(WEIGHTS_PATH))

    # Collect test images including .webp from directory and subdirectories
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

    print(f"Found {len(test_images)} test images (including WebP). Starting evaluation...")
    print("Controls:")
    print(" - Press any key (except 'q' or ESC) to display next image")
    print(" - Press 'q' or 'ESC' to exit viewer\n")

    for idx, img_path in enumerate(test_images, start=1):
        # Execute forward pass inference on the image file
        results = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            device=0,  # Local NVIDIA GPU
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # Output detection details to console
        print(
            f"[{idx}/{len(test_images)}] File: {Path(img_path).name} | Detections: {len(boxes)}"
        )
        if len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                cls_name = model.names[cls_id]
                conf = float(b.conf[0].item())
                xyxy = [round(x, 1) for x in b.xyxy[0].tolist()]
                print(
                    f"   -> [{cls_name}] (Confidence: {conf:.2%}) BBox: {xyxy}"
                )
        else:
            print("   -> Clean board / No defects identified above threshold.")

        # Generate annotated image
        annotated_frame = result.plot(line_width=2, font_size=0.5)

        # Render output in a resizable OpenCV GUI window
        window_title = f"PCB Defect Detection - {Path(img_path).name}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 900, 900)
        cv2.imshow(window_title, annotated_frame)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)

        # Break loop on 'q' or ESC key press
        if key == ord("q") or key == 27:
            print("\nEvaluation loop interrupted by user.")
            break

    cv2.destroyAllWindows()
    print("Testing session finished.")


if __name__ == "__main__":
    run_pcb_inference()