# Import necessary libraries
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor

# =========================================================
# 1. Relative Paths (Linux / VS Code Environment)
# =========================================================

# Path to the downloaded checkpoint directory under the workspace
model_dir = "./vaultlayer-results/d54bc972-8161-4c97-888c-993c8c10d2ec"

# Relative path to the test dataset folder
test_dir = "./cards_dataset/test"

# =========================================================
# 2. Load Model, Processor & Extract Classes
# =========================================================

# Load fine-tuned ViT model from local weights
model = ViTForImageClassification.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Base ViT processor
image_processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Extract class names automatically using ImageFolder
test_dataset = ImageFolder(test_dir)
class_names = test_dataset.classes
print(f"--> Total Classes: {len(class_names)}")

# =========================================================
# 3. Sample 6 Random Images from Test Dataset
# =========================================================

all_images = [
    (os.path.join(root, file), os.path.basename(root))
    for root, _, files in os.walk(test_dir)
    for file in files
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

random_image_paths = random.sample(all_images, 6)

# =========================================================
# 4. Predict & Visualize (2x3 Grid)
# =========================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for idx, (image_path, true_label) in enumerate(random_image_paths):
    sample_image = Image.open(image_path).convert("RGB")
    processed_sample = image_processor(sample_image, return_tensors="pt").to(
        device
    )

    with torch.no_grad():
        outputs = model(**processed_sample)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = class_names[predicted_class]

    # Green title for correct match, Red for mismatch
    title_color = "green" if true_label == predicted_label else "red"

    axes[idx].imshow(sample_image)
    axes[idx].axis("off")
    axes[idx].set_title(
        f"True: {true_label}\nPred: {predicted_label}",
        fontsize=11,
        color=title_color,
    )

plt.tight_layout()
plt.show()