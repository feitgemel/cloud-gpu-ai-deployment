# Import libraries
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# 1. Fully Dynamic Path Discovery & Fail-safe Audit
# =========================================================

import os

train_dir = None
valid_dir = None
test_dir = None

search_root = "."

print("=== Starting Workspace Scan for Dataset ===", flush=True)

# Walk through ALL directories recursively starting from current working dir
for root, dirs, files in os.walk(search_root):
    # Skip checkpoints/git to avoid infinite loops or overhead
    if "checkpoints" in root or ".git" in root or "__pycache__" in root:
        continue
        
    if "train" in dirs and "valid" in dirs and "test" in dirs:
        train_dir = os.path.join(root, "train")
        valid_dir = os.path.join(root, "valid")
        test_dir = os.path.join(root, "test")
        print(f"--> SUCCESS! Found 'train' at: {train_dir}", flush=True)
        print(f"--> SUCCESS! Found 'valid' at: {valid_dir}", flush=True)
        print(f"--> SUCCESS! Found 'test'  at: {test_dir}", flush=True)
        break

# If not found anywhere, print EVERY SINGLE folder in the server for total transparency
if not train_dir or not valid_dir or not test_dir:
    print("\n================ ERROR DIAGNOSTICS ================", flush=True)
    print("Could not find 'train', 'valid', and 'test' together.")
    print("Here is the FULL directory tree on this GPU instance:")
    for root, dirs, files in os.walk(search_root):
        # Print directory path and count of files/dirs
        print(f" Directory: {root} | Subfolders: {dirs}", flush=True)
    print("===================================================\n", flush=True)
    raise FileNotFoundError("Dataset structure ('train', 'valid', 'test') was NOT mounted on this instance.")

# Print dataset classes summary once found
print("\n=== Dataset Directory Inspection ===", flush=True)
for name, path in [("Train", train_dir), ("Valid", valid_dir), ("Test", test_dir)]:
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"[{name} Directory]: {path}", flush=True)
    print(f"  -> Total Categories/Classes: {len(categories)}", flush=True)
    print(f"  -> First 5 Classes: {categories[:5]}", flush=True)
print("===================================\n", flush=True)

checkpoint_dir = os.environ.get("VAULTLAYER_CHECKPOINT_DIR", "./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# =========================================================
# 2. Model & Processor Setup
# =========================================================

model_id = 'google/vit-base-patch16-224-in21k'
image_processor = ViTImageProcessor.from_pretrained(model_id)

def transform(image):
    inputs = image_processor(image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)

# Load train, validation, and test datasets using ImageFolder
train_dataset = ImageFolder(train_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)
test_dataset  = ImageFolder(test_dir, transform=transform)

def collate_fn(batch):
    images, labels = zip(*batch)
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    references = p.label_ids
    return metric.compute(predictions=predictions, references=references)

num_classes = len(train_dataset.classes)
model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================================================
# 3. Training Arguments & Trainer Setup
# =========================================================

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=200,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  # Evaluation performed on 'valid' during training steps
    processing_class=image_processor,
    callbacks=[early_stopping_callback],
)

# Train & Save
train_results = trainer.train()
trainer.save_model(checkpoint_dir)
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Final Evaluation on unseen TEST dataset
test_metrics = trainer.evaluate(test_dataset)
trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)

# =========================================================
# 4. Visualization
# =========================================================

all_images = [
    (os.path.join(root, file), os.path.basename(root)) 
    for root, _, files in os.walk(test_dir) 
    for file in files if file.endswith(('.png', '.jpg', '.jpeg'))
]

if len(all_images) >= 6:
    random_image_paths = random.sample(all_images, 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (image_path, true_label) in enumerate(random_image_paths):
        sample_image = Image.open(image_path).convert("RGB")
        processed_sample = image_processor(sample_image, return_tensors="pt").to(device)
        
        outputs = model(**processed_sample)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = train_dataset.classes[predicted_class]
        
        axes[idx].imshow(sample_image)
        axes[idx].axis("off")
        axes[idx].set_title(f"True: {true_label}\nPred: {predicted_label}", fontsize=12)

    plt.tight_layout()
    visualization_path = os.path.join(checkpoint_dir, "predictions_sample.png")
    plt.savefig(visualization_path)
    print(f"Sample predictions saved to {visualization_path}")