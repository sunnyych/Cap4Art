import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from transformers import ViTConfig, ViTModel, AutoImageProcessor, ViTForImageClassification
import wandb
from glob import glob

# ========== Setup ==========
save_dir = './semart_type'
cache_dir = '/juice2/scr2/syu03'
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model_convnext.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with your actual emotion labels
# EMOTION_LIST = [
#     'amusement', 'anger', 'awe', 'contentment', 'disgust',
#     'excitement', 'fear', 'sadness', 'something else'
# ]
EMOTION_LIST = [
   'religious', 'portrait', 'landscape', 'mythological', 'genre', 'still-life', 'historical', 'other', 'interior', 'study'
]

# ========== Load Model ==========

config = ViTConfig.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(EMOTION_LIST)
)
model = ViTForImageClassification(config)
model.load_state_dict(torch.load(best_model_path, map_location=device))

# Load corresponding image processor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir)

model.to(device)
model.eval()

# ========== Run Inference ==========
test_image_paths = sorted(glob("artpedia_images/*.png"))
results = []

for img_path in tqdm(test_image_paths, desc="Classifying test images"):
    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  # Resize if needed
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        continue

    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class_idx = logits.argmax(dim=1).item()
        pred_label = EMOTION_LIST[pred_class_idx]

        results.append({
            "image": os.path.basename(img_path),
            "prediction": pred_label
        })

        print(f"{os.path.basename(img_path)} ➜ {pred_label}")

# ========== Save to CSV ==========
df_preds = pd.DataFrame(results)
df_preds.to_csv("TEST_RESULTS.csv", index=False)
print("Inference complete. Results saved to test_predictions_small.csv")
