# CLIP image encoder + classification head for emotion classification
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import wandb

Image.LOAD_TRUNCATED_IMAGES = True

# ===============================
# Step 1: Load and Filter Dataset
# ===============================
df_artemis_processed = pd.read_csv("artemis_images_with_links.csv")
cache_dir = "/juice2/scr2/syu03"

wandb.init(
    project="clip-emotion-finetune_clip",
    config={
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "model": "openai/clip-vit-base-patch32",
        "dataset": "artemis",
    }
)

img_dir = 'images/'
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except (IOError, SyntaxError):
        return False

df_artemis_processed['img_path'] = df_artemis_processed['painting'].apply(lambda x: os.path.join(img_dir, x + '.jpg'))
df_artemis_processed = df_artemis_processed[df_artemis_processed['img_path'].apply(lambda x: os.path.exists(x) and is_valid_image(x))].reset_index(drop=True)

EMOTION_LIST = [
    'amusement', 'anger', 'awe', 'contentment', 'disgust',
    'excitement', 'fear', 'sadness', 'something else'
]
EMOTION2IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}

# ============================
# Step 2: Dataset and Dataloader
# ============================
class PaintingEmotionDataset(Dataset):
    def __init__(self, df, processor):
        self.data = df
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['img_path']).convert("RGB")
        label = EMOTION2IDX[row['emotion']]

        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)

        return pixel_values, label

# ==========================
# Step 3: Split
# ==========================
df_temp, df_test = train_test_split(df_artemis_processed, test_size=0.1, random_state=42, stratify=df_artemis_processed['emotion'])
df_train, df_val = train_test_split(df_temp, test_size=0.2, random_state=42, stratify=df_temp['emotion'])

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)

train_dataset = PaintingEmotionDataset(df_train, processor)
val_dataset   = PaintingEmotionDataset(df_val, processor)
test_dataset  = PaintingEmotionDataset(df_test, processor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ==========================
# Step 4: Model
# ==========================
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.fc = nn.Linear(clip_model.config.vision_config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
model = CLIPClassifier(clip_model, num_classes=len(EMOTION_LIST)).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# ==========================
# Step 5: Training Setup
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
save_dir = './checkpoints'; os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model_clip.pth')

# ==================
# Step 6: Training Loop
# ==================

best_val_acc = 0.0
EPOCHS = 30
device = next(model.parameters()).device

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    scheduler.step()

# ==================
# Step 7: Test Evaluation
# ==================
print("\nEvaluating best model on test set...")
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_correct, test_total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")
wandb.log({"test_acc": test_acc})
