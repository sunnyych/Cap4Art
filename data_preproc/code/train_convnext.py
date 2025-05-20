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
import wandb
from glob import glob
import argparse

Image.LOAD_TRUNCATED_IMAGES = True
os.environ["TORCH_HOME"] = "/juice2/scr2/syu03"
os.environ["TRANSFORMERS_CACHE"] = "/juice2/scr2/syu03/huggingface"
os.environ["TRITON_CACHE_DIR"] = "/juice2/scr2/syu03/triton_cache"


# ===============================
# Step 1: Load and Filter Dataset
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help="Path to CSV file with image paths and emotions")
parser.add_argument('--output_file', type=str, required=True, help="save path")
args = parser.parse_args()

df_artemis_processed = pd.read_csv(args.input_file)
cache_dir = "/juice2/scr2/syu03"

wandb.init(
    project="convnext-emotion-finetune",
    config={
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "model": "convnext_base.fb_in22k",
        "dataset": "artemis",
    }
)

img_dir = 'images/'
df_artemis_processed['img_path'] = df_artemis_processed['painting'].apply(lambda x: os.path.join(img_dir, x + '.jpg'))
df_artemis_processed = df_artemis_processed[df_artemis_processed['img_path'].apply(lambda x: os.path.exists(x))].reset_index(drop=True)

EMOTION_LIST = [
    'amusement', 'anger', 'awe', 'contentment', 'disgust',
    'excitement', 'fear', 'sadness', 'something else'
]
EMOTION2IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}

# ============================
# Step 2: Dataset and Dataloader
# ============================

class PaintingEmotionDataset(Dataset):
    def __init__(self, df, transform):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            image = Image.open(row['img_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {row['img_path']}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        image = self.transform(image)
        label = EMOTION2IDX[row['emotion']]
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ==========================
# Step 3: Train/Val/Test Split
# ==========================

df_temp, df_test = train_test_split(
    df_artemis_processed,
    test_size=0.1,
    random_state=42,
    stratify=df_artemis_processed['emotion']
)

df_train, df_val = train_test_split(
    df_temp,
    test_size=0.2,
    random_state=42,
    stratify=df_temp['emotion']
)

train_dataset = PaintingEmotionDataset(df_train, train_transform)
val_dataset = PaintingEmotionDataset(df_val, val_test_transform)
test_dataset = PaintingEmotionDataset(df_test, val_test_transform)

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ==========================
# Step 4: Model, Loss, Optimizer
# ==========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model(
    'convnext_base.fb_in22k',
    pretrained=True,
    num_classes=len(EMOTION_LIST)
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

save_dir = './checkpoints_convnext'
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model_convnext.pth')

# ==================
# Step 5: Training Loop
# ==================

best_val_acc = 0.0
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    for images, labels in loop:
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

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    train_acc = 100. * correct / total
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

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
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    scheduler.step()

# trained_model_dir = os.path.expanduser("~/trained_model_convnext")
trained_model_dir = "./trained_model_convnext"
os.makedirs(trained_model_dir, exist_ok=True)

model_path = os.path.join(trained_model_dir, "convnext_emotion_model.pt")
torch.save(model, model_path)
print(f"Model saved to {model_path}")
print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

# ==================
# Step 6: Test the Best Model
# ==================

print("🔎 Evaluating best model on test set...")
model.eval()

test_correct = 0
test_total = 0

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

# ==================
# Step 7: Inference on New Images
# ==================

test_image_paths = sorted(glob("test_images/*.png"))
model.eval()

results = []

for img_path in tqdm(test_image_paths, desc="Classifying test images"):
    try:
        image = Image.open(img_path).convert("RGB")
        image = val_test_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        continue

    with torch.no_grad():
        outputs = model(image)
        pred_class_idx = outputs.argmax(dim=1).item()
        pred_label = EMOTION_LIST[pred_class_idx]

        results.append({
            "image": os.path.basename(img_path),
            "prediction": pred_label
        })

        print(f"{os.path.basename(img_path)} ➜ {pred_label}")

df_preds = pd.DataFrame(results)
df_preds.to_csv(args.output_file, index=False)
print("Inference complete. Results saved to test_predictions_convnext.csv")
