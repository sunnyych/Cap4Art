import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification
import wandb
from glob import glob
from torch.nn.functional import softmax, log_softmax, cosine_similarity

Image.LOAD_TRUNCATED_IMAGES = True

# Environment setup
os.environ["TORCH_HOME"] = "/juice2/scr2/syu03/torch_cache"
os.environ["TRANSFORMERS_CACHE"] = "/juice2/scr2/syu03/huggingface"
os.environ["TRITON_CACHE_DIR"] = "/juice2/scr2/syu03/triton_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/juice2/scr2/syu03/huggingface"

# ===============================
# Step 1: Load and Filter Dataset
# ===============================
df = pd.read_csv("wikiart_training.csv")
cache_dir = "/juice2/scr2/syu03"
img_dir = 'images/'
df['img_path'] = df['painting'].apply(lambda x: os.path.join(img_dir, x + '.jpg'))
df = df[df['img_path'].apply(lambda x: os.path.exists(x))].reset_index(drop=True)

# Emotion labels
EMOTION_LIST = ['agreeableness', 'anger', 'anticipation', 'arrogance', 'disagreeableness', 'disgust', 'fear',
                'gratitude', 'happiness', 'humility', 'love', 'optimism', 'pessimism', 'regret', 'sadness',
                'shame', 'shyness', 'surprise', 'trust', 'neutral']

# Normalize emotion distributions
df[EMOTION_LIST] = df[EMOTION_LIST].div(df[EMOTION_LIST].sum(axis=1), axis=0)

wandb.init(
    project="vit-emotion-finetune-kl",
    config={"epochs": 10, "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01,
            "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
            "model": "google/vit-base-patch16-224-in21k", "dataset": "wikiart"}
)

# ============================
# Step 2: Dataset and Dataloader
# ============================
class PaintingEmotionDataset(Dataset):
    def __init__(self, df, image_processor):
        self.data = df
        self.processor = image_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            image = Image.open(row['img_path']).convert('RGB').resize((224, 224))
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)
        label = torch.tensor([row[e] for e in EMOTION_LIST], dtype=torch.float32)
        return pixel_values, label

# Train/val/test split
df_temp, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['majority_emotion'])
df_train, df_val = train_test_split(df_temp, test_size=0.2, random_state=42, stratify=df_temp['majority_emotion'])

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir)
train_dataset = PaintingEmotionDataset(df_train, image_processor)
val_dataset = PaintingEmotionDataset(df_val, image_processor)
test_dataset = PaintingEmotionDataset(df_test, image_processor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ==========================
# Step 3: Model, Loss, Optimizer
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=len(EMOTION_LIST),
    ignore_mismatched_sizes=True, cache_dir=cache_dir
).to(device)

for param in model.vit.parameters():
    param.requires_grad = False

criterion = nn.KLDivLoss(reduction="batchmean")
# optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ==================
# Step 4: Training Loop
# ==================
best_val_acc = 0.0
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images).logits
        log_probs = log_softmax(logits, dim=1)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += logits.argmax(1).eq(labels.argmax(1)).sum().item()
        total += labels.size(0)

    train_acc = 100. * correct / total
    val_loss, val_correct, val_total = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).logits
            log_probs = log_softmax(logits, dim=1)
            loss = criterion(log_probs, labels)
            val_loss += loss.item()
            val_correct += logits.argmax(1).eq(labels.argmax(1)).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    wandb.log({"epoch": epoch+1, "train_loss": train_loss / len(train_loader),
               "train_acc": train_acc, "val_loss": val_loss / len(val_loader), "val_acc": val_acc})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), './checkpoints_wikiart/best_model.pth')

    scheduler.step()

# ==================
# Step 5: Inference
# ==================
model.eval()
results = []
for img_path in tqdm(sorted(glob("test_images/*.png")), desc="Inference"):
    try:
        image = Image.open(img_path).convert("RGB").resize((224, 224))
    except:
        continue
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        probs = softmax(logits, dim=1).squeeze().cpu().numpy()
        dist = {e: float(p) for e, p in zip(EMOTION_LIST, probs)}
        pred_label = EMOTION_LIST[logits.argmax().item()]
        results.append({"image": os.path.basename(img_path), "prediction": pred_label, "distribution": dist})

pd.DataFrame(results).to_csv("distribution_results.csv", index=False)
