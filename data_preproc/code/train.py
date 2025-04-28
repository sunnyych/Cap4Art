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

# ===============================
# Step 1: Load and Filter Dataset
# ===============================
# Load the dataset
df_artemis_processed = pd.read_csv("artemis_images_with_links.csv")

# Only keep rows that have a valid link and corresponding image exists
img_dir = '/images'
df_artemis_processed['img_path'] = df_artemis_processed['painting'].apply(lambda x: os.path.join(img_dir, x + '.jpg'))
# drop the ones with no images
df_artemis_processed = df_artemis_processed[df_artemis_processed['img_path'].apply(os.path.exists)].reset_index(drop=True)

# Define emotion -> class ID mapping

EMOTION_LIST = [
    'amusement', 'anger', 'awe', 'contentment', 'disgust',
    'excitement', 'fear', 'sadness', 'something else'
]
EMOTION2IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}

# ============================
# Step 2: Dataset and Dataloader
# ============================

# Custom Dataset
class PaintingEmotionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_path']

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        emotion = row['emotion']
        label = EMOTION2IDX[emotion]

        return image, label

# Define transforms
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

# First split into train+val and test
df_temp, df_test = train_test_split(
    df_artemis_processed,
    test_size=0.1,
    random_state=42,
    stratify=df_artemis_processed['emotion']
)

# Then split train+val into train and val
df_train, df_val = train_test_split(
    df_temp,
    test_size=0.2,  # 20% of temp (~18% total)
    random_state=42,
    stratify=df_temp['emotion']
)

# Create Dataset objects
train_dataset = PaintingEmotionDataset(df=df_train, transform=train_transform)
val_dataset   = PaintingEmotionDataset(df=df_val, transform=val_test_transform)
test_dataset  = PaintingEmotionDataset(df=df_test, transform=val_test_transform)

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ==========================
# Step 4: Model, Loss, Optimizer
# ==========================

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=len(EMOTION_LIST))
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Paths
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model.pth')

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

        # Metrics
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

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    scheduler.step()

print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

# ==================
# Step 6: Test the Best Model
# ==================

print("🔎 Evaluating best model on test set...")
model.load_state_dict(torch.load(best_model_path))
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
