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
Image.LOAD_TRUNCATED_IMAGES = True
# try smaller models
# CNN, resnet for classification
# fine-tune
# for milestone try different models and see which ones perform best
os.environ["TORCH_HOME"] = "/juice2/scr2/syu03/torch_cache"
os.environ["TRANSFORMERS_CACHE"] = "/juice2/scr2/syu03/huggingface"
os.environ["TRITON_CACHE_DIR"] = "/juice2/scr2/syu03/triton_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/juice2/scr2/syu03/huggingface"

# ===============================
# Step 1: Load and Filter Dataset
# ===============================
# Load the dataset
df_artemis_processed = pd.read_csv("artemis_images_with_links_small.csv")
cache_dir = "/juice2/scr2/syu03"
wandb.init(
    project="vit-emotion-finetune_full",
    config={
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "model": "google/vit-base-patch16-224-in21k",
        "dataset": "artemis",
    }
)


def is_valid_image(path):
    try:
        img = Image.open(path)
        img = img.convert('RGB')  # Force convert
        img = img.resize((224, 224))  # Resize once to check
        img.verify()  # Quick integrity check
        return True
    except Exception:
        return False



# Only keep rows that have a valid link and corresponding image exists
img_dir = 'images/'
df_artemis_processed['img_path'] = df_artemis_processed['painting'].apply(lambda x: os.path.join(img_dir, x + '.jpg'))
df_artemis_processed = df_artemis_processed[df_artemis_processed['img_path'].apply(lambda x: os.path.exists(x))].reset_index(drop=True)
# df_artemis_processed = df_artemis_processed[:1000]  # Limit to 1000 samples for testing

# Define emotion -> class ID mapping

# EMOTION_LIST = [
#     'amusement', 'anger', 'awe', 'contentment', 'disgust',
#     'excitement', 'fear', 'sadness', 'something else'
# ]
EMOTION_LIST = ["Impressionism", "Northern_Renaissance", "Post_Impressionism", "Expressionism", "Abstract_Expressionism", 
"Romanticism", "Symbolism", "Symbolism", "Naive_Art_Primitivism", "Cubism", "Realism", "Minimalism", "Baroque", "Art_Nouveau_Modern", 
"Pop_Art", "Rococo", "Early_Renaissance", "Contemporary_Realism", "Color_Field_Painting", "Ukiyo_e", "Mannerism_Late_Renaissance", "High_Renaissance",
"New_Realism", "Fauvism", "Action_painting", "Synthetic_Cubism", "Analytical_Cubism"]


EMOTION2IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_LIST)}

# ============================
# Step 2: Dataset and Dataloader
# ============================

# Custom Dataset
class PaintingEmotionDataset(Dataset):
    def __init__(self, df, image_processor):
        self.data = df
        self.processor = image_processor

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
    #     image = Image.open(row['img_path']).convert("RGB")
    #     label = EMOTION2IDX[row['emotion']]

    #     # Apply processor here
    #     processed = self.processor(images=image, return_tensors="pt")
    #     pixel_values = processed['pixel_values'].squeeze(0)  # Remove batch dim

    #     return pixel_values, label
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            image = Image.open(row['img_path'])

            # Force convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Standardize size (especially if some images are weird sizes)
            image = image.resize((224, 224))

        except Exception as e:
            print(f"Error loading image {row['img_path']}: {e}")
            # Optionally raise an error or return a dummy image
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))  # Black image
            image = dummy_image

        label = EMOTION2IDX[row['art_style']]

        # Process image into model input
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed['pixel_values'].squeeze(0)  # (3,224,224)

        return pixel_values, label


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
    stratify=df_artemis_processed['art_style']  # Stratify by art style
)

# Then split train+val into train and val
df_train, df_val = train_test_split(
    df_temp,
    test_size=0.2,  # 20% of temp (~18% total)
    random_state=42,
    stratify=df_temp['art_style']
)

# Create Dataset objects
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=cache_dir)

train_dataset = PaintingEmotionDataset(df_train, image_processor)
val_dataset = PaintingEmotionDataset(df_val, image_processor)
test_dataset = PaintingEmotionDataset(df_test, image_processor)

# print the sizes of the datasets
print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

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
# model = timm.create_model('swin_tiny_patch4_window7_224', cache_dir = cache_dir,pretrained=True, num_classes=len(EMOTION_LIST))
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(EMOTION_LIST),
    ignore_mismatched_sizes=True,  # Allows resizing the classification head
    cache_dir=cache_dir
)
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Paths
save_dir = './checkpoints_style'
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, 'best_model.pth')

# ==================
# Step 5: Training Loop
# ==================

best_val_acc = 0.0
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.logits.max(1)
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
            # outputs = model(images)
            # loss = criterion(outputs, labels)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = outputs.logits.max(1)
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


    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    scheduler.step()

# Safer save directory
trained_model_dir = "./trained_model_small"
os.makedirs(trained_model_dir, exist_ok=True)

# Save full model safely
model_path = os.path.join(trained_model_dir, "vit_style_model.pt")
torch.save(model, model_path)
print(f"Model saved to {model_path}")

print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

# ==================
# Step 6: Test the Best Model
# ==================

print("🔎 Evaluating best model on test set...")
# model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.logits.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")
wandb.log({"test_acc": test_acc})


# ================== RUN INFERENCE

test_image_paths = sorted(glob("test_images/*.png"))
model.eval()

results = []

for img_path in tqdm(test_image_paths, desc="Classifying test images"):
    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        continue

    # Preprocess and move to device
    with torch.no_grad():
        # inputs = image_processor(images=image, return_tensors="pt").to(device)
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

# Save results to CSV
df_preds = pd.DataFrame(results)
df_preds.to_csv("style.csv", index=False)
print("Inference complete. Results saved to test_predictions.csv")






   