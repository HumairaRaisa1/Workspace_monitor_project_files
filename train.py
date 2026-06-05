import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import mean_absolute_error

# ─── CONFIG ───────────────────────────────────────────
TRAIN_IMG_DIR  = "TrainingSet/TrainingSet"
TEST_IMG_DIR   = "TestSet/TestSet"
TEST_JSON_DIR  = "TestSet_json/TestSet_json"
BATCH_SIZE     = 32
EPOCHS         = 7
LEARNING_RATE  = 0.001
MODEL_SAVE     = "gaze_model.pth"

# ─── DATASET ──────────────────────────────────────────
class GazeDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Loop through gaze direction folders
        for label_folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, label_folder)
            if not os.path.isdir(folder_path):
                continue
            for img_file in os.listdir(folder_path):
                if not img_file.endswith(".jpg"):
                    continue
                img_path = os.path.join(folder_path, img_file)
                # Match JSON by index
                json_id = os.path.splitext(img_file)[0]
                json_path = os.path.join(json_dir, f"{json_id}.json")
                if os.path.exists(json_path):
                    self.samples.append((img_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load label (gaze vector from look_vec)
        with open(json_path) as f:
            data = json.load(f)
        look_vec = data["eye_details"]["look_vec"]
        # Parse "(x, y, z, 0)" → [x, y, z]
        vals = [float(v) for v in look_vec.strip("()").split(",")]
        label = torch.tensor(vals[:3], dtype=torch.float32)

        return image, label

# ─── TRANSFORMS ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── LOAD DATA ────────────────────────────────────────
train_dataset = GazeDataset(TRAIN_IMG_DIR, TEST_JSON_DIR, transform)
test_dataset  = GazeDataset(TEST_IMG_DIR,  TEST_JSON_DIR, transform)

from torch.utils.data import Subset
train_dataset = Subset(train_dataset, range(20000))
test_dataset  = Subset(test_dataset,  range(4000)) 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")

# ─── MODEL ────────────────────────────────────────────
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # Output: x, y, z gaze

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# ─── TRAINING ─────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

# ─── SAVE MODEL ───────────────────────────────────────
torch.save(model.state_dict(), MODEL_SAVE)
print(f"Model saved to {MODEL_SAVE} ✅")

# ─── EVALUATION ───────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

mae = mean_absolute_error(all_labels, all_preds)
print(f"Test MAE: {mae:.4f}")