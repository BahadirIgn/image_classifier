import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ------------------- AYARLAR -------------------
MODEL_PATH = "/home/eurotolia-intern/Desktop/image_classifier/best_model.pth"
IDX_TO_CLASS_PATH = "/home/eurotolia-intern/Desktop/image_classifier/idx_to_class.json"
IMAGE_PATH = "/home/eurotolia-intern/Desktop/image_classifier/test_image.jpg"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- SINIF İSİMLERİ -------------------
if not os.path.exists(IDX_TO_CLASS_PATH):
    raise FileNotFoundError(f"{IDX_TO_CLASS_PATH} bulunamadı!")

with open(IDX_TO_CLASS_PATH, "r", encoding="utf-8") as f:
    idx_to_class = json.load(f)

# JSON’dan gelmişse keyler string olabilir, int’e çevir
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# ------------------- MODEL -------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------- TRANSFORM -------------------
transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------- FOTOĞRAF YÜKLEME -------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img_t = transform(img).unsqueeze(0).to(DEVICE)

# ------------------- TAHMİN -------------------
with torch.no_grad():
    outputs = model(img_t)
    pred_idx = outputs.argmax(dim=1).item()
    pred_label = idx_to_class[pred_idx]

print("Tahmin edilen sınıf:", pred_label)
