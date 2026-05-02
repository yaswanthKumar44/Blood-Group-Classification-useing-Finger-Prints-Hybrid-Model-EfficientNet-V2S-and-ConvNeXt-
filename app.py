# =========================================================
# app.py — Fingerprint Blood Group Detection (Clean Version)
# =========================================================
import os
import time
import uuid
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from flask import Flask, render_template, request
from PIL import Image

# =========================================================
# Flask Setup
# =========================================================
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
GRAY_FOLDER = os.path.join("static", "gray")
OVERLAY_FOLDER = os.path.join("static", "overlay")

for folder in [UPLOAD_FOLDER, GRAY_FOLDER, OVERLAY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# =========================================================
# Model Definition
# =========================================================
class HybridBloodGroupNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model1 = timm.create_model("tf_efficientnetv2_s.in21k", pretrained=False, num_classes=0)
        self.model2 = timm.create_model("convnext_small", pretrained=False, num_classes=0)
        dim = self.model1.num_features + self.model2.num_features
        self.fc = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f1 = self.model1(x)
        f2 = self.model2(x)
        f = torch.cat([f1, f2], dim=1)
        return self.fc(f)

# =========================================================
# Load Model
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

model = HybridBloodGroupNet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("model/best_hybrid_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully on", device)

# =========================================================
# Preprocessing + Ridge Detection
# =========================================================
IMG_SIZE = 224
transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

def detect_ridges(gray_img, save_overlay=None):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size
    contrast = np.std(gray_img) / 255.0
    print(f"🔍 Edge density: {edge_density:.4f}, Contrast: {contrast:.4f}")

    if save_overlay:
        overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        overlay[edges > 0] = (0, 0, 255)
        cv2.imwrite(save_overlay, overlay)

    if edge_density < 0.02 or contrast < 0.05:
        return False
    return True

def preprocess_image(path, gray_save=None, overlay_save=None):
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError("Cannot read uploaded image.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if gray_save:
        cv2.imwrite(gray_save, gray)

    has_ridges = detect_ridges(gray, save_overlay=overlay_save)
    if not has_ridges:
        raise RuntimeError("❌ No ridge pattern detected in the fingerprint.")

    eq = cv2.equalizeHist(gray)
    img_rgb = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0)
    return tensor

# =========================================================
# Prediction Function
# =========================================================
def predict_blood_group(image_path, gray_save, overlay_save):
    img_tensor = preprocess_image(image_path, gray_save, overlay_save).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_class = classes[pred_idx]
        confidence = float(probs[0][pred_idx].cpu().item() * 100.0)
    return pred_class, confidence

# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and 'file' in request.files:
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="❌ No file selected.")

        fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, fname)
        gray_path = os.path.join(GRAY_FOLDER, f"gray_{fname}")
        overlay_path = os.path.join(OVERLAY_FOLDER, f"overlay_{fname}")
        file.save(file_path)

        try:
            pred_class, confidence = predict_blood_group(file_path, gray_path, overlay_path)
            result = f"🩸 Predicted Blood Group: {pred_class} ({confidence:.2f}% confidence)"
            return render_template(
                "index.html",
                result=result,
                original=fname,
                gray=os.path.basename(gray_path),
                overlay=os.path.basename(overlay_path)
            )
        except RuntimeError as e:
            return render_template("index.html", result=str(e))
        except Exception as e:
            return render_template("index.html", result=f"❌ Error: {e}")

    return render_template("index.html")

# =========================================================
# Run Flask App
# =========================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
