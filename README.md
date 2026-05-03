# Blood Group Detection using Fingerprints



## 🎯 Overview

This repository contains a **hybrid deep‑learning model** that combines **EfficientNetV2‑S** and **ConvNeXt‑small** to classify blood groups from fingerprint images. The model achieves **96.24% accuracy** on the test set and includes Grad‑CAM visualisations for explainable predictions.

---

## ✨ Features

- **Hybrid Architecture** – EfficientNetV2‑S for high‑level feature extraction + ConvNeXt‑small for fine‑grained details.
- **State‑of‑the‑art Accuracy** – 96.24% on validation data.
- **Explainable AI** – Integrated Grad‑CAM heatmaps to visualise decision regions.
- **Cross‑Platform** – Exportable as a Flask API, desktop app (Tkinter) and mobile Flutter front‑end.
- **GPU‑accelerated training** – Uses PyTorch with mixed‑precision (`torch.cuda.amp`).

---

## 📦 Repository Structure

```
├── data/                     # Sample fingerprint dataset (excluded from Git)
├── models/                   # Saved model weights (.pth) and config files
│   └── hybrid_effnet_convnext.pth
├── notebooks/                # Exploration & training notebooks
├── src/                      # Core Python package
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset implementation
│   ├── model.py              # Hybrid model definition
│   ├── train.py              # Training script
│   ├── inference.py          # Inference utilities
│   └── visualisation.py      # Grad‑CAM helpers
├── api/                      # Flask web‑service entry‑point
│   ├── app.py                # Flask app
│   └── requirements.txt      # Dependencies for API
├── README.md                 # <‑ **THIS FILE**
└── LICENSE                  # MIT License
```

---

## 🛠️ Installation

> **Prerequisites** – Python 3.10+, PyTorch 2.2+, CUDA 12 (optional for GPU)

```bash
# Clone the repository
git clone "https://github.com/yaswanthKumar44/Blood-Group-Classification-useing-Finger-Prints-Hybrid-Model-EfficientNet-V2S-and-ConvNeXt-.git"
cd blood‑group‑fingerprint‑classifier

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r src/requirements.txt

# (Optional) Install API dependencies
pip install -r api/requirements.txt
```

---

## 🚀 Quick Start – Inference

```python
>>> from src.inference import predict_blood_group
>>> img_path = "samples/fingerprint_01.jpg"
>>> label, heatmap = predict_blood_group(img_path)
>>> print(f"Predicted blood group: {label}")
>>> heatmap.save("output/heatmap_01.png")
```

The function returns the predicted class (`"A+"`, `"B-"`, …) and a **Grad‑CAM** heatmap image.

---

## 📈 Training the Model

1. **Prepare the dataset** – Place your fingerprint images under `data/` following the structure `data/<class_name>/*.jpg`.
2. **Edit `src/config.yaml`** – Adjust paths, batch size, learning‑rate, and number of epochs.
3. **Run the training script**:
   ```bash
   python src/train.py --config src/config.yaml
   ```

   Check `runs/` for TensorBoard logs and the exported `.pth` checkpoint.

---

## 🌐 Deploying the Flask API

```bash
cd api
pip install -r requirements.txt
python app.py
```

The service will be reachable at `http://127.0.0.1:5000/predict`. Use a tool like **Postman** or **curl**:

```bash
curl -X POST -F "file=@samples/fingerprint_02.jpg" http://127.0.0.1:5000/predict
```

The JSON response contains the predicted blood group and a base64‑encoded Grad‑CAM overlay.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/your-feature`).
3. Ensure code passes linting (`flake8`) and tests (`pytest`).
4. Open a Pull Request with a clear description of changes.

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 📧 Contact

**Author:** Yaswanth Kumar Peddagamalla , Lakireddy Bali Reddy College of Engineering
**Email:**  [yashyaswanth714@gmail.com](mailto:your.email@example.com)

---

*Happy coding!*
