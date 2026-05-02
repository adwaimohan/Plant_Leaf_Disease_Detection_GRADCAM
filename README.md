# 🌿 CropGuard AI — Crop Disease Detection

An end-to-end deep learning system for detecting crop diseases from leaf images, powered by **EfficientNet-B0** with **GradCAM explainability**. Upload a photo of a plant leaf and get an instant diagnosis along with a visual heatmap showing exactly where the model is looking.

---

## ✨ Features

- **15-Class Disease Detection** — Covers Tomato, Potato, and Pepper Bell diseases
- **GradCAM Heatmaps** — Visual explanation of model predictions highlighting affected regions
- **Modern Web UI** — Dark-themed drag-and-drop interface with real-time analysis
- **REST API** — FastAPI backend with Swagger documentation
- **CPU & GPU Support** — Runs on any machine with automatic device detection

---

## 🖼️ Supported Classes

| Crop | Diseases |
|------|----------|
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Pepper Bell** | Bacterial Spot, Healthy |

---

## 📁 Project Structure

```
crop_disease/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI server + frontend serving
├── configs/
│   └── classes.json          # Class label definitions
├── data/
│   ├── train/                # Training dataset
│   ├── val/                  # Validation dataset
│   └── test/                 # Test dataset
├── frontend/
│   ├── index.html            # Web UI
│   ├── style.css             # Styling (dark glassmorphism theme)
│   └── app.js                # Upload + results rendering logic
├── models/
│   └── disease_detection_model.pth  # Trained model weights
├── notebooks/                # Jupyter notebooks for experimentation
├── src/
│   ├── model.py              # EfficientNet-B0 model builder & loader
│   ├── predict.py            # Inference logic
│   ├── explain.py            # GradCAM implementation
│   ├── transforms.py         # Image preprocessing transforms
│   ├── train.py              # Training script
│   └── utils.py              # Validation & formatting helpers
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/crop-disease-detection.git
cd crop-disease-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Open in Browser

Navigate to **[http://localhost:8000](http://localhost:8000)** to use the web UI.

- Upload a leaf image (JPG, PNG, BMP, or WebP)
- Click **"Analyze Leaf"**
- View the diagnosis, confidence scores, and GradCAM heatmap

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model status & health check |
| `GET` | `/docs` | Swagger interactive API docs |
| `POST` | `/predict` | Upload image → top-5 disease predictions |
| `POST` | `/explain` | Upload image → prediction + GradCAM heatmap |

### Example: Predict via cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "filename": "leaf_image.jpg",
  "predictions": [
    { "rank": 1, "class": "Tomato_Early_blight", "confidence": "87.23%" },
    { "rank": 2, "class": "Tomato_Late_blight", "confidence": "6.41%" },
    { "rank": 3, "class": "Tomato_Septoria_leaf_spot", "confidence": "3.12%" }
  ]
}
```

### Example: Explain via cURL

```bash
curl -X POST http://localhost:8000/explain \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "filename": "leaf_image.jpg",
  "predicted_class": "Tomato_Early_blight",
  "confidence": 0.8723,
  "gradcam_image": "<base64-encoded PNG>"
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 |
| Input Size | 224 × 224 |
| Classes | 15 |
| Framework | PyTorch |
| Explainability | GradCAM (last convolutional layer) |

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Deep Learning:** PyTorch, TorchVision
- **Explainability:** GradCAM (custom implementation)
- **Image Processing:** Pillow, OpenCV
- **Frontend:** Vanilla HTML/CSS/JS

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
