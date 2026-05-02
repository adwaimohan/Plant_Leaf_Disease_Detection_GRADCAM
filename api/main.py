from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model
from src.predict import predict_from_bytes
from src.explain import explain_from_bytes
from src.utils import validate_image, format_predictions

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = FastAPI(
    title="Crop Disease Detection API",
    description="EfficientNet-B0 based crop disease detection with GradCAM explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load model once at startup
print("Loading model...")
model, class_names, device = load_model(
    model_path="models/disease_detection_model.pth",
    classes_path="configs/classes.json"
)
print("Model ready!")


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "EfficientNet-B0",
        "num_classes": len(class_names),
        "device": str(device)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validate file type
    try:
        validate_image(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    # read image bytes
    image_bytes = await file.read()

    # run prediction
    try:
        predictions = predict_from_bytes(image_bytes, model, class_names, device)
        formatted = format_predictions(predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {
        "filename": file.filename,
        "predictions": formatted
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    # validate file type
    try:
        validate_image(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    # read image bytes
    image_bytes = await file.read()

    # run gradcam explanation
    try:
        result = explain_from_bytes(image_bytes, model, class_names, device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

    return {
        "filename": file.filename,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "gradcam_image": result["gradcam_image"]
    }


# Mount frontend static files (CSS, JS) — must be AFTER API routes
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")