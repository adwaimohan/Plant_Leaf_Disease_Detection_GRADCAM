import os
import json


def validate_image(filename):
    """check the uploaded file is an image"""
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise ValueError(f"Invalid file type '{ext}'. Allowed: {allowed}")


def load_class_names(classes_path="configs/classes.json"):
    """load class names from json file"""
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"classes.json not found at {classes_path}")
    with open(classes_path, "r") as f:
        data = json.load(f)
    return data["classes"]


def format_predictions(predictions):
    """format prediction results for clean API response"""
    return [
        {
            "rank": i + 1,
            "class": p["class"],
            "confidence": f"{p['confidence'] * 100:.2f}%"
        }
        for i, p in enumerate(predictions)
    ]