import torch
import torch.nn.functional as F
from src.transforms import get_val_transforms
from PIL import Image

def predict(image_path, model, class_names, device, top_k=5):
    # load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    # run inference
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze()

    # get top K predictions
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": round(prob.item(), 4)
        })

    return results


def predict_from_bytes(image_bytes, model, class_names, device, top_k=5):
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze()

    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class": class_names[idx.item()],
            "confidence": round(prob.item(), 4)
        })

    return results