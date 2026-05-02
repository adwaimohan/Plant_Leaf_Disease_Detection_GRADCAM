import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64
from src.transforms import get_val_transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor, class_idx):
        self.model.zero_grad()

        output = self.model(tensor)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)

        # pool gradients across channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # weight the activations
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def explain_from_bytes(image_bytes, model, class_names, device):
    # load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)

    # preprocess
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # get prediction first
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze()
        predicted_idx = probs.argmax().item()
        confidence = round(probs[predicted_idx].item(), 4)
        predicted_class = class_names[predicted_idx]

    # run gradcam
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(tensor, predicted_idx)

    # overlay heatmap on image
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.5, heatmap_rgb, 0.5, 0)

    # convert to base64 so API can return it as JSON
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "gradcam_image": encoded
    }