import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


data_directory = 'dataset/PlantVillage'
model_path = 'disease_detection_model.pth'


train_ds = datasets.ImageFolder(root=f'{data_directory}/train')
classes = train_ds.classes


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def generate_gradcam(model, input_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.detach().numpy()

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam


st.title("🌿 Crop Disease Detection (EfficientNet-B0 + Grad-CAM)")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_img is None:
        st.error("Invalid image")
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        st.image(cv_img, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(cv_img).unsqueeze(0)

        
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100

        st.write(f"### Prediction: {predicted_class}")
        st.write(f"### Confidence: {confidence_score:.2f}%")

        
        cam = generate_gradcam(model, input_tensor, predicted.item())

        
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        
        cam = cv2.resize(cam, (cv_img.shape[1], cv_img.shape[0]))

        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        
        cv_img_uint8 = cv_img.astype(np.uint8)
        heatmap_uint8 = heatmap.astype(np.uint8)

    
        overlay = cv2.addWeighted(cv_img_uint8, 0.6, heatmap_uint8, 0.4, 0)

        st.subheader("Grad-CAM Visualization")
        st.image(overlay, caption="Model Attention Heatmap", use_column_width=True)