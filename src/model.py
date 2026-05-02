import torch
import torch.nn as nn
from torchvision import models
import json
import os

def load_class_names(classes_path="configs/classes.json"):
    with open(classes_path,"r") as f:
        data=json.load(f)
    return data["classes"]


def build_model(num_classes):
    model=models.efficientnet_b0(weights=None)
    model.classifier[1]=nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


def load_model(model_path="models/disease_detection_model.pth",classes_path="configs/classes.json"):
    class_names=load_class_names(classes_path)
    num_classes=len(class_names)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(num_classes)

    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded successfully with {num_classes} classes.")
    return model,class_names,device


