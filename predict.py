import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.cuda import amp
from torchvision.models import convnext_tiny
import os
import requests
from huggingface_hub import hf_hub_download

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if Device.type == "cuda" else torch.float32

HF_REPO_ID = "owenpha7/ConvNext_tiny"
HF_FILENAME = "ConvNext_Tiny.pt"  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # biasanya /app
CKPT_PATH = os.path.join(BASE_DIR, "weights", "ConvNext_Tiny.pt")
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class_names = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]
num_classes = len(class_names)

eval_tf = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def build_model():
    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model.to(Device)

def load_trained_model(ckpt_path):
    model = build_model()

    ckpt = torch.load(ckpt_path, map_location= Device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        print("[load_trained_model] Using ckpt['model_state']")
    else:
        # Kalau ternyata langsung state_dict
        state_dict = ckpt
        print("[load_trained_model] Using checkpoint as raw state_dict")

    model.load_state_dict(state_dict)
    model.eval()

    return model

def setup_model(repo_id, filename, local_path):
    if local_path is not None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        ckpt_path = local_path
    else:
        ckpt_path = None

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(local_path) if local_path is not None else None,
    )

    model = load_trained_model(downloaded_path)
    return model

model = setup_model(HF_REPO_ID, HF_FILENAME, CKPT_PATH)

@torch.no_grad()
def predict_one(img_pil: Image.Image, model = model):
    img = eval_tf(img_pil).unsqueeze(0).to(Device)

    logits = model(img)

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs

