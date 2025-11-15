import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.cuda import amp
from torchvision.models import convnext_tiny
import os
import requests

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if Device.type == "cuda" else torch.float32

CKPT_URL = "https://drive.google.com/uc?export=download&id=1D0IrQWjj_OSADcPFV2UilCmPZZNUKyfs"

def download_model_if_missing(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print("Downloading model from Google Drive...")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("Model download is done")

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
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model

def setup_model(ckpt_url, ckpt_path):
    download_model_if_missing(ckpt_url, ckpt_path)
    model = load_trained_model(ckpt_path)
    return model

model = setup_model(CKPT_URL, CKPT_PATH)

@torch.no_grad()
def predict_one(img_pil: Image.Image, model = model):
    img = eval_tf(img_pil).unsqueeze(0).to(Device)

    logits = model(img)

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs

