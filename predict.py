from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD
from model import DiffusionEnhancedDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one image for deepfake likelihood.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DiffusionEnhancedDetector(pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(image)).item()
    print(f"deepfake_probability={probability:.4f}")
    print("prediction=" + ("fake" if probability >= 0.5 else "real"))


if __name__ == "__main__":
    main()
