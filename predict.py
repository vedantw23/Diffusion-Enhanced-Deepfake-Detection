from __future__ import annotations

import argparse

import cv2
import torch
from PIL import Image
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD
from model import DiffusionEnhancedDetector


def score_images(model: DiffusionEnhancedDetector, images: list[Image.Image], preprocess, device: torch.device) -> list[float]:
    """Return an AI-generated probability for each RGB image.

    ImageFolder assigns fake=0 and real=1, so the network output is converted
    from real probability to deepfake probability here.
    """
    batch = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        real_probabilities = torch.sigmoid(model(batch)).cpu().tolist()
    return [1.0 - probability for probability in real_probabilities]


def sample_video_frames(video_path: str, frame_count: int) -> list[Image.Image]:
    """Read evenly spaced frames so a video can be scored by the image model."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError("The video does not contain readable frames.")
    positions = [round(index * (total_frames - 1) / max(frame_count - 1, 1)) for index in range(min(frame_count, total_frames))]
    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise ValueError("Could not extract frames from the video.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an image or sampled video frames for deepfake likelihood.")
    parser.add_argument("--checkpoint", required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to one image.")
    input_group.add_argument("--video", help="Path to a video; evenly spaced frames are analyzed.")
    parser.add_argument("--frames", type=int, default=24, help="Maximum video frames to analyze (default: 24).")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = DiffusionEnhancedDetector(pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if args.image:
        probabilities = score_images(model, [Image.open(args.image).convert("RGB")], preprocess, device)
    else:
        frames = sample_video_frames(args.video, args.frames)
        probabilities = score_images(model, frames, preprocess, device)
        print(f"frames_analyzed={len(probabilities)}")
        print(f"highest_frame_probability={max(probabilities):.4f}")
    probability = sum(probabilities) / len(probabilities)
    print(f"deepfake_probability={probability:.4f}")
    print("prediction=" + ("fake" if probability >= 0.5 else "real"))


if __name__ == "__main__":
    main()
