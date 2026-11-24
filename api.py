"""FastAPI service for real image and frame-based video deepfake inference."""

from __future__ import annotations

import os
import io
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from data import IMAGENET_MEAN, IMAGENET_STD
from model import DiffusionEnhancedDetector

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "outputs/best.pt"))
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_FRAMES = 24
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class Detector:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DiffusionEnhancedDetector | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not CHECKPOINT_PATH.is_file():
            return
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
        model = DiffusionEnhancedDetector(pretrained_backbone=False).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model

    def predict(self, images: list[Image.Image]) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model is not ready.")
        batch = torch.stack([PREPROCESS(image.convert("RGB")) for image in images]).to(self.device)
        with torch.no_grad():
            real_probabilities = torch.sigmoid(self.model(batch)).cpu().tolist()
        # ImageFolder assigns fake=0 and real=1, so expose fake probability.
        return [1.0 - probability for probability in real_probabilities]


detector = Detector()


@asynccontextmanager
async def lifespan(_: FastAPI):
    detector.load()
    yield


app = FastAPI(title="Diffusion Enhanced Deepfake Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def sample_video_frames(path: Path, count: int = MAX_VIDEO_FRAMES) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="The video could not be read.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        capture.release()
        raise HTTPException(status_code=400, detail="The video has no readable frames.")
    positions = [round(index * (total - 1) / max(min(count, total) - 1, 1)) for index in range(min(count, total))]
    frames: list[Image.Image] = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()
        if success:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        raise HTTPException(status_code=400, detail="No usable frames were found in the video.")
    return frames


def classify(probabilities: list[float], media_type: str) -> dict:
    ai_probability = sum(probabilities) / len(probabilities)
    if ai_probability >= 0.60:
        decision = "likely_ai_generated"
        explanation = "The trained model found AI-generation patterns in the submitted media."
    elif ai_probability <= 0.40:
        decision = "likely_real"
        explanation = "The trained model found stronger evidence of a naturally captured image."
    else:
        decision = "uncertain"
        explanation = "The model is not confident enough to classify this media. Manual review is recommended."
    return {
        "decision": decision,
        "ai_probability": round(ai_probability, 4),
        "confidence": round(max(ai_probability, 1 - ai_probability), 4),
        "media_type": media_type,
        "frames_analyzed": len(probabilities) if media_type == "video" else 1,
        "frame_score_range": [round(min(probabilities), 4), round(max(probabilities), 4)],
        "explanation": explanation,
    }


@app.get("/health")
def health() -> dict:
    return {
        "model_ready": detector.ready,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "message": "Ready" if detector.ready else "Checkpoint not found. Train the model and place best.pt in outputs/.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not detector.ready:
        raise HTTPException(status_code=503, detail="Model checkpoint is unavailable. Train the model before requesting predictions.")
    if file.content_type not in IMAGE_TYPES | VIDEO_TYPES:
        raise HTTPException(status_code=415, detail="Use JPG, PNG, WEBP, MP4, WEBM, or MOV media.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file must be 100 MB or smaller.")
    is_video = file.content_type in VIDEO_TYPES
    if not is_video:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except UnidentifiedImageError as error:
            raise HTTPException(status_code=400, detail="The image could not be read.") from error
        return classify(detector.predict([image]), "image")

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    try:
        return classify(detector.predict(sample_video_frames(temp_path)), "video")
    finally:
        temp_path.unlink(missing_ok=True)


app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="frontend")
