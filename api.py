"""FastAPI service using a pretrained AI-image detector and video-frame sampling."""

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
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# A 25M-parameter ResNet-50 detector. It is deliberately fixed to this model
# so the service always fits within Render Free's 512 MB RAM, even if an older
# MODEL_ID setting remains in the Render dashboard.
MODEL_ID = "Imagers/Imager-1-tiny"
MODEL_WEIGHTS = "model/model_imager-1-tiny.pth"
AI_DECISION_THRESHOLD = 0.80
REAL_DECISION_THRESHOLD = 0.20
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_FRAMES = 12
INFERENCE_BATCH_SIZE = 4
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}
class Detector:
    def __init__(self) -> None:
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = None
        self.ai_class_index = 1

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        # The saved checkpoint contains a two-class ResNet state dictionary.
        # Its source files describe an older single-logit variant, so use the
        # actual state-dictionary layout reported by the checkpoint itself.
        checkpoint = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_WEIGHTS)
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(model.fc.in_features, 2),
        )
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_data["model"])
        self.ai_class_index = self._find_ai_class_index(checkpoint_data.get("class_mapping"))
        model.eval()
        self.model = model

    @staticmethod
    def _find_ai_class_index(class_mapping: object) -> int:
        if not isinstance(class_mapping, dict):
            return 1
        for key, value in class_mapping.items():
            if str(value).lower() in {"ai", "fake", "ai-generated", "ai_generated"}:
                return int(key)
            if str(key).lower() in {"ai", "fake", "ai-generated", "ai_generated"}:
                return int(value)
        return 1

    def predict(self, images: list[Image.Image]) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model is not ready.")
        scores: list[float] = []
        # Videos are processed in small batches so a long upload cannot exhaust
        # the limited RAM of a free CPU web service.
        for start in range(0, len(images), INFERENCE_BATCH_SIZE):
            batch = torch.stack([
                self.preprocess(image.convert("RGB"))
                for image in images[start:start + INFERENCE_BATCH_SIZE]
            ])
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(batch), dim=1)[:, self.ai_class_index].tolist()
            scores.extend(probabilities)
        return scores


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
    # Use a deliberately conservative decision band. The lightweight model can
    # overreact to compression, editing, or unusual lighting in real photos;
    # scores in the middle are reported as uncertain rather than false claims.
    if ai_probability >= AI_DECISION_THRESHOLD:
        decision = "likely_ai_generated"
        explanation = "The trained model found AI-generation patterns in the submitted media."
    elif ai_probability <= REAL_DECISION_THRESHOLD:
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
        "model": MODEL_ID,
        "message": "Ready" if detector.ready else "Pretrained model is loading or unavailable.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not detector.ready:
        raise HTTPException(status_code=503, detail="The pretrained detector is still loading. Please retry shortly.")
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
