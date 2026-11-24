"""FastAPI gateway for Sightengine AI image and short-video detection."""

from __future__ import annotations

import os
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

SIGHTENGINE_API_USER = os.getenv("SIGHTENGINE_API_USER", "")
SIGHTENGINE_API_SECRET = os.getenv("SIGHTENGINE_API_SECRET", "")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime"}
IMAGE_ENDPOINT = "https://api.sightengine.com/1.0/check.json"
VIDEO_ENDPOINT = "https://api.sightengine.com/1.0/video/check-sync.json"

app = FastAPI(title="Diffusion Enhanced Deepfake Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def configured() -> bool:
    return bool(SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET)


def verdict(scores: list[float], media_type: str) -> dict:
    score = sum(scores) / len(scores)
    if score >= 0.80:
        decision = "likely_ai_generated"
        explanation = "The external detection service found strong AI-generation evidence."
    elif score <= 0.20:
        decision = "likely_real"
        explanation = "The external detection service found stronger real-media evidence."
    else:
        decision = "uncertain"
        explanation = "The detector found mixed evidence; manual review is recommended."
    return {
        "decision": decision,
        "ai_probability": round(score, 4),
        "confidence": round(max(score, 1 - score), 4),
        "media_type": media_type,
        "frames_analyzed": len(scores) if media_type == "video" else 1,
        "frame_score_range": [round(min(scores), 4), round(max(scores), 4)],
        "explanation": explanation,
        "provider": "Sightengine",
    }


def response_scores(payload: dict, is_video: bool) -> list[float]:
    if not is_video:
        score = payload.get("type", {}).get("ai_generated")
        if isinstance(score, (int, float)):
            return [float(score)]
    else:
        frames = payload.get("data", {}).get("frames", [])
        scores = [
            float(frame["type"]["ai_generated"])
            for frame in frames
            if isinstance(frame.get("type", {}).get("ai_generated"), (int, float))
        ]
        if scores:
            return scores
    raise HTTPException(status_code=502, detail="The detection provider returned no usable AI score.")


def analyze(content: bytes, filename: str, content_type: str, is_video: bool) -> dict:
    endpoint = VIDEO_ENDPOINT if is_video else IMAGE_ENDPOINT
    try:
        response = requests.post(
            endpoint,
            files={"media": (filename, content, content_type)},
            data={
                "models": "genai",
                "api_user": SIGHTENGINE_API_USER,
                "api_secret": SIGHTENGINE_API_SECRET,
            },
            timeout=120,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail="The external detection service could not be reached. Please retry.") from error
    try:
        payload = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="The detection provider returned an invalid response.") from error
    if not response.ok or payload.get("status") != "success":
        error_data = payload.get("error", {})
        message = error_data.get("description") or error_data.get("message") or "The detection provider rejected this media."
        raise HTTPException(status_code=502, detail=message)
    return verdict(response_scores(payload, is_video), "video" if is_video else "image")


@app.get("/health")
def health() -> dict:
    return {
        "provider": "Sightengine",
        "configured": configured(),
        "message": "Ready" if configured() else "Set SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET in Render.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not configured():
        raise HTTPException(status_code=503, detail="Sightengine credentials are not configured on the server.")
    if file.content_type not in IMAGE_TYPES | VIDEO_TYPES:
        raise HTTPException(status_code=415, detail="Use JPG, PNG, WEBP, MP4, WEBM, or MOV media.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file must be 100 MB or smaller.")
    return analyze(content, file.filename or "upload", file.content_type, file.content_type in VIDEO_TYPES)


app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="frontend")
