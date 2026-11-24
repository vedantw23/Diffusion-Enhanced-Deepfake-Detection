# Diffusion-Enhanced Deepfake Detection

PyTorch starter project for binary real-vs-deepfake image classification. The model integrates:

- **Global features:** EfficientNet-B0 learns face-wide semantic and texture evidence.
- **Local features:** a patch encoder scores localized artifacts.
- **Diffusion enhancement:** a frozen, multi-level Gaussian-noise residual pyramid highlights generation artifacts at multiple noise levels.
- **Fusion:** gated local-global attention combines all three sources before classification.

## Real detection web app

The dashboard sends uploads to the FastAPI service in `api.py`; it does not generate results from a file name or file size. The service forwards media to the Sightengine AI-generated-content API and returns **likely AI-generated**, **likely real**, or **uncertain**.

Set `SIGHTENGINE_API_USER` and `SIGHTENGINE_API_SECRET` as private Render environment variables. Do not put these credentials in browser JavaScript or commit them to Git. The deployed API uses a conservative decision band: it labels media **likely AI-generated** only for scores at or above 80%, **likely real** only for scores at or below 20%, and reports the middle range as **uncertain**.

The original local-global architecture remains available for research and retraining. The deployed API uses the pretrained detector so it can work immediately without a custom `best.pt` checkpoint.

Install dependencies and start the app:

```powershell
pip install -r requirements.txt
uvicorn api:app --reload
```

Open `http://localhost:8000`. The API supports JPG, PNG, WEBP, MP4, WEBM, and MOV files up to 100 MB. Videos are assessed by sampling up to 12 evenly spaced frames and averaging their model scores.

For a static Vercel frontend, deploy the FastAPI service separately (the included `render.yaml` and `Dockerfile` support Render) and set `window.DETECTOR_API_URL` in `api-config.js` to that API's HTTPS URL before redeploying the frontend.

## Setup

```powershell
cd C:\Users\HP\OneDrive\Desktop\minor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset layout

```text
data/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/              # optional; same class folders
    real/
    fake/
```

Use face crops when possible. Keep identities and source videos disjoint across splits to prevent leakage.

## Train

```powershell
python train.py --data-dir data --epochs 15 --batch-size 16 --output-dir outputs
```

The best validation-AUROC checkpoint is saved to `outputs/best.pt`.

## Predict

```powershell
python predict.py --checkpoint outputs/best.pt --image path\to\face.jpg
```

## Predict a video

The video checker samples evenly spaced frames, scores each one with the image detector, and averages their deepfake probabilities.

```powershell
python predict.py --checkpoint outputs/best.pt --video path\to\video.mp4 --frames 24
```

`--frames` controls the maximum number of frames analyzed. This is frame-based video detection; it does not yet model motion consistency between frames.

This is a research baseline, not a forensic guarantee. Validate on held-out manipulation methods and report calibration, AUROC, precision, recall, F1-score, and operating-point error rates before claiming a production accuracy level.
