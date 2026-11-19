# Diffusion-Enhanced Deepfake Detection

PyTorch starter project for binary real-vs-deepfake image classification. The model integrates:

- **Global features:** EfficientNet-B0 learns face-wide semantic and texture evidence.
- **Local features:** a patch encoder scores localized artifacts.
- **Diffusion enhancement:** a frozen, multi-level Gaussian-noise residual pyramid highlights generation artifacts at multiple noise levels.
- **Fusion:** gated local-global attention combines all three sources before classification.

## Web dashboard

Open `index.html` in a browser for a responsive interactive detector dashboard. It is a presentation prototype: its in-browser results are simulated from the selected file and do not run the PyTorch checkpoint. Connect `predict.py` to an API before using it for real detections.

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

This is a research baseline, not a forensic guarantee. Validate on held-out manipulation methods and report calibration, AUROC, and operating-point error rates before deployment.
