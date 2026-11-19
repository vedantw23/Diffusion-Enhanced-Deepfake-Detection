from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from data import build_loaders
from model import DiffusionEnhancedDetector


def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(targets.tolist())
    predictions = [value >= 0.5 for value in probabilities]
    accuracy = sum(prediction == bool(label) for prediction, label in zip(predictions, labels)) / len(labels)
    return accuracy, roc_auc_score(labels, probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = build_loaders(args.data_dir, args.batch_size, args.workers)
    model = DiffusionEnhancedDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_function = nn.BCEWithLogitsLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_function(logits, targets.float().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        scheduler.step()
        accuracy, auc = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={running_loss / len(train_loader.dataset):.4f} val_acc={accuracy:.4f} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx, "val_auc": auc}, output_dir / "best.pt")


if __name__ == "__main__":
    main()
