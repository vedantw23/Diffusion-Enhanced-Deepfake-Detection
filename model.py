"""Diffusion-enhanced local-global deepfake detector."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class DiffusionResidualPyramid(nn.Module):
    """Build deterministic multi-scale residual views for artifact analysis."""

    def __init__(self, blur_sizes: tuple[int, ...] = (3, 5, 9)) -> None:
        super().__init__()
        self.blur_sizes = blur_sizes

    def forward(self, images: Tensor) -> Tensor:
        residuals = []
        for size in self.blur_sizes:
            smoothed = F.avg_pool2d(images, kernel_size=size, stride=1, padding=size // 2, count_include_pad=False)
            residuals.append((images - smoothed).abs())
        return torch.cat(residuals, dim=1)


class LocalPatchEncoder(nn.Module):
    """Extract artifact evidence from a grid of local image patches."""

    def __init__(self, in_channels: int = 9, feature_dim: int = 256, grid_size: int = 4) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, residuals: Tensor) -> Tensor:
        batch_size, channels, height, width = residuals.shape
        patch_height, patch_width = height // self.grid_size, width // self.grid_size
        cropped = residuals[:, :, : patch_height * self.grid_size, : patch_width * self.grid_size]
        patches = cropped.reshape(batch_size, channels, self.grid_size, patch_height, self.grid_size, patch_width)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, patch_height, patch_width)
        features = self.projection(self.encoder(patches).flatten(1))
        return features.reshape(batch_size, self.grid_size * self.grid_size, -1)


class LocalGlobalFusion(nn.Module):
    def __init__(self, feature_dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())

    def forward(self, global_feature: Tensor, local_features: Tensor) -> Tensor:
        query = global_feature.unsqueeze(1)
        attended_local, _ = self.attention(query, local_features, local_features)
        attended_local = attended_local.squeeze(1)
        gate = self.gate(torch.cat((global_feature, attended_local), dim=1))
        return gate * global_feature + (1.0 - gate) * attended_local


class DiffusionEnhancedDetector(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, feature_dim: int = 256) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        backbone = efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()
        self.global_backbone = backbone
        self.global_projection = nn.Linear(1280, feature_dim)
        self.diffusion = DiffusionResidualPyramid()
        self.local_encoder = LocalPatchEncoder(feature_dim=feature_dim)
        self.fusion = LocalGlobalFusion(feature_dim=feature_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(0.25), nn.Linear(feature_dim, 1))

    def forward(self, images: Tensor) -> Tensor:
        global_feature = self.global_projection(self.global_backbone(images))
        local_features = self.local_encoder(self.diffusion(images))
        fused_feature = self.fusion(global_feature, local_features)
        return self.classifier(fused_feature).squeeze(1)
