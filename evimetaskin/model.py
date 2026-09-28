"""Core EviMetaSkin model used in the paper experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EvidentialHead(nn.Module):
    """Auxiliary CE logits and non-negative evidence from a shared representation."""

    def __init__(self, in_dim: int, num_classes: int = 2, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.logit_head = nn.Linear(hidden_dim // 2, num_classes)
        self.evidence_head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.logit_head(features)
        evidence = F.softplus(self.evidence_head(features))
        alpha = evidence + 1.0
        probabilities = alpha / alpha.sum(dim=1, keepdim=True)
        uncertainty = alpha.size(1) / alpha.sum(dim=1)
        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "evidential_probs": probabilities,
            "uncertainty": uncertainty,
        }


class EviMetaSkin(nn.Module):
    """Image and structured-metadata classifier with gated fusion and EDL."""

    def __init__(
        self,
        visual_backbone: str = "efficientnet_b0",
        metadata_dim: int = 11,
        num_classes: int = 2,
        freeze_image_encoder: bool = False,
        pretrained_visual: bool = False,
        dropout: float = 0.2,
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"gated", "concat", "visual_only"}:
            raise ValueError("fusion_mode must be one of: gated, concat, visual_only")
        self.fusion_mode = fusion_mode

        visual_backbone = visual_backbone.lower()
        if visual_backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_visual else None
            self.visual_encoder = models.efficientnet_b0(weights=weights)
        elif visual_backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained_visual else None
            self.visual_encoder = models.efficientnet_b3(weights=weights)
        elif visual_backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained_visual else None
            self.visual_encoder = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported visual_backbone: {visual_backbone}")

        if hasattr(self.visual_encoder, "classifier"):
            visual_dim = self.visual_encoder.classifier[1].in_features
            self.visual_encoder.classifier = nn.Identity()
        else:
            visual_dim = self.visual_encoder.fc.in_features
            self.visual_encoder.fc = nn.Identity()

        if freeze_image_encoder:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad = False

        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid(),
        )

        if fusion_mode == "gated":
            classifier_dim = 512 + 128 + 512
        elif fusion_mode == "concat":
            classifier_dim = 512 + 128
        else:
            classifier_dim = 512
        self.classifier = EvidentialHead(classifier_dim, num_classes, hidden_dim=512, dropout=dropout)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_features = self.visual_encoder(images.float())
        visual_embedding = F.normalize(self.visual_projector(visual_features), dim=-1)
        metadata_embedding = self.metadata_encoder(metadata.float())
        gate = None

        if self.fusion_mode == "gated":
            gate = self.gate(torch.cat([visual_embedding, metadata_embedding], dim=1))
            gated_visual = gate * visual_embedding
            fused = torch.cat([visual_embedding, metadata_embedding, gated_visual], dim=1)
        elif self.fusion_mode == "concat":
            fused = torch.cat([visual_embedding, metadata_embedding], dim=1)
        else:
            fused = visual_embedding

        outputs = self.classifier(fused)
        outputs["features"] = fused
        if gate is not None:
            outputs["fusion_gate_mean"] = gate.mean(dim=1)
        return outputs


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    ln_b_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    ln_b_beta = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    digamma_diff = torch.digamma(alpha) - torch.digamma(sum_alpha)
    kl = (alpha - beta) * digamma_diff
    return (kl.sum(dim=1, keepdim=True) + ln_b_alpha + ln_b_beta).squeeze(1)


def evidential_classification_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int = 1,
    annealing_step: int = 10,
) -> torch.Tensor:
    """Dirichlet evidential loss with annealed KL regularization."""
    num_classes = alpha.size(1)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    strength = alpha.sum(dim=1, keepdim=True)
    nll = torch.sum(labels_one_hot * (torch.digamma(strength) - torch.digamma(alpha)), dim=1).mean()
    annealing = min(1.0, float(epoch) / float(max(1, annealing_step)))
    alpha_tilde = labels_one_hot + (1.0 - labels_one_hot) * alpha
    return nll + annealing * _dirichlet_kl_to_uniform(alpha_tilde).mean()


# Compatibility alias for checkpoints and scripts created during development.
StructuredMetadataFusionClassifier = EviMetaSkin
