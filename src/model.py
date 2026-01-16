"""
LaborView AI - Multi-Head Model Architecture
Fine-tuned MedGemma encoder with task-specific heads for intrapartum ultrasound
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LaborViewOutput:
    """Model output container"""
    plane_logits: torch.Tensor          # (B, num_classes) - plane classification
    seg_masks: torch.Tensor             # (B, num_seg_classes, H, W) - segmentation
    labor_params: torch.Tensor          # (B, 2) - AoP and HSD predictions
    features: Optional[torch.Tensor] = None  # (B, D) - for visualization/analysis


class FeatureProjector(nn.Module):
    """Projects encoder features to shared representation space"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class ClassificationHead(nn.Module):
    """Standard plane classification head"""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    """Labor parameters regression head (AoP, HSD)"""

    def __init__(self, input_dim: int, num_outputs: int = 2, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SegmentationDecoder(nn.Module):
    """
    Lightweight segmentation decoder using FPN-style upsampling.
    Designed to be edge-friendly while maintaining accuracy.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        decoder_channels: list = [256, 128, 64],
        input_resolution: int = 24,  # Typical ViT patch grid size for 384 input
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_classes = num_classes

        # Initial projection from transformer features
        self.input_proj = nn.Conv2d(input_dim, decoder_channels[0], 1)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]

        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                )
            )
            in_ch = out_ch

        # Final upsampling to full resolution and classification
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) transformer features or (B, D, H, W) conv features
            target_size: (H, W) target output size
        """
        B = x.shape[0]

        # Handle transformer output (B, N, D) -> (B, D, H, W)
        if x.dim() == 3:
            # Remove CLS token if present
            if x.shape[1] == self.input_resolution ** 2 + 1:
                x = x[:, 1:, :]

            H = W = int(x.shape[1] ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        # Decode
        x = self.input_proj(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.final_up(x)
        x = self.classifier(x)

        # Resize to target if specified
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class UncertaintyWeighting(nn.Module):
    """
    Learned uncertainty weighting for multi-task learning.
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Log variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: Dict of task losses {"classification": ..., "segmentation": ..., "regression": ...}
        Returns:
            Weighted combined loss
        """
        task_names = ["classification", "segmentation", "regression"]
        total_loss = 0

        for i, name in enumerate(task_names):
            if name in losses:
                # precision = exp(-log_var) = 1/var
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * losses[name] + self.log_vars[i]

        return total_loss

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights for logging"""
        weights = torch.exp(-self.log_vars).detach().cpu().numpy()
        return {
            "classification": float(weights[0]),
            "segmentation": float(weights[1]),
            "regression": float(weights[2]),
        }


class LaborViewModel(nn.Module):
    """
    Multi-head model for intrapartum ultrasound analysis.

    Architecture:
    - Encoder: MedGemma SigLIP (or MobileViT for edge)
    - Shared projection layer
    - Task heads: Classification, Segmentation, Regression
    """

    def __init__(
        self,
        encoder_name: str = "medgemma",
        encoder_pretrained: str = "google/medgemma-4b-it",
        encoder_hidden_dim: int = 1152,
        projection_dim: int = 512,
        num_plane_classes: int = 6,
        num_seg_classes: int = 3,
        num_labor_params: int = 2,
        seg_decoder_channels: list = [256, 128, 64],
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder

        # Load encoder based on type
        self.encoder = self._load_encoder(encoder_name, encoder_pretrained)
        self.encoder_hidden_dim = encoder_hidden_dim

        # Shared feature projection
        self.projector = FeatureProjector(encoder_hidden_dim, projection_dim, dropout)

        # Task-specific heads
        self.classification_head = ClassificationHead(projection_dim, num_plane_classes, dropout)
        self.regression_head = RegressionHead(projection_dim, num_labor_params, dropout)
        self.segmentation_decoder = SegmentationDecoder(
            encoder_hidden_dim,  # Seg decoder uses raw encoder features
            num_seg_classes,
            seg_decoder_channels,
        )

        # Uncertainty weighting for multi-task learning
        self.uncertainty_weighting = UncertaintyWeighting(num_tasks=3)

        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()

    def _load_encoder(self, encoder_name: str, pretrained: str) -> nn.Module:
        """Load the vision encoder"""

        if encoder_name == "medgemma":
            return self._load_medgemma_encoder(pretrained)
        elif encoder_name == "siglip":
            return self._load_siglip_encoder(pretrained)
        elif encoder_name == "mobilevit":
            return self._load_mobilevit_encoder(pretrained)
        elif encoder_name == "efficientnet":
            return self._load_efficientnet_encoder(pretrained)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def _load_medgemma_encoder(self, pretrained: str) -> nn.Module:
        """Extract vision encoder from MedGemma"""
        from transformers import AutoModel

        print(f"Loading MedGemma vision encoder from {pretrained}...")

        # Load full model
        model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Extract vision tower (SigLIP encoder)
        if hasattr(model, 'vision_tower'):
            vision_encoder = model.vision_tower
        elif hasattr(model, 'vision_model'):
            vision_encoder = model.vision_model
        else:
            raise ValueError("Could not find vision encoder in MedGemma model")

        # Clean up the rest
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return vision_encoder

    def _load_siglip_encoder(self, pretrained: str) -> nn.Module:
        """Load standalone SigLIP encoder"""
        from transformers import SiglipVisionModel

        print(f"Loading SigLIP encoder from {pretrained}...")
        return SiglipVisionModel.from_pretrained(pretrained)

    def _load_mobilevit_encoder(self, pretrained: str) -> nn.Module:
        """Load MobileViT for edge deployment"""
        from transformers import MobileViTModel

        print(f"Loading MobileViT encoder from {pretrained}...")
        return MobileViTModel.from_pretrained(pretrained)

    def _load_efficientnet_encoder(self, pretrained: str) -> nn.Module:
        """Load EfficientNet for edge deployment"""
        import timm

        print(f"Loading EfficientNet encoder...")
        # Remove classifier head
        return timm.create_model(pretrained, pretrained=True, num_classes=0)

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        print("Encoder unfrozen")

    def _encode(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from encoder.
        Returns:
            pooled: (B, D) pooled features for classification/regression
            sequence: (B, N, D) sequence features for segmentation
        """
        if self.encoder_name in ["medgemma", "siglip"]:
            outputs = self.encoder(pixel_values)
            sequence = outputs.last_hidden_state  # (B, N, D)
            pooled = sequence.mean(dim=1)  # Global average pooling

        elif self.encoder_name == "mobilevit":
            outputs = self.encoder(pixel_values)
            sequence = outputs.last_hidden_state  # (B, D, H, W)
            pooled = outputs.pooler_output  # (B, D)
            # Reshape for segmentation decoder
            B, D, H, W = sequence.shape
            sequence = sequence.flatten(2).transpose(1, 2)  # (B, H*W, D)

        elif self.encoder_name == "efficientnet":
            sequence = self.encoder.forward_features(pixel_values)  # (B, D, H, W)
            pooled = sequence.mean(dim=[2, 3])  # Global average pooling
            B, D, H, W = sequence.shape
            sequence = sequence.flatten(2).transpose(1, 2)  # (B, H*W, D)

        return pooled, sequence

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False,
    ) -> LaborViewOutput:
        """
        Forward pass through multi-head model.

        Args:
            pixel_values: (B, C, H, W) input images
            return_features: Whether to return intermediate features

        Returns:
            LaborViewOutput with predictions from all heads
        """
        # Encode
        pooled_features, sequence_features = self._encode(pixel_values)

        # Project pooled features for classification and regression
        projected = self.projector(pooled_features)

        # Task heads
        plane_logits = self.classification_head(projected)
        labor_params = self.regression_head(projected)
        seg_masks = self.segmentation_decoder(
            sequence_features,
            target_size=pixel_values.shape[-2:]
        )

        return LaborViewOutput(
            plane_logits=plane_logits,
            seg_masks=seg_masks,
            labor_params=labor_params,
            features=projected if return_features else None,
        )

    def compute_loss(
        self,
        outputs: LaborViewOutput,
        plane_labels: Optional[torch.Tensor] = None,
        seg_labels: Optional[torch.Tensor] = None,
        labor_labels: Optional[torch.Tensor] = None,
        use_uncertainty_weighting: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs
            plane_labels: (B,) classification labels
            seg_labels: (B, H, W) segmentation labels
            labor_labels: (B, 2) regression labels [AoP, HSD]
            use_uncertainty_weighting: Use learned loss weights

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses for logging
        """
        losses = {}

        # Classification loss
        if plane_labels is not None:
            losses["classification"] = F.cross_entropy(outputs.plane_logits, plane_labels)

        # Segmentation loss (Dice + BCE)
        if seg_labels is not None:
            seg_probs = F.softmax(outputs.seg_masks, dim=1)

            # Dice loss
            dice_loss = self._dice_loss(seg_probs, seg_labels)

            # Cross entropy loss
            ce_loss = F.cross_entropy(outputs.seg_masks, seg_labels)

            losses["segmentation"] = dice_loss + ce_loss

        # Regression loss (Smooth L1)
        if labor_labels is not None:
            losses["regression"] = F.smooth_l1_loss(outputs.labor_params, labor_labels)

        # Combine losses
        if use_uncertainty_weighting and len(losses) > 1:
            total_loss = self.uncertainty_weighting(losses)
        else:
            total_loss = sum(losses.values())

        return total_loss, losses

    def _dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute Dice loss for segmentation"""
        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Compute Dice per class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)

        # Average over classes and batch
        return 1 - dice.mean()


def create_model(config) -> LaborViewModel:
    """Factory function to create model from config"""
    return LaborViewModel(
        encoder_name=config.model.encoder_name,
        encoder_pretrained=config.model.encoder_pretrained,
        encoder_hidden_dim=config.model.encoder_hidden_dim,
        projection_dim=config.model.projection_dim,
        num_plane_classes=config.model.num_plane_classes,
        num_seg_classes=config.model.num_seg_classes,
        num_labor_params=config.model.num_labor_params,
        seg_decoder_channels=config.model.seg_decoder_channels,
        dropout=config.model.dropout,
        freeze_encoder=config.model.freeze_encoder,
    )


# For edge deployment - smaller model variant
class LaborViewModelEdge(LaborViewModel):
    """
    Edge-optimized variant of LaborViewModel.
    Uses MobileViT encoder and simplified heads.
    """

    def __init__(self, **kwargs):
        # Override defaults for edge
        kwargs.setdefault("encoder_name", "mobilevit")
        kwargs.setdefault("encoder_pretrained", "apple/mobilevit-small")
        kwargs.setdefault("encoder_hidden_dim", 640)  # MobileViT-S hidden dim
        kwargs.setdefault("projection_dim", 256)
        kwargs.setdefault("seg_decoder_channels", [128, 64, 32])
        kwargs.setdefault("freeze_encoder", False)

        super().__init__(**kwargs)

    def export_onnx(self, path: str, input_size: Tuple[int, int] = (256, 256)):
        """Export model to ONNX format for edge deployment"""
        self.eval()

        dummy_input = torch.randn(1, 3, *input_size)

        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=["image"],
            output_names=["plane_logits", "seg_masks", "labor_params"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "plane_logits": {0: "batch_size"},
                "seg_masks": {0: "batch_size"},
                "labor_params": {0: "batch_size"},
            },
            opset_version=14,
        )
        print(f"Exported ONNX model to {path}")
