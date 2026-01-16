"""
LaborView AI - HuggingFace Jobs Training Script
Self-contained script for cloud GPU training

Run with:
    hf_jobs uv --script train_hf_job.py --hardware gpu.a10g.small
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.40.0",
#   "accelerate>=0.27.0",
#   "albumentations>=1.3.0",
#   "pillow>=10.0.0",
#   "numpy>=1.24.0",
#   "tqdm>=4.65.0",
#   "huggingface_hub>=0.20.0",
#   "timm>=0.9.0",
#   "pandas>=2.0.0",
#   "opencv-python-headless>=4.8.0",
# ]
# ///

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBU_AVAILABLE = True
except ImportError:
    ALBU_AVAILABLE = False
    import torchvision.transforms as T


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Data
    data_url: str = "https://zenodo.org/records/17655183/files/DatasetV3.zip?download=1"
    data_dir: Path = Path("./data")
    image_size: int = 384

    # Model
    encoder_name: str = "mobilevit"  # Use smaller model for faster training
    encoder_pretrained: str = "apple/mobilevit-small"
    encoder_hidden_dim: int = 640
    projection_dim: int = 256
    num_plane_classes: int = 2  # pos/neg for standard plane
    num_seg_classes: int = 3   # bg, pubic symphysis, fetal head
    num_labor_params: int = 2  # AoP, HSD

    # Training
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    gradient_accumulation: int = 2

    # Output
    output_dir: Path = Path("./outputs")
    hub_model_id: str = "samwell/laborview-ultrasound"
    push_to_hub: bool = True

    seed: int = 42


# ============================================================================
# Dataset
# ============================================================================

class UltrasoundDataset(Dataset):
    """Dataset for maternal-fetal ultrasound segmentation"""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        image_size: int = 384,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # Find all segmentation samples
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples for {split}")

        # Setup transforms
        self.transform = self._get_transform(augment and split == "train")

    def _find_samples(self) -> List[Dict]:
        """Find all image-mask pairs"""
        samples = []
        seg_dir = self.data_dir / self.split / "seg"

        if not seg_dir.exists():
            print(f"Warning: {seg_dir} not found")
            return samples

        for video_dir in seg_dir.iterdir():
            if not video_dir.is_dir():
                continue

            mask_dir = video_dir / "mask"
            if not mask_dir.exists():
                continue

            # Find original images (if available) or use masks
            for mask_path in mask_dir.glob("*.png"):
                # Parse filename to get frame info
                # Format: VIDEOID_FRAME_6.png
                parts = mask_path.stem.rsplit("_", 2)
                if len(parts) >= 2:
                    video_id = parts[0]
                    frame_idx = parts[1]

                samples.append({
                    "mask_path": str(mask_path),
                    "video_id": video_dir.name,
                    "frame_idx": frame_idx if len(parts) >= 2 else "0",
                })

        return samples

    def _get_transform(self, augment: bool):
        if ALBU_AVAILABLE:
            if augment:
                return A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(var_limit=(10, 50), p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                return A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            transforms = [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            return T.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load mask (grayscale)
        mask = Image.open(sample["mask_path"]).convert("L")
        mask = np.array(mask)

        # For now, use mask as pseudo-image (grayscale -> RGB)
        # In production, you'd load the actual ultrasound frame
        image = np.stack([mask, mask, mask], axis=-1)

        # Normalize mask to class indices (0, 1, 2)
        mask = (mask > 0).astype(np.int64)  # Binary for now

        if ALBU_AVAILABLE:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = self.transform(Image.fromarray(image))
            mask = torch.from_numpy(
                np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                ))
            ).long()

        return {
            "pixel_values": image,
            "seg_labels": mask,
            "plane_labels": torch.tensor(1, dtype=torch.long),  # Assume positive samples
        }


# ============================================================================
# Model
# ============================================================================

class SegmentationDecoder(nn.Module):
    """Lightweight segmentation decoder"""

    def __init__(self, input_dim: int, num_classes: int, decoder_channels=[128, 64, 32]):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, decoder_channels[0], 1)

        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x, target_size=None):
        B = x.shape[0]
        if x.dim() == 3:
            H = W = int(x.shape[1] ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.input_proj(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.final_up(x)
        x = self.classifier(x)

        if target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class LaborViewModel(nn.Module):
    """Multi-head model for ultrasound analysis"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load encoder
        self.encoder = self._load_encoder(config)

        # Projection
        self.projector = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )

        # Heads
        self.cls_head = nn.Linear(config.projection_dim, config.num_plane_classes)
        self.seg_decoder = SegmentationDecoder(
            config.encoder_hidden_dim,
            config.num_seg_classes,
        )

    def _load_encoder(self, config):
        if config.encoder_name == "mobilevit":
            from transformers import MobileViTModel
            return MobileViTModel.from_pretrained(config.encoder_pretrained)
        else:
            raise ValueError(f"Unknown encoder: {config.encoder_name}")

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        hidden = outputs.last_hidden_state  # (B, H*W, D) or (B, D, H, W)

        if hidden.dim() == 4:
            B, D, H, W = hidden.shape
            pooled = hidden.mean(dim=[2, 3])
            seq = hidden.flatten(2).transpose(1, 2)
        else:
            pooled = hidden.mean(dim=1)
            seq = hidden

        projected = self.projector(pooled)
        plane_logits = self.cls_head(projected)
        seg_masks = self.seg_decoder(seq, target_size=pixel_values.shape[-2:])

        return plane_logits, seg_masks

    def compute_loss(self, plane_logits, seg_masks, plane_labels, seg_labels):
        losses = {}

        if plane_labels is not None:
            losses["cls"] = F.cross_entropy(plane_logits, plane_labels)

        if seg_labels is not None:
            # Dice + CE loss
            seg_probs = F.softmax(seg_masks, dim=1)
            target_oh = F.one_hot(seg_labels.long(), self.config.num_seg_classes)
            target_oh = target_oh.permute(0, 3, 1, 2).float()

            intersection = (seg_probs * target_oh).sum(dim=(2, 3))
            union = seg_probs.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1 - dice.mean()

            ce_loss = F.cross_entropy(seg_masks, seg_labels.long())
            losses["seg"] = dice_loss + ce_loss

        total = sum(losses.values())
        return total, losses


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, device, config):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        seg_labels = batch["seg_labels"].to(device)
        plane_labels = batch["plane_labels"].to(device)

        with autocast("cuda", enabled=True):
            plane_logits, seg_masks = model(pixel_values)
            loss, _ = model.compute_loss(plane_logits, seg_masks, plane_labels, seg_labels)

        loss = loss / config.gradient_accumulation
        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * config.gradient_accumulation
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item() * config.gradient_accumulation:.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Validating"):
        pixel_values = batch["pixel_values"].to(device)
        seg_labels = batch["seg_labels"].to(device)
        plane_labels = batch["plane_labels"].to(device)

        plane_logits, seg_masks = model(pixel_values)
        loss, _ = model.compute_loss(plane_logits, seg_masks, plane_labels, seg_labels)

        # Compute IoU
        seg_preds = seg_masks.argmax(dim=1)
        intersection = ((seg_preds == 1) & (seg_labels == 1)).sum().item()
        union = ((seg_preds == 1) | (seg_labels == 1)).sum().item()
        iou = intersection / (union + 1e-6)

        total_loss += loss.item()
        total_iou += iou
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches


def download_dataset(config: Config):
    """Download and extract dataset"""
    config.data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = config.data_dir / "dataset.zip"

    if not (config.data_dir / "train").exists():
        print(f"Downloading dataset from {config.data_url}...")
        urllib.request.urlretrieve(config.data_url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(config.data_dir)

        # Handle nested zip
        inner_zip = config.data_dir / "DatasetV3.zip"
        if inner_zip.exists():
            with zipfile.ZipFile(inner_zip, 'r') as z:
                z.extractall(config.data_dir)

        # Extract train/val/test
        for split in ["train", "val", "test"]:
            split_zips = list((config.data_dir / "DatasetV3").glob(f"{split}*.zip"))
            for sz in split_zips:
                print(f"Extracting {sz.name}...")
                with zipfile.ZipFile(sz, 'r') as z:
                    z.extractall(config.data_dir / "DatasetV3")

        # Cleanup
        zip_path.unlink(missing_ok=True)
        inner_zip.unlink(missing_ok=True)

    return config.data_dir / "DatasetV3"


def main():
    config = Config()

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download data
    data_dir = download_dataset(config)
    print(f"Data directory: {data_dir}")

    # Create datasets
    train_dataset = UltrasoundDataset(data_dir, "train", config.image_size, augment=True)
    val_dataset = UltrasoundDataset(data_dir, "val", config.image_size, augment=False)

    # If val is empty, use train subset
    if len(val_dataset) == 0:
        print("No validation data found, using 10% of train")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print(f"\nCreating model with {config.encoder_name} encoder...")
    model = LaborViewModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps,
        pct_start=config.warmup_epochs / config.num_epochs
    )
    scaler = GradScaler("cuda")

    # Output dir
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training")
    print(f"{'='*60}")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config)
        val_loss, val_iou = validate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
            }, config.output_dir / "best.pt")
            print("  >>> New best model!")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config),
    }, config.output_dir / "final.pt")

    # Push to Hub
    if config.push_to_hub:
        try:
            from huggingface_hub import HfApi, create_repo

            print(f"\nPushing to Hub: {config.hub_model_id}")
            create_repo(config.hub_model_id, exist_ok=True)

            api = HfApi()
            api.upload_folder(
                folder_path=str(config.output_dir),
                repo_id=config.hub_model_id,
                commit_message=f"LaborView v1 - Val IoU: {val_iou:.4f}",
            )
            print(f"Uploaded to https://huggingface.co/{config.hub_model_id}")
        except Exception as e:
            print(f"Hub upload failed: {e}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
