"""
LaborView AI - MedSigLIP Training Script
Fine-tune MedSigLIP vision encoder for ultrasound segmentation
Self-contained script for HuggingFace Jobs
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.50.0",
#   "accelerate>=0.27.0",
#   "albumentations>=1.3.0",
#   "pillow>=10.0.0",
#   "numpy>=1.24.0",
#   "tqdm>=4.65.0",
#   "huggingface_hub>=0.20.0",
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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBU_AVAILABLE = True
except ImportError:
    ALBU_AVAILABLE = False
    import torchvision.transforms as T


@dataclass
class Config:
    # Data
    data_url: str = "https://zenodo.org/records/17655183/files/DatasetV3.zip?download=1"
    data_dir: Path = Path("./data")
    image_size: int = 448  # MedSigLIP native resolution

    # Model - MedSigLIP (HAI-DEF model for competition)
    encoder_name: str = "medsiglip"
    encoder_pretrained: str = "google/medsiglip-448"
    encoder_hidden_dim: int = 1152  # SigLIP-SO400M hidden dim
    projection_dim: int = 256

    # Task heads
    num_plane_classes: int = 2
    num_seg_classes: int = 3  # background, symphysis, head

    # Training
    batch_size: int = 4  # Reduced for memory when encoder unfrozen
    num_epochs: int = 30
    learning_rate: float = 5e-5  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    gradient_accumulation: int = 8  # Increased to maintain effective batch size
    freeze_encoder_epochs: int = 3  # Freeze encoder initially
    use_gradient_checkpointing: bool = True  # Save memory

    # Output
    output_dir: Path = Path("./outputs")
    hub_model_id: str = "samwell/laborview-medsiglip"
    push_to_hub: bool = True
    seed: int = 42


class UltrasoundDataset(Dataset):
    """Dataset for ultrasound segmentation"""

    def __init__(self, data_dir: Path, split: str = "train", image_size: int = 448, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.samples = self._find_samples()
        print(f"Found {len(self.samples)} samples for {split}")
        self.transform = self._get_transform(augment and split == "train")

    def _find_samples(self) -> List[Dict]:
        samples = []
        seg_dir = self.data_dir / self.split / "seg"

        if not seg_dir.exists():
            print(f"Warning: {seg_dir} not found")
            return samples

        for video_dir in seg_dir.iterdir():
            if not video_dir.is_dir():
                continue

            # Check for images and masks
            image_dir = video_dir / "image"
            mask_dir = video_dir / "mask"

            if mask_dir.exists():
                for mask_path in mask_dir.glob("*.png"):
                    # Try to find corresponding image
                    image_path = None
                    if image_dir.exists():
                        potential_image = image_dir / mask_path.name
                        if potential_image.exists():
                            image_path = str(potential_image)

                    samples.append({
                        "mask_path": str(mask_path),
                        "image_path": image_path,
                        "video_id": video_dir.name,
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
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # MedSigLIP normalization
                    ToTensorV2()
                ])
            else:
                return A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2()
                ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load mask
        mask = Image.open(sample["mask_path"]).convert("L")
        mask = np.array(mask)

        # Load or create image from mask
        if sample["image_path"] and os.path.exists(sample["image_path"]):
            image = Image.open(sample["image_path"]).convert("RGB")
            image = np.array(image)
        else:
            # Use mask as grayscale image
            image = np.stack([mask, mask, mask], axis=-1)

        # Convert mask to class labels (0=background, 1=symphysis, 2=head)
        # Assuming mask has different intensity values for different structures
        mask_classes = np.zeros_like(mask, dtype=np.int64)
        mask_classes[mask > 0] = 1  # Any non-zero is foreground
        mask_classes[mask > 127] = 2  # Higher intensity is second class

        if ALBU_AVAILABLE:
            transformed = self.transform(image=image, mask=mask_classes)
            image, mask = transformed["image"], transformed["mask"]
        else:
            image = self.transform(Image.fromarray(image))
            mask = torch.from_numpy(
                np.array(Image.fromarray(mask_classes.astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.NEAREST
                ))
            ).long()

        return {
            "pixel_values": image,
            "seg_labels": mask,
            "plane_labels": torch.tensor(1, dtype=torch.long)  # Standard plane
        }


class SegmentationDecoder(nn.Module):
    """Decoder for upsampling vision features to segmentation mask"""

    def __init__(self, input_dim: int, num_classes: int, decoder_channels=[512, 256, 128, 64]):
        super().__init__()

        self.input_proj = nn.Conv2d(input_dim, decoder_channels[0], 1)

        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ))
            in_ch = out_ch

        # Final upsampling to full resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, target_size=None):
        B = x.shape[0]

        # Handle different input shapes
        if x.dim() == 3:
            # [B, num_patches, hidden_dim] -> [B, hidden_dim, H, W]
            num_patches = x.shape[1]
            H = W = int(num_patches ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.input_proj(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.final_up(x)
        x = self.classifier(x)

        if target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class LaborViewMedSigLIP(nn.Module):
    """LaborView model with MedSigLIP vision encoder"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Load MedSigLIP
        print(f"Loading MedSigLIP from {config.encoder_pretrained}...")
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            config.encoder_pretrained,
            trust_remote_code=True
        )

        # Get vision model from SigLIP
        if hasattr(self.encoder, 'vision_model'):
            self.vision_encoder = self.encoder.vision_model
        else:
            self.vision_encoder = self.encoder

        # Get hidden dimension from config
        if hasattr(self.vision_encoder.config, 'hidden_size'):
            hidden_dim = self.vision_encoder.config.hidden_size
        else:
            hidden_dim = config.encoder_hidden_dim

        print(f"Vision encoder hidden dim: {hidden_dim}")

        # Projector for classification
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )

        # Classification head
        self.cls_head = nn.Linear(config.projection_dim, config.num_plane_classes)

        # Segmentation decoder
        self.seg_decoder = SegmentationDecoder(hidden_dim, config.num_seg_classes)

    def forward(self, pixel_values):
        # Get vision features
        if hasattr(self, 'vision_encoder'):
            outputs = self.vision_encoder(pixel_values)
        else:
            outputs = self.encoder.get_image_features(pixel_values, return_dict=True)

        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            hidden = outputs.pooler_output
        else:
            hidden = outputs

        # Handle different output formats
        if hidden.dim() == 2:
            # [B, hidden_dim] - pooled output
            pooled = hidden
            # Create spatial features for segmentation
            B, D = hidden.shape
            seq = hidden.unsqueeze(1).expand(B, 32*32, D)
        elif hidden.dim() == 3:
            # [B, num_patches, hidden_dim]
            pooled = hidden.mean(dim=1)
            seq = hidden
        else:
            # [B, D, H, W]
            B, D, H, W = hidden.shape
            pooled = hidden.mean(dim=[2, 3])
            seq = hidden.flatten(2).transpose(1, 2)

        # Classification
        projected = self.projector(pooled)
        plane_logits = self.cls_head(projected)

        # Segmentation
        seg_masks = self.seg_decoder(seq, target_size=pixel_values.shape[-2:])

        return plane_logits, seg_masks

    def compute_loss(self, plane_logits, seg_masks, plane_labels, seg_labels):
        losses = {}

        # Classification loss
        if plane_labels is not None:
            losses["cls"] = F.cross_entropy(plane_logits, plane_labels)

        # Segmentation loss (Dice + CE)
        if seg_labels is not None:
            # Cross entropy
            ce_loss = F.cross_entropy(seg_masks, seg_labels.long())

            # Dice loss
            seg_probs = F.softmax(seg_masks, dim=1)
            target_oh = F.one_hot(seg_labels.long(), self.config.num_seg_classes).permute(0, 3, 1, 2).float()

            intersection = (seg_probs * target_oh).sum(dim=(2, 3))
            union = seg_probs.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
            dice_loss = 1 - ((2 * intersection + 1e-6) / (union + 1e-6)).mean()

            losses["seg"] = dice_loss + ce_loss

        return sum(losses.values()), losses

    def freeze_encoder(self):
        """Freeze the vision encoder"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")

    def unfreeze_encoder(self, use_gradient_checkpointing=True):
        """Unfreeze the vision encoder with optional gradient checkpointing"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
                self.vision_encoder.gradient_checkpointing_enable()
                print("Encoder unfrozen with gradient checkpointing")
            else:
                print("Encoder unfrozen (gradient checkpointing not available)")
        else:
            print("Encoder unfrozen")


def train_epoch(model, loader, optimizer, scheduler, scaler, device, config, epoch):
    model.train()
    total_loss, num_batches = 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training")
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
    total_loss, total_iou, num_batches = 0, 0, 0

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

        total_loss += loss.item()
        total_iou += intersection / (union + 1e-6)
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches


def download_with_retry(url, dest_path, max_retries=3):
    """Download file with retry logic using subprocess for robustness"""
    import subprocess
    import shutil

    # Try wget first (more robust for large files)
    if shutil.which("wget"):
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries} with wget...")
                result = subprocess.run(
                    ["wget", "-c", "-O", str(dest_path), url],
                    check=True, capture_output=True, text=True
                )
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    return True
            except subprocess.CalledProcessError as e:
                print(f"wget failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)

    # Fallback to curl
    if shutil.which("curl"):
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries} with curl...")
                result = subprocess.run(
                    ["curl", "-L", "-C", "-", "-o", str(dest_path), url],
                    check=True, capture_output=True, text=True
                )
                if dest_path.exists() and dest_path.stat().st_size > 0:
                    return True
            except subprocess.CalledProcessError as e:
                print(f"curl failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(5)

    # Last resort: urllib with chunked download
    print("Falling back to urllib chunked download...")
    import urllib.request
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192 * 16  # 128KB chunks
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\rDownloaded {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
                print()
                return True
        except Exception as e:
            print(f"urllib attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5)

    raise Exception(f"Failed to download {url} after {max_retries} attempts")


def download_dataset(config):
    """Download and extract dataset"""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = config.data_dir / "dataset.zip"

    if not (config.data_dir / "train").exists():
        print(f"Downloading dataset from {config.data_url}...")
        download_with_retry(config.data_url, zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(config.data_dir)

        # Handle nested zips
        inner_zip = config.data_dir / "DatasetV3.zip"
        if inner_zip.exists():
            with zipfile.ZipFile(inner_zip, 'r') as z:
                z.extractall(config.data_dir)

        # Extract split zips
        dataset_dir = config.data_dir / "DatasetV3"
        if dataset_dir.exists():
            for split in ["train", "val", "test"]:
                for sz in dataset_dir.glob(f"{split}*.zip"):
                    print(f"Extracting {sz.name}...")
                    with zipfile.ZipFile(sz, 'r') as z:
                        z.extractall(dataset_dir)

        # Cleanup
        zip_path.unlink(missing_ok=True)
        inner_zip.unlink(missing_ok=True)

    # Return the correct data directory
    dataset_v3 = config.data_dir / "DatasetV3"
    if dataset_v3.exists():
        return dataset_v3
    return config.data_dir


def main():
    config = Config()

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Download dataset
    data_dir = download_dataset(config)
    print(f"Data directory: {data_dir}")

    # Create datasets
    train_dataset = UltrasoundDataset(data_dir, "train", config.image_size, augment=True)
    val_dataset = UltrasoundDataset(data_dir, "val", config.image_size, augment=False)

    if len(val_dataset) == 0:
        print("No validation data, using 10% of train")
        train_size = int(0.9 * len(train_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, len(train_dataset) - train_size]
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print(f"Creating model with {config.encoder_name} encoder...")
    model = LaborViewMedSigLIP(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Freeze encoder initially for stable training
    model.freeze_encoder()

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_epochs / config.num_epochs
    )

    # Scaler for mixed precision
    scaler = GradScaler("cuda")

    # Output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    best_val_loss = float("inf")
    print("Starting training")

    for epoch in range(config.num_epochs):
        # Unfreeze encoder after initial epochs
        if epoch == config.freeze_encoder_epochs:
            # Clear memory before unfreezing
            torch.cuda.empty_cache()
            model.unfreeze_encoder(use_gradient_checkpointing=config.use_gradient_checkpointing)
            # Recreate optimizer with all parameters
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate * 0.1,  # Lower LR for encoder
                weight_decay=config.weight_decay
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config.learning_rate * 0.1,
                total_steps=len(train_loader) * (config.num_epochs - epoch),
                pct_start=0.1
            )

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, epoch)
        val_loss, val_iou = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train: {train_loss:.4f}, Val: {val_loss:.4f}, IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "config": vars(config)
            }, config.output_dir / "best.pt")
            print("  >>> New best!")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config)
    }, config.output_dir / "final.pt")

    # Push to Hub
    if config.push_to_hub:
        try:
            from huggingface_hub import HfApi, create_repo
            print(f"Pushing to Hub: {config.hub_model_id}")
            create_repo(config.hub_model_id, exist_ok=True)
            HfApi().upload_folder(
                folder_path=str(config.output_dir),
                repo_id=config.hub_model_id,
                commit_message=f"LaborView MedSigLIP v1 - IoU: {val_iou:.4f}"
            )
            print(f"Uploaded to https://huggingface.co/{config.hub_model_id}")
        except Exception as e:
            print(f"Hub upload failed: {e}")

    print(f"Training complete! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
