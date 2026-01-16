"""
LaborView AI - Training Script
Multi-task training for intrapartum ultrasound analysis
Optimized for HuggingFace Jobs cloud training with Hub upload
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
#   "wandb>=0.16.0",
#   "huggingface_hub>=0.20.0",
#   "timm>=0.9.0",
# ]
# ///

import sys
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import json
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import LaborViewConfig, get_config
from model import LaborViewModel, create_model
from dataset import create_dataloaders


class Trainer:
    """
    Multi-task trainer for LaborView model.
    Supports mixed precision, gradient accumulation, and Hub upload.
    """

    def __init__(
        self,
        model: LaborViewModel,
        config: LaborViewConfig,
        train_loader,
        val_loader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Setup optimizer with different LR for encoder
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Mixed precision
        self.scaler = GradScaler("cuda") if config.training.use_amp else None

        # Output directory
        self.output_dir = Path(config.training.output_dir) / config.training.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.history = {"train": [], "val": []}

        # Try to setup wandb
        self.wandb_run = self._setup_wandb()

    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with different learning rates for encoder and heads"""
        cfg = self.config.training

        # Separate encoder and head parameters
        encoder_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": head_params, "lr": cfg.learning_rate},
            {"params": encoder_params, "lr": cfg.encoder_lr},
        ]

        return AdamW(param_groups, weight_decay=cfg.weight_decay)

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        cfg = self.config.training
        total_steps = len(self.train_loader) * cfg.num_epochs

        return OneCycleLR(
            self.optimizer,
            max_lr=[cfg.learning_rate, cfg.encoder_lr],
            total_steps=total_steps,
            pct_start=cfg.warmup_epochs / cfg.num_epochs,
            anneal_strategy="cos",
        )

    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            import wandb

            run = wandb.init(
                project="laborview-ai",
                name=self.config.training.experiment_name,
                config={
                    "model": vars(self.config.model),
                    "training": vars(self.config.training),
                    "data": {k: str(v) for k, v in vars(self.config.data).items()},
                },
            )
            print(f"W&B run: {run.url}")
            return run
        except Exception as e:
            print(f"W&B not available: {e}")
            return None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        cfg = self.config.training

        total_loss = 0
        task_losses = {"classification": 0, "segmentation": 0, "regression": 0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            plane_labels = batch.get("plane_labels")
            seg_labels = batch.get("seg_labels")
            labor_labels = batch.get("labor_labels")

            if plane_labels is not None:
                plane_labels = plane_labels.to(self.device)
            if seg_labels is not None:
                seg_labels = seg_labels.to(self.device)
            if labor_labels is not None:
                labor_labels = labor_labels.to(self.device)

            # Forward pass with mixed precision
            with autocast("cuda", enabled=cfg.use_amp):
                outputs = self.model(pixel_values)
                loss, losses = self.model.compute_loss(
                    outputs,
                    plane_labels=plane_labels,
                    seg_labels=seg_labels,
                    labor_labels=labor_labels,
                    use_uncertainty_weighting=cfg.use_uncertainty_weighting,
                )

            # Scale loss for gradient accumulation
            loss = loss / cfg.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with accumulation)
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Track losses
            total_loss += loss.item() * cfg.gradient_accumulation_steps
            for key in task_losses:
                if key in losses:
                    task_losses[key] += losses[key].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * cfg.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log to wandb
            if self.wandb_run and self.global_step % cfg.log_every_n_steps == 0:
                import wandb
                log_dict = {
                    "train/loss": loss.item() * cfg.gradient_accumulation_steps,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                }
                for key, val in losses.items():
                    log_dict[f"train/{key}_loss"] = val.item()

                # Log uncertainty weights
                if cfg.use_uncertainty_weighting:
                    weights = self.model.uncertainty_weighting.get_weights()
                    for key, val in weights.items():
                        log_dict[f"train/{key}_weight"] = val

                wandb.log(log_dict)

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}

        return {"loss": avg_loss, **avg_task_losses}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        task_losses = {"classification": 0, "segmentation": 0, "regression": 0}
        num_batches = 0

        # Metrics tracking
        correct_planes = 0
        total_planes = 0
        seg_iou_sum = 0
        seg_iou_count = 0
        param_mae_sum = 0
        param_mae_count = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(self.device)
            plane_labels = batch.get("plane_labels")
            seg_labels = batch.get("seg_labels")
            labor_labels = batch.get("labor_labels")

            if plane_labels is not None:
                plane_labels = plane_labels.to(self.device)
            if seg_labels is not None:
                seg_labels = seg_labels.to(self.device)
            if labor_labels is not None:
                labor_labels = labor_labels.to(self.device)

            with autocast("cuda", enabled=self.config.training.use_amp):
                outputs = self.model(pixel_values)
                loss, losses = self.model.compute_loss(
                    outputs,
                    plane_labels=plane_labels,
                    seg_labels=seg_labels,
                    labor_labels=labor_labels,
                    use_uncertainty_weighting=False,  # Use raw losses for validation
                )

            total_loss += loss.item()
            for key in task_losses:
                if key in losses:
                    task_losses[key] += losses[key].item()
            num_batches += 1

            # Classification accuracy
            if plane_labels is not None:
                preds = outputs.plane_logits.argmax(dim=1)
                correct_planes += (preds == plane_labels).sum().item()
                total_planes += plane_labels.size(0)

            # Segmentation IoU
            if seg_labels is not None:
                seg_preds = outputs.seg_masks.argmax(dim=1)
                iou = self._compute_iou(seg_preds, seg_labels)
                seg_iou_sum += iou
                seg_iou_count += 1

            # Regression MAE
            if labor_labels is not None:
                mae = (outputs.labor_params - labor_labels).abs().mean().item()
                param_mae_sum += mae
                param_mae_count += 1

        # Compute averages
        metrics = {
            "loss": total_loss / num_batches,
            **{k: v / num_batches for k, v in task_losses.items()},
        }

        if total_planes > 0:
            metrics["plane_accuracy"] = correct_planes / total_planes

        if seg_iou_count > 0:
            metrics["seg_miou"] = seg_iou_sum / seg_iou_count

        if param_mae_count > 0:
            metrics["param_mae"] = param_mae_sum / param_mae_count

        return metrics

    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute mean IoU for segmentation"""
        num_classes = self.config.model.num_seg_classes
        ious = []

        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()

            if union > 0:
                ious.append(intersection / union)

        return sum(ious) / len(ious) if ious else 0.0

    def train(self):
        """Full training loop"""
        cfg = self.config.training

        print(f"\n{'='*60}")
        print(f"Starting training: {cfg.experiment_name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {cfg.num_epochs}")
        print(f"Batch size: {cfg.batch_size} x {cfg.gradient_accumulation_steps} (effective: {cfg.batch_size * cfg.gradient_accumulation_steps})")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        patience_counter = 0

        for epoch in range(cfg.num_epochs):
            self.epoch = epoch

            # Check if we should unfreeze encoder
            if (
                self.config.model.freeze_encoder
                and epoch == self.config.model.unfreeze_encoder_epoch
            ):
                print(f"\n>>> Unfreezing encoder at epoch {epoch}")
                self.model.unfreeze_encoder()
                # Reinitialize optimizer with encoder params
                self.optimizer = self._setup_optimizer()
                self.scheduler = self._setup_scheduler()

            # Train
            train_metrics = self.train_epoch()
            self.history["train"].append(train_metrics)

            # Validate
            val_metrics = self.validate()
            self.history["val"].append(val_metrics)

            # Log
            print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            if "plane_accuracy" in val_metrics:
                print(f"  Plane Acc:  {val_metrics['plane_accuracy']:.4f}")
            if "seg_miou" in val_metrics:
                print(f"  Seg mIoU:   {val_metrics['seg_miou']:.4f}")
            if "param_mae" in val_metrics:
                print(f"  Param MAE:  {val_metrics['param_mae']:.4f}")

            # Log to wandb
            if self.wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })

            # Checkpointing
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self.save_checkpoint("best.pt")
                print(f"  >>> New best model!")
            else:
                patience_counter += 1

            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Early stopping
            if patience_counter >= cfg.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Save final model
        self.save_checkpoint("final.pt")

        # Save training history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Push to Hub
        if self.config.push_to_hub:
            self.push_to_hub()

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.output_dir / filename

        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": vars(self.config.model),
                "training": vars(self.config.training),
            },
        }, path)

        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Loaded checkpoint from {path}")

    def push_to_hub(self):
        """Push model to Hugging Face Hub"""
        from huggingface_hub import HfApi, create_repo

        hub_id = self.config.hub_model_id
        print(f"\nPushing to Hub: {hub_id}")

        try:
            # Create repo if it doesn't exist
            create_repo(hub_id, exist_ok=True)

            api = HfApi()

            # Upload best model
            api.upload_file(
                path_or_fileobj=str(self.output_dir / "best.pt"),
                path_in_repo="best.pt",
                repo_id=hub_id,
            )

            # Upload config
            config_dict = {
                "model": vars(self.config.model),
                "training": vars(self.config.training),
                "best_val_loss": self.best_val_loss,
            }
            config_path = self.output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.json",
                repo_id=hub_id,
            )

            # Create model card
            model_card = self._create_model_card()
            card_path = self.output_dir / "README.md"
            with open(card_path, "w") as f:
                f.write(model_card)

            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=hub_id,
            )

            print(f"Successfully pushed to https://huggingface.co/{hub_id}")

        except Exception as e:
            print(f"Failed to push to Hub: {e}")

    def _create_model_card(self) -> str:
        """Create model card for Hub"""
        return f"""---
tags:
- medical
- ultrasound
- obstetrics
- segmentation
- classification
- medgemma
- edge-ai
license: cc-by-4.0
datasets:
- maternal-fetal-ultrasound
---

# LaborView AI

Multi-task model for intrapartum ultrasound analysis, fine-tuned from MedGemma.

## Model Description

LaborView AI is designed for point-of-care labor monitoring using ultrasound.
It performs three tasks simultaneously:

1. **Standard Plane Classification**: Identifies the ultrasound view type
2. **Segmentation**: Segments pubic symphysis and fetal head
3. **Labor Parameter Estimation**: Predicts Angle of Progression (AoP) and Head-Symphysis Distance (HSD)

## Intended Use

- Labor progress monitoring in clinical settings
- Point-of-care ultrasound assistance
- Training and education for sonographers

## Training

- **Base Model**: MedGemma (google/medgemma-4b-it)
- **Dataset**: Maternal-Fetal Ultrasound Dataset
- **Best Validation Loss**: {self.best_val_loss:.4f}

## Edge Deployment

This model is designed for mobile deployment. See the `edge/` directory for:
- ONNX export
- Core ML conversion (iOS)
- TFLite conversion (Android)

## Citation

Part of the MedGemma Impact Challenge submission.
"""


def main():
    """Main training entrypoint"""
    import argparse

    parser = argparse.ArgumentParser(description="Train LaborView AI")
    parser.add_argument("--data-dir", type=str, default="./data/DatasetV3",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--edge", action="store_true",
                        help="Use edge-optimized configuration")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Get config
    config = get_config(edge_mode=args.edge)

    # Override with CLI args
    config.data.data_dir = Path(args.data_dir)
    config.training.output_dir = Path(args.output_dir)

    if args.experiment:
        config.training.experiment_name = args.experiment
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "edge" if args.edge else "cloud"
        config.training.experiment_name = f"laborview_{mode}_{timestamp}"

    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    print(f"\nCreating model: {config.model.encoder_name}")
    model = create_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Create dataloaders
    print(f"\nLoading dataset from {config.data.data_dir}")
    loaders = create_dataloaders(config, splits=["train", "val"])

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
