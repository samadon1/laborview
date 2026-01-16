"""
LaborView AI - Configuration
Edge-optimized multi-task model for intrapartum ultrasound analysis
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder options: "medgemma", "siglip", "mobilevit", "efficientnet"
    encoder_name: str = "medgemma"
    encoder_pretrained: str = "google/medgemma-4b-it"

    # For edge deployment, can swap to smaller encoders
    edge_encoder_name: str = "mobilevit"
    edge_encoder_pretrained: str = "apple/mobilevit-small"

    # Feature dimensions
    encoder_hidden_dim: int = 1152  # SigLIP output dim
    projection_dim: int = 512  # Shared feature dimension

    # Task heads
    num_plane_classes: int = 6  # Standard plane categories
    num_seg_classes: int = 3   # Background, pubic symphysis, fetal head
    num_labor_params: int = 2  # AoP (angle), HSD (distance)

    # Segmentation decoder
    seg_decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Freeze encoder initially
    freeze_encoder: bool = True
    unfreeze_encoder_epoch: int = 3

    # Dropout
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_dir: Path = Path("./data/DatasetV3")
    image_size: int = 384  # MedGemma's expected input size

    # For edge, can use smaller images
    edge_image_size: int = 256

    # Augmentation
    use_augmentation: bool = True

    # Data splits (if not predefined)
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Dataloader
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    learning_rate: float = 1e-4
    encoder_lr: float = 1e-5  # Lower LR for pretrained encoder
    weight_decay: float = 0.01

    # Schedule
    num_epochs: int = 50
    warmup_epochs: int = 2

    # Batch size
    batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Loss weights for multi-task learning
    loss_weights: dict = field(default_factory=lambda: {
        "classification": 1.0,
        "segmentation": 2.0,  # Segmentation is harder
        "regression": 1.0,
    })

    # Use uncertainty weighting (learned loss weights)
    use_uncertainty_weighting: bool = True

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10

    # Logging
    log_every_n_steps: int = 10

    # Output
    output_dir: Path = Path("./outputs")
    experiment_name: str = "laborview_v1"


@dataclass
class EdgeConfig:
    """Edge deployment configuration"""
    # Quantization
    quantization: str = "int8"  # "fp16", "int8", "int4"

    # Export formats
    export_onnx: bool = True
    export_coreml: bool = True  # iOS
    export_tflite: bool = True  # Android

    # Optimization
    optimize_for_mobile: bool = True

    # Target devices
    ios_min_version: str = "15.0"
    android_min_sdk: int = 26


@dataclass
class LaborViewConfig:
    """Main configuration combining all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)

    # Hub upload
    hub_model_id: str = "samwell/laborview-ultrasound"
    push_to_hub: bool = True

    # Reproducibility
    seed: int = 42


def get_config(edge_mode: bool = False) -> LaborViewConfig:
    """Get configuration, optionally optimized for edge"""
    config = LaborViewConfig()

    if edge_mode:
        config.model.encoder_name = config.model.edge_encoder_name
        config.model.encoder_pretrained = config.model.edge_encoder_pretrained
        config.model.encoder_hidden_dim = 256  # MobileViT hidden dim
        config.data.image_size = config.data.edge_image_size
        config.model.freeze_encoder = False  # Fine-tune smaller model fully

    return config
