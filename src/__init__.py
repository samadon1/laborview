"""
LaborView AI - Multi-task Intrapartum Ultrasound Analysis

Fine-tuned MedGemma for labor monitoring:
- Standard plane classification
- Pubic symphysis and fetal head segmentation
- Labor parameter estimation (AoP, HSD)

Built for the MedGemma Impact Challenge.
"""

from .config import LaborViewConfig, get_config
from .model import LaborViewModel, LaborViewModelEdge, create_model
from .dataset import UltrasoundDataset, create_dataloaders
from .demo import LaborViewInference

__version__ = "1.0.0"
__author__ = "samwell"

__all__ = [
    "LaborViewConfig",
    "get_config",
    "LaborViewModel",
    "LaborViewModelEdge",
    "create_model",
    "UltrasoundDataset",
    "create_dataloaders",
    "LaborViewInference",
]
