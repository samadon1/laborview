"""
LaborView AI - Dataset Module
DataLoader for Maternal-Fetal Ultrasound Dataset (Zenodo 17655183)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from PIL import Image
import numpy as np

# Optional imports for augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    import torchvision.transforms as T


class UltrasoundDataset(Dataset):
    """
    Dataset for Maternal-Fetal Intrapartum Ultrasound.

    Expected directory structure (after extracting DatasetV3.zip):
    data_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── masks/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json

    Annotation format (expected):
    {
        "image_id": "xxx.png",
        "plane_class": 0-5,
        "aop": float (angle of progression),
        "hsd": float (head-symphysis distance),
        "mask_path": "masks/train/xxx_mask.png"
    }
    """

    # Standard plane class names
    PLANE_CLASSES = [
        "transperineal",
        "transabdominal",
        "oblique",
        "sagittal",
        "axial",
        "other"
    ]

    # Segmentation classes
    SEG_CLASSES = [
        "background",
        "pubic_symphysis",
        "fetal_head"
    ]

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 384,
        transform: Optional[Callable] = None,
        use_augmentation: bool = True,
        return_metadata: bool = False,
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            split: One of "train", "val", "test"
            image_size: Target image size
            transform: Custom transform (overrides default)
            use_augmentation: Whether to use augmentation (train only)
            return_metadata: Whether to return image paths and extra info
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.return_metadata = return_metadata

        # Load annotations
        self.samples = self._load_annotations()

        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(
                use_augmentation and split == "train"
            )

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_annotations(self) -> List[Dict]:
        """Load annotation file for this split"""
        anno_path = self.data_dir / "annotations" / f"{self.split}.json"

        if anno_path.exists():
            with open(anno_path, "r") as f:
                return json.load(f)
        else:
            # If no annotation file, try to infer from directory structure
            print(f"No annotation file found at {anno_path}, scanning images...")
            return self._scan_images()

    def _scan_images(self) -> List[Dict]:
        """Scan image directory and create basic annotations"""
        image_dir = self.data_dir / "images" / self.split
        mask_dir = self.data_dir / "masks" / self.split

        samples = []

        if not image_dir.exists():
            # Try flat structure
            image_dir = self.data_dir / "images"
            mask_dir = self.data_dir / "masks"

        for img_path in sorted(image_dir.glob("*.png")):
            # Look for corresponding mask
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name

            if not mask_path.exists():
                # Try same name
                mask_path = mask_dir / img_path.name

            samples.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "plane_class": None,  # Unknown
                "aop": None,
                "hsd": None,
            })

        return samples

    def _get_default_transform(self, augment: bool) -> Callable:
        """Get default transforms with optional augmentation"""

        if ALBUMENTATIONS_AVAILABLE:
            if augment:
                return A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(var_limit=(10, 50), p=0.2),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.15,
                        rotate_limit=15,
                        p=0.5
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        p=0.2
                    ),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ])
            else:
                return A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    ToTensorV2(),
                ])
        else:
            # Fallback to torchvision
            transforms = [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
            if augment:
                transforms.insert(1, T.RandomHorizontalFlip(p=0.5))

            return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img_path = sample.get("image_path") or str(
            self.data_dir / "images" / self.split / sample["image_id"]
        )
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # Load mask if available
        mask = None
        if sample.get("mask_path") and Path(sample["mask_path"]).exists():
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = np.array(mask)

        # Apply transforms
        if ALBUMENTATIONS_AVAILABLE and mask is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        elif ALBUMENTATIONS_AVAILABLE:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = self.transform(Image.fromarray(image))
            if mask is not None:
                mask = torch.from_numpy(
                    np.array(Image.fromarray(mask).resize(
                        (self.image_size, self.image_size),
                        Image.NEAREST
                    ))
                ).long()

        # Build output dict
        output = {"pixel_values": image}

        # Add labels if available
        if sample.get("plane_class") is not None:
            output["plane_labels"] = torch.tensor(sample["plane_class"], dtype=torch.long)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            output["seg_labels"] = mask

        if sample.get("aop") is not None and sample.get("hsd") is not None:
            output["labor_labels"] = torch.tensor(
                [sample["aop"], sample["hsd"]],
                dtype=torch.float32
            )

        if self.return_metadata:
            output["image_path"] = img_path
            output["sample_id"] = sample.get("image_id", Path(img_path).stem)

        return output


class VideoUltrasoundDataset(Dataset):
    """
    Dataset for video sequences - useful for temporal modeling.
    Returns sequences of frames from the same video.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        image_size: int = 384,
        sequence_length: int = 8,
        stride: int = 4,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.stride = stride

        # Group frames by video
        self.videos = self._group_by_video()

        # Create sequence indices
        self.sequences = self._create_sequences()

        # Transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()

        print(f"Created {len(self.sequences)} sequences from {len(self.videos)} videos")

    def _group_by_video(self) -> Dict[str, List[Path]]:
        """Group frames by video ID"""
        video_dir = self.data_dir / "videos" / self.split

        videos = {}
        for frame_path in sorted(video_dir.glob("*/*.png")):
            video_id = frame_path.parent.name
            if video_id not in videos:
                videos[video_id] = []
            videos[video_id].append(frame_path)

        # Sort frames within each video
        for video_id in videos:
            videos[video_id].sort(key=lambda p: int(p.stem.split("_")[-1]))

        return videos

    def _create_sequences(self) -> List[Tuple[str, int]]:
        """Create (video_id, start_idx) tuples for all valid sequences"""
        sequences = []

        for video_id, frames in self.videos.items():
            num_frames = len(frames)
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                sequences.append((video_id, start_idx))

        return sequences

    def _get_default_transform(self) -> Callable:
        """Get transform for video frames"""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, start_idx = self.sequences[idx]
        frames = self.videos[video_id]

        # Load sequence of frames
        sequence = []
        for i in range(start_idx, start_idx + self.sequence_length):
            img = Image.open(frames[i]).convert("RGB")
            img = np.array(img)

            if ALBUMENTATIONS_AVAILABLE:
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(Image.fromarray(img))

            sequence.append(img)

        # Stack into (T, C, H, W)
        sequence = torch.stack(sequence)

        return {
            "pixel_values": sequence,
            "video_id": video_id,
            "frame_indices": list(range(start_idx, start_idx + self.sequence_length)),
        }


def create_dataloaders(
    config,
    splits: List[str] = ["train", "val", "test"],
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for specified splits.

    Args:
        config: LaborViewConfig object
        splits: List of splits to create loaders for

    Returns:
        Dict of DataLoader objects
    """
    loaders = {}

    for split in splits:
        dataset = UltrasoundDataset(
            data_dir=config.data.data_dir,
            split=split,
            image_size=config.data.image_size,
            use_augmentation=config.data.use_augmentation and split == "train",
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that handles missing labels gracefully.
    """
    collated = {}

    # Always present
    collated["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

    # Optional labels - only include if all samples have them
    if all("plane_labels" in b for b in batch):
        collated["plane_labels"] = torch.stack([b["plane_labels"] for b in batch])

    if all("seg_labels" in b for b in batch):
        collated["seg_labels"] = torch.stack([b["seg_labels"] for b in batch])

    if all("labor_labels" in b for b in batch):
        collated["labor_labels"] = torch.stack([b["labor_labels"] for b in batch])

    return collated


# Convenience function for quick testing
def get_sample_batch(data_dir: str, batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """Get a sample batch for testing"""
    dataset = UltrasoundDataset(data_dir, split="train", use_augmentation=False)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return next(iter(loader))
