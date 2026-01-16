---
license: apache-2.0
library_name: transformers
pipeline_tag: image-segmentation
tags:
  - medical
  - ultrasound
  - obstetrics
  - segmentation
  - classification
  - regression
  - multi-task
  - intrapartum
  - labor-monitoring
  - onnx
  - medsiglip
  - siglip
  - clinical-ai
datasets:
  - custom
language:
  - en
metrics:
  - iou
  - dice
  - accuracy
  - mae
base_model:
  - google/medsiglip-448
---

# LaborView MedSigLIP

**Multi-task AI model for intrapartum ultrasound analysis during labor**

## Model Description

LaborView MedSigLIP is a **multi-task vision model** for comprehensive analysis of transperineal ultrasound during labor. Unlike single-task segmentation models, it simultaneously performs:

| Task | Output | Description |
|------|--------|-------------|
| **Segmentation** | 3-class mask (H×W) | Pubic symphysis, fetal head, background |
| **Classification** | 6-class logits | Standard ultrasound plane detection |
| **Regression** | 2 values | Direct AoP and HSD predictions |

### Why Multi-Task?

- **Efficiency**: Single forward pass for all outputs
- **Shared Features**: Tasks benefit from shared visual representations
- **Clinical Workflow**: Provides complete assessment, not just masks
- **Uncertainty Weighting**: Learned task weights balance losses automatically

### Architecture

```
Input Image (448×448 RGB)
         │
         ▼
┌─────────────────────────┐
│      MedSigLIP          │  Vision Encoder
│   (SigLIP-SO400M)       │  1152-dim features
│   google/medsiglip-448  │
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌─────────────┐
│ Pooled  │  │  Sequence   │
│Features │  │  Features   │
│ (1152)  │  │(N×1152)     │
└────┬────┘  └──────┬──────┘
     │              │
     ▼              ▼
┌─────────┐  ┌─────────────┐
│Projector│  │ Seg Decoder │
│  (512)  │  │ (FPN-style) │
└────┬────┘  └──────┬──────┘
     │              │
  ┌──┴──┐           │
  ▼     ▼           ▼
┌────┐┌────┐  ┌──────────┐
│Cls ││Reg │  │ Seg Mask │
│Head││Head│  │(3×H×W)   │
└────┘└────┘  └──────────┘
  │     │          │
  ▼     ▼          ▼
Plane  AoP,    Symphysis,
Logits HSD     Head Masks
(6)    (2)     (3×448×448)
```

### Model Outputs

```python
@dataclass
class LaborViewOutput:
    plane_logits: Tensor    # (B, 6) - Standard plane classification
    seg_masks: Tensor       # (B, 3, H, W) - Segmentation masks
    labor_params: Tensor    # (B, 2) - [AoP degrees, HSD pixels]
```

## Training

- **Dataset**: [HAI-DEF Challenge](https://zenodo.org/records/17655183) - Transperineal ultrasound with expert annotations
- **Base Model**: `google/medsiglip-448` (1152-dim, ~400M encoder params)
- **Multi-Task Loss**: Uncertainty-weighted combination (Kendall et al.)
  - Segmentation: Dice + Cross-Entropy
  - Classification: Cross-Entropy
  - Regression: Smooth L1
- **Training Strategy**:
  - Epochs 1-3: Frozen encoder (head warmup)
  - Epochs 4+: Full fine-tuning with gradient checkpointing
  - OneCycleLR scheduler, 5e-5 max LR
- **Augmentation**: HorizontalFlip, RandomBrightnessContrast, GaussNoise, ShiftScaleRotate

## Intended Use

### Primary Use Cases

1. **Automated Labor Assessment**: Real-time analysis of labor progress
2. **Clinical Decision Support**: AI-assisted measurements for clinicians
3. **Training/Education**: Teaching tool for ultrasound interpretation
4. **Research**: Standardized measurement extraction for studies

### Output Interpretation

#### Segmentation Classes

| Class | ID | Color | Anatomical Structure |
|-------|-----|-------|---------------------|
| Background | 0 | Transparent | Non-anatomical |
| Pubic Symphysis | 1 | Cyan | Pelvic joint landmark |
| Fetal Head | 2 | Magenta | Presenting fetal part |

#### Plane Classification

| Class | Description |
|-------|-------------|
| 0 | Transperineal (standard) |
| 1 | Transabdominal |
| 2 | Oblique |
| 3 | Sagittal |
| 4 | Axial |
| 5 | Other/Non-standard |

#### Labor Parameters

| Parameter | Range | Clinical Meaning |
|-----------|-------|------------------|
| **AoP** (Angle of Progression) | 90-160° | Head descent angle |
| **HSD** (Head-Symphysis Distance) | 0-100+ px | Head-to-pelvis distance |

**AoP Interpretation:**

| AoP | Stage | Status |
|-----|-------|--------|
| < 110° | Early labor | Head not engaged |
| 110-120° | Active labor | Descending |
| 120-140° | Advanced | Good progress |
| > 140° | Late labor | Delivery imminent |

### Users

- Obstetric ultrasound software developers
- Medical device manufacturers
- Clinical researchers in maternal-fetal medicine
- Healthcare AI developers
- Medical education platforms

### Out of Scope

- Direct clinical diagnosis without physician oversight
- Replacement for clinical judgment
- Non-transperineal ultrasound views
- Fetal anomaly or malformation detection
- Gestational age estimation

## How to Use

### PyTorch Inference

```python
import torch
from model import LaborViewMedSigLIP
from config import Config

# Load model
config = Config()
model = LaborViewMedSigLIP(config)
checkpoint = torch.load("best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
image = preprocess_image("ultrasound.png")  # (1, 3, 448, 448)
with torch.no_grad():
    plane_logits, seg_masks = model(image)

# Parse outputs
plane_class = plane_logits.argmax(dim=1).item()
seg_mask = seg_masks.argmax(dim=1)[0].numpy()
```

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("laborview.onnx")

# Preprocess
image = Image.open("ultrasound.png").convert("RGB").resize((448, 448))
img = np.array(image).astype(np.float32) / 255.0
img = (img - 0.5) / 0.5  # MedSigLIP normalization [-1, 1]
img = img.transpose(2, 0, 1)[np.newaxis, ...]

# Run multi-task inference
plane_logits, seg_masks, labor_params = session.run(None, {"image": img})

# Parse all outputs
plane_class = np.argmax(plane_logits, axis=1)[0]
seg_mask = np.argmax(seg_masks, axis=1)[0]
aop, hsd = labor_params[0]

print(f"Plane: {['transperineal','transabdominal','oblique','sagittal','axial','other'][plane_class]}")
print(f"AoP: {aop:.1f}°, HSD: {hsd:.1f}px")
```

### Clinical Metrics from Segmentation

```python
from clinical_metrics import compute_all_metrics

# Compute comprehensive clinical assessment
metrics = compute_all_metrics(
    segmentation_mask=seg_mask,
    symphysis_class=1,
    head_class=2
)

print(f"Angle of Progression: {metrics.aop:.1f}°")
print(f"  → {metrics.aop_interpretation}")
print(f"Head-Symphysis Distance: {metrics.hsd:.1f} px")
print(f"  → {metrics.hsd_interpretation}")
print(f"Head Circumference: {metrics.head_circumference:.0f} px")
print(f"Head Area: {metrics.head_area:.0f} px²")
print(f"Segmentation Quality: {metrics.segmentation_quality} ({metrics.confidence:.0%})")
print(f"Labor Progress: {metrics.labor_progress.upper()}")
print(f"Recommendation: {metrics.recommendation}")
```

## Model Files

| File | Description | Size |
|------|-------------|------|
| `best.pt` | Best validation checkpoint | ~1.6 GB |
| `final.pt` | Final epoch checkpoint | ~1.6 GB |
| `laborview.onnx` | ONNX export (all heads) | ~1.6 GB |
| `config.json` | Model configuration | 1 KB |

## Performance

### Multi-Task Metrics

| Task | Metric | Value |
|------|--------|-------|
| Segmentation | Mean IoU | TBD |
| Segmentation | Dice Score | TBD |
| Classification | Accuracy | TBD |
| Regression (AoP) | MAE | TBD |
| Regression (HSD) | MAE | TBD |

### Inference Speed

| Platform | Resolution | Latency |
|----------|------------|---------|
| NVIDIA A100 | 448×448 | ~15ms |
| Apple M1 | 448×448 | ~50ms |
| CPU (8 cores) | 448×448 | ~200ms |

## Limitations

1. **Training Data**: Single dataset/protocol; may need fine-tuning for different equipment
2. **Population Coverage**: May not generalize to all patient demographics
3. **Image Quality Dependence**: Degrades with poor quality, shadows, artifacts
4. **Anatomical Variations**: May struggle with unusual presentations
5. **Calibration Required**: Pixel values need device-specific mm conversion
6. **Regression vs Computed**: Direct AoP/HSD predictions may differ from geometry-computed values

## Ethical Considerations

- **Decision Support Only**: Not a replacement for clinical judgment
- **Validation Required**: Must validate on local populations before deployment
- **Bias Monitoring**: Monitor performance across demographic groups
- **Regulatory Compliance**: FDA/CE approval required for clinical use
- **Transparency**: Always disclose AI assistance to patients

## Citation

```bibtex
@software{laborview_medsiglip_2024,
  title = {LaborView MedSigLIP: Multi-Task AI for Intrapartum Ultrasound},
  author = {Samuel},
  year = {2024},
  url = {https://huggingface.co/samwell/laborview-medsiglip},
  note = {Multi-task model: segmentation + classification + regression}
}
```

## Related Resources

- [laborview-ultrasound](https://huggingface.co/samwell/laborview-ultrasound) - Edge-optimized variant (~21MB)
- [Demo Space](https://huggingface.co/spaces/samwell/laborview-demo) - Try online
- [HAI-DEF Challenge](https://hai-def.org/) - Dataset and competition
- [MedSigLIP](https://huggingface.co/google/medsiglip-448) - Base encoder

## License

Apache 2.0
