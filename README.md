# LaborView AI

**Multimodal AI for Intrapartum Care in Resource-Limited Settings**

[![HuggingFace](https://img.shields.io/badge/🤗-Collection-yellow)](https://huggingface.co/collections/samwell/laborview-ai-the-medgemma-impact-challenge)
[![Demo](https://img.shields.io/badge/🚀-Demo-blue)](https://huggingface.co/spaces/samwell/laborview-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## Overview

LaborView is an end-to-end AI pipeline for labor monitoring, combining:

| Model | Modality | Task | Size |
|-------|----------|------|------|
| **[MedASR-Ghana](https://huggingface.co/samwell/medasr-ghana)** | Audio | Ghanaian English ASR | 105M |
| **[LaborView-MedSigLIP](https://huggingface.co/samwell/laborview-medsiglip)** | Vision | Multi-task ultrasound (full) | 400M |
| **[LaborView-Ultrasound](https://huggingface.co/samwell/laborview-ultrasound)** | Vision | Multi-task ultrasound (edge) | 5.6M |
| **[MedGemma](https://huggingface.co/google/medgemma-4b-it)** | Text | Clinical interpretation | 4B |

```
┌─────────────────────────────────────────────────────────────────┐
│                    LABORVIEW AI PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   🎤 VOICE                    📷 ULTRASOUND                     │
│      │                              │                           │
│      ▼                              ▼                           │
│  ┌────────────┐              ┌─────────────────┐                │
│  │  MedASR    │              │   LaborView     │                │
│  │  Ghana     │              │   (MedSigLIP)   │                │
│  └─────┬──────┘              └────────┬────────┘                │
│        │                              │                         │
│        ▼                              ▼                         │
│  ┌───────────┐               ┌────────────────┐                 │
│  │Transcribed│               │• Segmentation  │                 │
│  │  Clinical │               │• Classification│                 │
│  │   Notes   │               │• AoP, HSD      │                 │
│  └─────┬─────┘               └───────┬────────┘                 │
│        │                             │                          │
│        └──────────┬──────────────────┘                          │
│                   ▼                                             │
│          ┌───────────────┐                                      │
│          │   MedGemma    │                                      │
│          │  (Clinical    │                                      │
│          │Interpretation)│                                      │
│          └───────┬───────┘                                      │
│                  ▼                                              │
│          ┌───────────────┐                                      │
│          │   CLINICAL    │                                      │
│          │    REPORT     │                                      │
│          └───────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Problem

**Obstructed labor** causes ~8% of maternal deaths globally, disproportionately affecting sub-Saharan Africa. Key challenges:

- Subjective assessment of labor progress
- Shortage of trained sonographers
- Limited documentation in busy labor wards
- Language barriers with clinical AI tools

## Our Solution

LaborView provides:

1. **Objective Measurements**: AI-computed Angle of Progression (AoP) and Head-Symphysis Distance (HSD)
2. **Multi-task Analysis**: Segmentation + classification + regression in one forward pass
3. **Voice Documentation**: MedASR transcribes Ghanaian-accented English (Twi, Akan, Fante)
4. **Edge Deployment**: 21MB model runs on mobile devices for point-of-care use

---

## Multi-Task Model Outputs

LaborView performs three tasks simultaneously:

| Task | Output | Description |
|------|--------|-------------|
| **Segmentation** | 3-class mask | Pubic symphysis, fetal head, background |
| **Classification** | 6-class logits | Ultrasound plane detection |
| **Regression** | 2 values | Direct AoP (degrees) and HSD (pixels) |

### Clinical Interpretation

| AoP Range | Stage | Action |
|-----------|-------|--------|
| < 110° | Early labor | Monitor closely |
| 110-120° | Active labor | Continue |
| 120-140° | Advanced | Good progress |
| > 140° | Late labor | Prepare delivery |

---

## Quick Start

### Installation

```bash
git clone https://github.com/samadon1/laborview
cd laborview
pip install -r requirements.txt
```

### Inference (ONNX)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("laborview.onnx")

# Preprocess
image = Image.open("ultrasound.png").convert("RGB").resize((256, 256))
img = np.array(image).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

# Multi-task inference
plane_logits, seg_masks, labor_params = session.run(None, {"image": img})

# Parse outputs
plane = np.argmax(plane_logits)
mask = np.argmax(seg_masks, axis=1)[0]
aop, hsd = labor_params[0]

print(f"AoP: {aop:.1f}°, HSD: {hsd:.1f}px")
```

### Clinical Metrics

```python
from src.clinical_metrics import compute_all_metrics

metrics = compute_all_metrics(mask, symphysis_class=1, head_class=2)

print(f"AoP: {metrics.aop:.1f}° - {metrics.aop_interpretation}")
print(f"HSD: {metrics.hsd:.1f}px - {metrics.hsd_interpretation}")
print(f"Progress: {metrics.labor_progress}")
print(f"Recommendation: {metrics.recommendation}")
```

---

## Project Structure

```
laborview/
├── README.md                 # This file
├── requirements.txt          # Dependencies
│
├── docs/
│   ├── PIPELINE.md           # End-to-end flow documentation
│   ├── MODEL_CARD_MEDSIGLIP.md
│   └── MODEL_CARD_ULTRASOUND.md
│
├── src/
│   ├── model.py              # Multi-task architecture
│   ├── config.py             # Configuration
│   ├── dataset.py            # Data loading
│   └── clinical_metrics.py   # AoP, HSD, HC computation
│
├── training/
│   ├── train_medsiglip.py    # MedSigLIP fine-tuning
│   ├── train.py              # General training script
│   └── train_hf_job.py       # HuggingFace Jobs runner
│
├── export/
│   ├── export_medsiglip.py   # ONNX export (full model)
│   ├── export_edge.py        # ONNX export (edge model)
│   └── edge_export.py        # CoreML/TFLite export
│
├── demo/
│   ├── demo.py               # Basic demo
│   ├── demo_medsiglip.py     # MedSigLIP demo
│   └── test_onnx_model.py    # ONNX inference test
│
├── flutter_app/              # Mobile app (iOS/Android)
│   ├── lib/
│   │   ├── services/
│   │   │   └── laborview_service.dart
│   │   └── screens/
│   │       └── analysis_screen.dart
│   └── pubspec.yaml
│
├── spaces/                   # HuggingFace Spaces demo
│   └── app.py
│
└── assets/
    └── test_ultrasound.jpg
```

---

## Training

### MedSigLIP (Full Model)

```bash
# Run on HuggingFace Jobs (GPU)
python training/train_medsiglip.py
```

**Config:**
- Base: `google/medsiglip-448`
- Input: 448×448
- Epochs: 30 (3 frozen + 27 fine-tuning)
- Loss: Dice + CE (segmentation), CE (classification), SmoothL1 (regression)

### Edge Model

```bash
python training/train.py --edge
```

**Config:**
- Input: 256×256
- Size: ~21MB ONNX

---

## Export

### ONNX

```bash
python export/export_edge.py --checkpoint best.pt --output laborview.onnx
```

### CoreML (iOS)

```bash
python export/edge_export.py --format coreml
```

### TFLite (Android)

```bash
python export/edge_export.py --format tflite
```

---

## Mobile App

Flutter app for iOS/Android deployment:

```bash
cd flutter_app
flutter pub get
flutter run
```

Features:
- Real-time camera capture
- On-device ONNX inference
- Segmentation overlay
- Clinical metrics display
- Voice notes (MedASR integration planned)

---

## Models on HuggingFace

| Model | Link | Description |
|-------|------|-------------|
| MedASR-Ghana | [samwell/medasr-ghana](https://huggingface.co/samwell/medasr-ghana) | Ghanaian English ASR (105M) |
| LaborView-MedSigLIP | [samwell/laborview-medsiglip](https://huggingface.co/samwell/laborview-medsiglip) | Full multi-task model (400M) |
| LaborView-Ultrasound | [samwell/laborview-ultrasound](https://huggingface.co/samwell/laborview-ultrasound) | Edge model (5.6M) |

---

## Sample Report

```
╔═══════════════════════════════════════════════════════════════╗
║              INTRAPARTUM ULTRASOUND ASSESSMENT                ║
╠═══════════════════════════════════════════════════════════════╣
║ AI MEASUREMENTS (LaborView)                                   ║
║ ─────────────────────────────────────────────────────────────║
║ Angle of Progression:     127.3°                              ║
║ Head-Symphysis Distance:  45.2 px                             ║
║ Ultrasound Plane:         Transperineal (standard)            ║
║ Labor Progress:           NORMAL ✓                            ║
╠═══════════════════════════════════════════════════════════════╣
║ CLINICAL NOTES (MedASR Transcription)                         ║
║ ─────────────────────────────────────────────────────────────║
║ Patient is a 28-year-old G2P1 at 39 weeks. Cervix 8           ║
║ centimeters dilated. Good uterine contractions every 3        ║
║ minutes. Fetal heart rate 140, reactive.                      ║
╠═══════════════════════════════════════════════════════════════╣
║ RECOMMENDATION                                                ║
║ Labor progressing well. Continue routine monitoring.          ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Why Ghana?

- **MedASR-Ghana** understands Twi, Akan, Fante-accented English
- Designed for resource-limited settings
- Edge models work offline in rural clinics
- Voice documentation reduces paperwork burden

---

## Citation

```bibtex
@software{laborview_2024,
  title = {LaborView AI: Multimodal Pipeline for Intrapartum Care},
  author = {Samuel},
  year = {2024},
  url = {https://huggingface.co/collections/samwell/laborview-ai-the-medgemma-impact-challenge}
}
```

---

## License

Apache 2.0

---

## Acknowledgments

- [HAI-DEF Challenge](https://hai-def.org/) - Dataset
- [Google MedSigLIP](https://huggingface.co/google/medsiglip-448) - Base encoder
- [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) - ASR dataset
- MedGemma Impact Challenge

---

*Built for the MedGemma Impact Challenge — AI for maternal health.*
