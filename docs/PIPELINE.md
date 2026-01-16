# LaborView AI: End-to-End Clinical Pipeline

**Multimodal AI for Intrapartum Care in Resource-Limited Settings**

---

## Overview

LaborView combines three AI models into a unified pipeline for labor monitoring:

| Model | Modality | Task | Size |
|-------|----------|------|------|
| **MedASR-Ghana** | Audio | Speech-to-text (Ghanaian English) | 105M |
| **LaborView-MedSigLIP** | Vision | Multi-task ultrasound analysis | 400M |
| **LaborView-Ultrasound** | Vision | Edge-optimized for mobile | 5.6M |

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
│  │  Ghana     │              │   MedSigLIP or  │                │
│  │  (105M)    │              │   MobileViT     │                │
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
│          │   CLINICAL    │                                      │
│          │    REPORT     │                                      │
│          └───────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Pipeline

### Step 1: Ultrasound Capture

Clinician performs transperineal ultrasound. The device captures frames showing:
- Pubic symphysis (pelvic landmark)
- Fetal head (presenting part)

### Step 2: AI Analysis (LaborView)

LaborView processes each frame with three simultaneous outputs:

```
Input: Ultrasound frame (256×256 or 448×448)
          │
          ▼
    ┌─────────────┐
    │  LaborView  │
    │   Model     │
    └──────┬──────┘
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
   Seg   Class  Reg
    │      │      │
    ▼      ▼      ▼
  Mask  Plane   AoP
  (3×H×W) (6)   HSD
                (2)
```

**Outputs:**
- **Segmentation**: Pixel-wise mask identifying symphysis and fetal head
- **Classification**: Ultrasound plane type (transperineal, transabdominal, etc.)
- **Regression**: Direct AoP (angle) and HSD (distance) predictions

### Step 3: Clinical Metrics

From the segmentation mask, compute comprehensive measurements:

```python
metrics = compute_all_metrics(segmentation_mask)

# Returns:
# - Angle of Progression (AoP): 127.3°
# - Head-Symphysis Distance (HSD): 45.2 px
# - Head Circumference: 892 px
# - Head Area: 15,420 px²
# - Segmentation Quality: good (87%)
# - Labor Progress: NORMAL
# - Recommendation: "Continue routine monitoring"
```

### Step 4: Voice Documentation (MedASR)

Clinician speaks observations in Ghanaian-accented English:

```
🎤 "Patient is a 28-year-old G2P1 at 39 weeks.
    Cervix 8 centimeters dilated.
    Good uterine contractions every 3 minutes.
    Fetal heart rate 140, reactive."
          │
          ▼
    ┌─────────────┐
    │   MedASR    │
    │   Ghana     │
    └──────┬──────┘
          │
          ▼
    📝 Transcribed text
```

**Why MedASR-Ghana?**
- Understands Twi, Akan, Fante-influenced English
- Medical vocabulary (cervix, G2P1, reactive)
- Hands-free during active labor

### Step 5: Report Generation

Combine AI measurements with clinician notes:

```
╔═══════════════════════════════════════════════════════════════╗
║              INTRAPARTUM ULTRASOUND ASSESSMENT                ║
╠═══════════════════════════════════════════════════════════════╣
║ Date: 2024-01-16 14:32                                        ║
║ Facility: Korle Bu Teaching Hospital                          ║
╠═══════════════════════════════════════════════════════════════╣
║ AI MEASUREMENTS (LaborView)                                   ║
║ ─────────────────────────────────────────────────────────────║
║ Angle of Progression:     127.3°                              ║
║ Head-Symphysis Distance:  45.2 px                             ║
║ Head Circumference:       892 px                              ║
║ Ultrasound Plane:         Transperineal (standard)            ║
║ Labor Progress:           NORMAL ✓                            ║
║ AI Confidence:            87%                                 ║
╠═══════════════════════════════════════════════════════════════╣
║ CLINICAL NOTES (MedASR Transcription)                         ║
║ ─────────────────────────────────────────────────────────────║
║ Patient is a 28-year-old G2P1 at 39 weeks. Cervix 8           ║
║ centimeters dilated. Good uterine contractions every 3        ║
║ minutes. Fetal heart rate 140, reactive.                      ║
╠═══════════════════════════════════════════════════════════════╣
║ RECOMMENDATION                                                ║
║ ─────────────────────────────────────────────────────────────║
║ Labor progressing well. Continue routine monitoring.          ║
║ Reassess in 30-60 minutes.                                    ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## Deployment Options

### Cloud (Full Accuracy)
```
MedSigLIP (448×448) → Best segmentation quality
MedASR → Server-side transcription
```

### Edge/Mobile (Real-time)
```
MobileViT (256×256) → 21MB, <50ms on phone
MedASR ONNX → On-device transcription
```

### Hybrid
```
MobileViT on device → Real-time feedback
Upload frames → MedSigLIP for final report
```

---

## Why This Matters

### For Ghana & West Africa

1. **Language**: MedASR understands local accents—no need to code-switch
2. **Infrastructure**: Edge models work offline, critical for rural clinics
3. **Workforce**: AI assists midwives with objective measurements
4. **Documentation**: Voice notes reduce paperwork burden

### Clinical Impact

| Without AI | With LaborView Pipeline |
|------------|------------------------|
| Subjective assessment | Quantified AoP/HSD |
| Manual documentation | Voice-transcribed notes |
| Delayed interpretation | Real-time feedback |
| Expert-dependent | Standardized measurements |

### Obstructed Labor

Obstructed labor causes ~8% of maternal deaths globally, disproportionately in sub-Saharan Africa. Early detection through objective measurements (AoP < 110°) can prompt timely intervention.

---

## Technical Stack

```
┌─────────────────────────────────────────────┐
│              FLUTTER APP                    │
├─────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐         │
│  │ Camera/     │    │ Microphone  │         │
│  │ Image Pick  │    │ Recording   │         │
│  └──────┬──────┘    └──────┬──────┘         │
│         │                  │                │
│         ▼                  ▼                │
│  ┌─────────────┐    ┌─────────────┐         │
│  │ ONNX        │    │ ONNX        │         │
│  │ Runtime     │    │ Runtime     │         │
│  │ (LaborView) │    │ (MedASR)    │         │
│  └──────┬──────┘    └──────┬──────┘         │
│         │                  │                │
│         ▼                  ▼                │
│  ┌─────────────────────────────────┐        │
│  │      Report Generator           │        │
│  │   (Metrics + Transcription)     │        │
│  └─────────────────────────────────┘        │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────┐        │
│  │   Export: PDF / FHIR / HL7      │        │
│  └─────────────────────────────────┘        │
└─────────────────────────────────────────────┘
```

---

## Models

| Model | HuggingFace | Task |
|-------|-------------|------|
| MedASR-Ghana | [samwell/medasr-ghana](https://huggingface.co/samwell/medasr-ghana) | Ghanaian English ASR |
| LaborView-MedSigLIP | [samwell/laborview-medsiglip](https://huggingface.co/samwell/laborview-medsiglip) | Full multi-task model |
| LaborView-Ultrasound | [samwell/laborview-ultrasound](https://huggingface.co/samwell/laborview-ultrasound) | Edge multi-task model |

---

## Citation

```bibtex
@software{laborview_pipeline_2024,
  title = {LaborView AI: Multimodal Pipeline for Intrapartum Care},
  author = {Samuel},
  year = {2024},
  url = {https://huggingface.co/collections/samwell/laborview-ai},
  note = {MedASR + LaborView for voice-enabled labor monitoring}
}
```

---

*Built for the MedGemma Impact Challenge — AI for maternal health in resource-limited settings.*
