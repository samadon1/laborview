---
title: LaborView AI
emoji: "👶"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - samwell/laborview-medsiglip
  - google/medsiglip-448
tags:
  - medical
  - ultrasound
  - segmentation
  - medgemma
  - labor-monitoring
---

# LaborView AI - Intrapartum Ultrasound Analysis

AI-powered labor monitoring using **MedSigLIP** from Google's MedGemma family.

## Features

- **Automated Segmentation** - Identifies pubic symphysis and fetal head
- **Angle of Progression (AoP)** - Key clinical indicator for labor progress
- **Head-Symphysis Distance (HSD)** - Distance measurement between structures
- **Clinical Assessment** - AI-generated labor progress evaluation

## Model

Built on [MedSigLIP](https://huggingface.co/google/medsiglip-448) (878M parameters), fine-tuned on transperineal ultrasound images.

- **Validation IoU:** 90%
- **Input:** 448x448 RGB ultrasound images
- **Output:** 3-class segmentation (background, symphysis, fetal head) + plane classification

## Clinical Metrics

### Angle of Progression (AoP)
- < 110: Early labor - head not engaged
- 110-120: Active labor - head descending
- 120-140: Advanced labor - good progress
- > 140: Late labor - delivery imminent

### Head-Symphysis Distance (HSD)
Measures the distance between inferior symphysis and closest point of fetal head.

## Disclaimer

This is a research prototype developed for the **MedGemma Impact Challenge**. It is NOT intended for clinical use. Always consult qualified medical professionals for clinical decisions.

## Citation

If you use this work, please cite:
```
@misc{laborview2025,
  title={LaborView AI: MedSigLIP-based Intrapartum Ultrasound Analysis},
  author={Team LaborView},
  year={2025},
  publisher={HuggingFace}
}
```
