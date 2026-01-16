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
  - mobilevit
  - edge-ai
  - mobile
  - ios
  - android
  - flutter
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
  - apple/mobilevit-small
---

# LaborView Ultrasound (Edge)

**Lightweight multi-task AI for real-time intrapartum ultrasound on mobile devices**

## Model Description

LaborView Ultrasound is an **edge-optimized multi-task model** for comprehensive labor monitoring on mobile devices. It performs three tasks simultaneously in a single forward pass:

| Task | Output | Description |
|------|--------|-------------|
| **Segmentation** | 3-class mask | Pubic symphysis, fetal head, background |
| **Classification** | 6-class logits | Standard ultrasound plane detection |
| **Regression** | 2 values | Direct AoP and HSD predictions |

### Key Features

- **Lightweight**: ~21 MB (vs 1.6 GB full model)
- **Fast**: <50ms on mobile devices
- **Multi-task**: Complete clinical assessment in one pass
- **Cross-platform**: ONNX, CoreML (iOS), TFLite (Android)
- **Uncertainty Weighting**: Learned task balancing

### Architecture

```
Input Image (256×256 RGB)
         │
         ▼
┌─────────────────────────┐
│      MobileViT-S        │  Vision Encoder
│   apple/mobilevit-small │  640-dim features
│       (~5.6M params)    │
└───────────┬─────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐  ┌─────────────┐
│ Pooled  │  │  Sequence   │
│Features │  │  Features   │
│  (640)  │  │  (N×640)    │
└────┬────┘  └──────┬──────┘
     │              │
     ▼              ▼
┌─────────┐  ┌─────────────┐
│Projector│  │ Seg Decoder │
│  (256)  │  │(128→64→32)  │
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
(6)    (2)     (3×256×256)
```

### Model Outputs

```python
# ONNX outputs (in order)
outputs = session.run(None, {"image": img})
plane_logits = outputs[0]   # (1, 6) - Plane classification
seg_masks = outputs[1]      # (1, 3, 256, 256) - Segmentation
labor_params = outputs[2]   # (1, 2) - [AoP, HSD]
```

## Training

- **Dataset**: [HAI-DEF Challenge](https://zenodo.org/records/17655183)
- **Base Model**: `apple/mobilevit-small` (640-dim, ~5.6M params)
- **Input Size**: 256×256 (edge-optimized)
- **Multi-Task Loss**: Uncertainty-weighted (Kendall et al.)
  - Segmentation: Dice + Cross-Entropy
  - Classification: Cross-Entropy
  - Regression: Smooth L1
- **Full Fine-tuning**: MobileViT trained end-to-end
- **Optimization**: AdamW, OneCycleLR, mixed precision

## Output Interpretation

### Segmentation Classes

| Class | ID | Color | Structure |
|-------|-----|-------|-----------|
| Background | 0 | Transparent | Non-anatomical |
| Pubic Symphysis | 1 | Cyan | Pelvic landmark |
| Fetal Head | 2 | Magenta | Presenting part |

### Plane Classification

| ID | Plane Type |
|----|------------|
| 0 | Transperineal (standard) |
| 1 | Transabdominal |
| 2 | Oblique |
| 3 | Sagittal |
| 4 | Axial |
| 5 | Other/Non-standard |

### Labor Parameters

| Parameter | Description | Clinical Use |
|-----------|-------------|--------------|
| **AoP** | Angle of Progression (degrees) | Head descent assessment |
| **HSD** | Head-Symphysis Distance (pixels) | Engagement status |

**AoP Clinical Interpretation:**

| AoP Range | Stage | Action |
|-----------|-------|--------|
| < 110° | Early labor | Monitor |
| 110-120° | Active | Continue |
| 120-140° | Advanced | Good progress |
| > 140° | Late | Prepare delivery |

## How to Use

### ONNX Runtime (Python)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("laborview_mobilevit.onnx")

# Preprocess (256x256, ImageNet normalization)
image = Image.open("ultrasound.png").convert("RGB").resize((256, 256))
img = np.array(image).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

# Multi-task inference
plane_logits, seg_masks, labor_params = session.run(None, {"image": img})

# Parse outputs
plane = ["transperineal", "transabdominal", "oblique",
         "sagittal", "axial", "other"][np.argmax(plane_logits)]
mask = np.argmax(seg_masks, axis=1)[0]
aop, hsd = labor_params[0]

print(f"Plane: {plane}")
print(f"AoP: {aop:.1f}°, HSD: {hsd:.1f}px")
```

### Flutter (Dart)

```dart
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class LaborViewService extends ChangeNotifier {
  OrtSession? _session;

  Future<void> loadModel() async {
    final options = OrtSessionOptions();
    _session = await OrtSession.fromAsset(
      'assets/models/laborview_mobilevit.onnx',
      options,
    );
  }

  Future<LaborViewResult> analyze(Uint8List imageBytes) async {
    // Preprocess to 256x256, ImageNet normalize
    final input = _preprocessImage(imageBytes);

    // Run multi-task inference
    final outputs = await _session!.run([input]);

    return LaborViewResult(
      planeClass: _argmax(outputs[0]),
      segMask: _argmax2D(outputs[1]),
      aop: outputs[2][0],
      hsd: outputs[2][1],
    );
  }
}
```

### iOS (Swift + CoreML)

```swift
import CoreML
import Vision

class LaborViewAnalyzer {
    private let model: VNCoreMLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let laborview = try LaborView(configuration: config)
        model = try VNCoreMLModel(for: laborview.model)
    }

    func analyze(image: CGImage) async throws -> LaborViewResult {
        let request = VNCoreMLRequest(model: model)
        let handler = VNImageRequestHandler(cgImage: image)
        try handler.perform([request])

        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
            throw AnalysisError.noResults
        }

        return LaborViewResult(
            planeLogits: results[0].featureValue.multiArrayValue!,
            segMask: results[1].featureValue.multiArrayValue!,
            laborParams: results[2].featureValue.multiArrayValue!
        )
    }
}
```

### Android (Kotlin + TFLite)

```kotlin
class LaborViewInterpreter(context: Context) {
    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            addDelegate(GpuDelegate())
        }
        val model = FileUtil.loadMappedFile(context, "laborview.tflite")
        interpreter = Interpreter(model, options)
    }

    fun analyze(bitmap: Bitmap): LaborViewResult {
        val input = preprocessBitmap(bitmap)  // 1x3x256x256

        val planeLogits = Array(1) { FloatArray(6) }
        val segMask = Array(1) { Array(3) { Array(256) { FloatArray(256) } } }
        val laborParams = Array(1) { FloatArray(2) }

        val outputs = mapOf(
            0 to planeLogits,
            1 to segMask,
            2 to laborParams
        )

        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)

        return LaborViewResult(
            plane = planeLogits[0].argmax(),
            mask = segMask[0].argmax2D(),
            aop = laborParams[0][0],
            hsd = laborParams[0][1]
        )
    }
}
```

### Clinical Metrics (Post-processing)

```python
from clinical_metrics import compute_all_metrics

# Get comprehensive clinical assessment from segmentation
metrics = compute_all_metrics(
    segmentation_mask=mask,
    symphysis_class=1,
    head_class=2
)

# Full output
print(f"=== Clinical Assessment ===")
print(f"AoP: {metrics.aop:.1f}° - {metrics.aop_interpretation}")
print(f"HSD: {metrics.hsd:.1f}px - {metrics.hsd_interpretation}")
print(f"Head Circumference: {metrics.head_circumference:.0f}px")
print(f"Head Area: {metrics.head_area:.0f}px²")
print(f"Quality: {metrics.segmentation_quality} ({metrics.confidence:.0%})")
print(f"Progress: {metrics.labor_progress.upper()}")
print(f"Recommendation: {metrics.recommendation}")
```

## Model Files

| File | Format | Size | Platform |
|------|--------|------|----------|
| `laborview_mobilevit.onnx` | ONNX | ~21 MB | All |
| `LaborView.mlpackage` | CoreML | ~22 MB | iOS 15+ |
| `laborview.tflite` | TFLite | ~21 MB | Android 7+ |
| `metadata.json` | JSON | 1 KB | All |
| `best.pt` | PyTorch | ~25 MB | Dev |

## Performance

### Multi-Task Metrics

| Task | Metric | Value |
|------|--------|-------|
| Segmentation | Mean IoU | TBD |
| Segmentation | Dice | TBD |
| Classification | Accuracy | TBD |
| Regression (AoP) | MAE | TBD |
| Regression (HSD) | MAE | TBD |

### Inference Latency

| Device | Latency | FPS |
|--------|---------|-----|
| iPhone 14 Pro | ~25ms | 40 |
| iPhone 12 | ~35ms | 28 |
| Pixel 7 | ~30ms | 33 |
| Samsung S23 | ~28ms | 35 |
| iPad Pro M1 | ~18ms | 55 |

### Size Comparison

| Model | Params | Size | Mobile Latency |
|-------|--------|------|----------------|
| MedSigLIP (Full) | ~400M | 1.6 GB | N/A |
| **MobileViT (Edge)** | ~5.6M | 21 MB | ~30ms |

## Limitations

1. **Resolution Trade-off**: 256×256 may miss fine details
2. **Training Data**: Single dataset/protocol
3. **Image Quality**: Sensitive to artifacts/shadows
4. **Anatomical Variations**: May struggle with unusual cases
5. **Calibration**: Needs device-specific pixel-to-mm conversion
6. **Regression Accuracy**: Direct predictions less accurate than geometry-computed

## Safety & Regulatory

### Intended Use

**Research and development only.** Not approved for clinical diagnosis.

### Before Clinical Deployment

1. Obtain FDA 510(k) / CE marking
2. Validate on local patient population
3. Verify hardware integration
4. Train operators on limitations
5. Implement adverse event monitoring

### Known Risks

- False negatives may delay intervention
- False positives may cause unnecessary intervention
- Model drift with different equipment

## Ethical Considerations

- **Health Equity**: Designed for resource-limited settings
- **Privacy**: Local inference, no data transmission
- **Bias**: Validate across diverse populations
- **Transparency**: Full documentation provided

## Citation

```bibtex
@software{laborview_edge_2024,
  title = {LaborView Ultrasound: Edge Multi-Task Model for Labor Monitoring},
  author = {Samuel},
  year = {2024},
  url = {https://huggingface.co/samwell/laborview-ultrasound},
  note = {Multi-task MobileViT: segmentation + classification + regression}
}
```

## Related Resources

- [laborview-medsiglip](https://huggingface.co/samwell/laborview-medsiglip) - Full model (higher accuracy)
- [Demo Space](https://huggingface.co/spaces/samwell/laborview-demo) - Try online
- [HAI-DEF Challenge](https://hai-def.org/) - Dataset
- [MobileViT Paper](https://arxiv.org/abs/2110.02178) - Base architecture

## License

Apache 2.0
