# LaborView AI - Flutter App

AI-powered intrapartum ultrasound analysis using MedSigLIP from Google's MedGemma family.

## Features

- **Real-time Analysis**: Capture or import ultrasound images for instant AI analysis
- **Segmentation Overlay**: Visual identification of pubic symphysis and fetal head
- **Clinical Metrics**:
  - Angle of Progression (AoP)
  - Head-Symphysis Distance (HSD)
  - Head Circumference (HC)
- **Labor Progress Assessment**: AI-powered evaluation with clinical recommendations

## Requirements

- Flutter SDK 3.16+
- iOS 14.0+ / Android API 24+
- ~1.7GB storage for model download

## Setup

### 1. Install Dependencies

```bash
cd laborview/flutter_app
flutter pub get
```

### 2. iOS Configuration

Add camera and photo library permissions to `ios/Runner/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>LaborView needs camera access to capture ultrasound images</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>LaborView needs photo library access to import ultrasound images</string>
```

### 3. Android Configuration

Add permissions to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
```

### 4. Run the App

```bash
# iOS Simulator
flutter run -d ios

# Android Emulator
flutter run -d android

# Physical device
flutter run
```

## Model

The app downloads the MedSigLIP ONNX model (~1.6GB) from HuggingFace on first launch:
- Repository: `samwell/laborview-medsiglip`
- Model: `laborview_medsiglip.onnx`
- Input: 448x448 RGB images
- Output: 3-class segmentation + plane classification

## Architecture

```
lib/
├── main.dart                 # App entry point
├── models/
│   └── analysis_result.dart  # Data model with clinical interpretations
├── services/
│   └── laborview_service.dart # ONNX inference & metrics computation
├── screens/
│   ├── home_screen.dart      # Main UI with capture options
│   └── analysis_screen.dart  # Results display
└── widgets/
    ├── segmentation_overlay.dart  # Visual overlay on image
    ├── metrics_card.dart          # Individual metric display
    └── progress_indicator_card.dart # Labor progress status
```

## Clinical Metrics

### Angle of Progression (AoP)
- Measures fetal head descent relative to pubic symphysis
- <110°: Early labor, head not engaged
- 120-130°: Active labor with good progress
- >140°: Late labor, delivery imminent

### Head-Symphysis Distance (HSD)
- Distance from symphysis to nearest point of fetal head
- Indicates station and engagement

## Disclaimer

**Research prototype only. Not for clinical diagnosis.**

This app is a demonstration for the MedGemma Impact Challenge. All clinical decisions should be made by qualified healthcare professionals.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Google MedGemma team for MedSigLIP
- MedGemma Impact Challenge
