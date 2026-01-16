import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import '../models/analysis_result.dart';
import 'coreml_service.dart';

/// Model type enum for switching between models
enum ModelType {
  mobileViT,  // Default - small, fast, ~1.6MB
  medSigLIP,  // Large, accurate, ~1.6GB (disabled by default for RAM)
}

class LaborViewService extends ChangeNotifier {
  OnnxRuntime? _ort;
  OrtSession? _session;
  CoreMLService? _coreMLService;
  bool _useCoreML = false;
  bool _isInitialized = false;
  bool _isLoading = false;
  double _downloadProgress = 0.0;
  String _statusMessage = 'Ready';
  AnalysisResult? _lastResult;

  // Current model type - MobileViT is default (small, fast)
  ModelType _currentModel = ModelType.mobileViT;

  // ============================================================
  // MobileViT Model Configuration (DEFAULT - recommended)
  // Small (~1.6MB), fast inference, good accuracy (IoU: 0.8424)
  // ============================================================
  static const int _mobileViTInputSize = 256;
  static const String _mobileViTModelUrl =
      'https://huggingface.co/samwell/laborview-ultrasound/resolve/main/laborview_mobilevit.onnx';

  // ImageNet normalization for MobileViT
  static const List<double> _mobileViTMean = [0.485, 0.456, 0.406];
  static const List<double> _mobileViTStd = [0.229, 0.224, 0.225];

  // ============================================================
  // MedSigLIP Model Configuration (DISABLED - too large for mobile)
  // Large (~1.6GB), slower, highest accuracy (IoU: 0.8999)
  // Kept for reference but not loaded to avoid RAM issues
  // ============================================================
  // static const int _medSigLIPInputSize = 448;
  // static const String _medSigLIPModelUrl =
  //     'https://huggingface.co/samwell/laborview-medsiglip/resolve/main/laborview_medsiglip.onnx';
  // static const String _medSigLIPModelDataUrl =
  //     'https://huggingface.co/samwell/laborview-medsiglip/resolve/main/laborview_medsiglip.onnx.data';
  // // SigLIP normalization: (x - 0.5) / 0.5 -> [-1, 1]

  // Dynamic getters based on current model
  int get inputSize => _currentModel == ModelType.mobileViT
      ? _mobileViTInputSize
      : 448; // MedSigLIP size

  bool get isInitialized => _isInitialized;
  bool get isLoading => _isLoading;
  bool get useCoreML => _useCoreML;
  double get downloadProgress => _downloadProgress;
  String get statusMessage => _statusMessage;
  AnalysisResult? get lastResult => _lastResult;
  ModelType get currentModel => _currentModel;

  /// Initialize the inference engine
  Future<void> initialize() async {
    if (_isInitialized || _isLoading) return;

    _isLoading = true;
    _statusMessage = 'Initializing...';
    notifyListeners();

    try {
      await _initializeONNX();
    } catch (e) {
      _statusMessage = 'Error: ${e.toString()}';
      debugPrint('LaborView init error: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Initialize ONNX Runtime with MobileViT model
  Future<void> _initializeONNX() async {
    _ort = OnnxRuntime();

    // Get model file (downloads if needed)
    final modelFile = await _getModelFile();

    _statusMessage = 'Loading model...';
    notifyListeners();

    // Create session
    _session = await _ort!.createSession(modelFile.path);

    _useCoreML = false;
    _isInitialized = true;
    _statusMessage = 'Ready (MobileViT)';
  }

  /// Get model file, downloading if necessary
  Future<File> _getModelFile() async {
    final dir = await getApplicationDocumentsDirectory();

    // Clean up old MedSigLIP model files to free space
    final oldModelPath = '${dir.path}/laborview_medsiglip.onnx';
    final oldDataPath = '${dir.path}/laborview_medsiglip.onnx.data';
    final oldModel = File(oldModelPath);
    final oldData = File(oldDataPath);
    if (await oldModel.exists()) {
      await oldModel.delete();
      debugPrint('Deleted old MedSigLIP model');
    }
    if (await oldData.exists()) {
      await oldData.delete();
      debugPrint('Deleted old MedSigLIP data file');
    }

    // Use MobileViT model (small, fast) - v2 to force re-download
    final modelPath = '${dir.path}/laborview_mobilevit_v2.onnx';
    final modelFile = File(modelPath);

    // Also clean up old v1 model if exists
    final oldV1Path = '${dir.path}/laborview_mobilevit.onnx';
    final oldV1DataPath = '${dir.path}/laborview_mobilevit.onnx.data';
    final oldV1 = File(oldV1Path);
    final oldV1Data = File(oldV1DataPath);
    if (await oldV1.exists()) await oldV1.delete();
    if (await oldV1Data.exists()) await oldV1Data.delete();

    if (await modelFile.exists()) {
      final fileSize = await modelFile.length();
      // MobileViT model should be ~21MB
      if (fileSize > 20000000) {
        _statusMessage = 'Model found locally';
        return modelFile;
      }
    }

    // Download MobileViT model (~21MB - fast!)
    _statusMessage = 'Downloading model...';
    notifyListeners();

    await _downloadFile(_mobileViTModelUrl, modelFile, 'MobileViT model');

    return modelFile;
  }

  /// Download a file from URL with progress tracking
  Future<void> _downloadFile(String url, File destination, String label) async {
    final client = http.Client();

    try {
      final request = http.Request('GET', Uri.parse(url));
      final response = await client.send(request);

      final totalBytes = response.contentLength ?? 0;
      var receivedBytes = 0;

      final sink = destination.openWrite();

      await for (final chunk in response.stream) {
        sink.add(chunk);
        receivedBytes += chunk.length;

        if (totalBytes > 0) {
          _downloadProgress = receivedBytes / totalBytes;
          _statusMessage =
              'Downloading $label: ${(_downloadProgress * 100).toStringAsFixed(1)}%';
          notifyListeners();
        }
      }

      await sink.close();
      _downloadProgress = 1.0;
    } finally {
      client.close();
    }
  }

  /// Analyze an image file
  Future<AnalysisResult?> analyze(File imageFile) async {
    if (!_isInitialized) {
      await initialize();
      if (!_isInitialized) return null;
    }

    _isLoading = true;
    _statusMessage = 'Analyzing...';
    // Defer notification to avoid calling during build
    Future.microtask(() => notifyListeners());

    try {
      AnalysisResult? result;

      // Use CoreML on iOS if available
      if (_useCoreML && _coreMLService != null) {
        result = await _coreMLService!.analyze(imageFile);
      } else {
        // Use ONNX Runtime
        result = await _analyzeWithONNX(imageFile);
      }

      _lastResult = result;
      _statusMessage = 'Analysis complete';
      return result;
    } catch (e) {
      _statusMessage = 'Error: ${e.toString()}';
      debugPrint('Analysis error: $e');
      return null;
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Analyze using ONNX Runtime
  Future<AnalysisResult?> _analyzeWithONNX(File imageFile) async {
    if (_session == null) return null;

    // Load and preprocess image
    final bytes = await imageFile.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw Exception('Failed to decode image');
    }

    // Create input tensor
    final inputTensor = await _preprocessImage(image);

    // Run inference
    final inputs = {'pixel_values': inputTensor};
    final outputs = await _session!.run(inputs);

    // Parse outputs
    final result = await _parseOutputs(outputs);

    // Dispose input tensor
    await inputTensor.dispose();

    return result;
  }

  /// Analyze from raw bytes (for camera)
  Future<AnalysisResult?> analyzeBytes(Uint8List bytes) async {
    if (!_isInitialized) {
      await initialize();
      if (!_isInitialized) return null;
    }

    // Use CoreML directly with bytes if available (faster, no temp file)
    if (_useCoreML && _coreMLService != null) {
      _isLoading = true;
      _statusMessage = 'Analyzing...';
      notifyListeners();

      try {
        final result = await _coreMLService!.analyzeBytes(bytes);
        _lastResult = result;
        _statusMessage = 'Analysis complete';
        return result;
      } catch (e) {
        _statusMessage = 'Error: ${e.toString()}';
        debugPrint('CoreML analysis error: $e');
        return null;
      } finally {
        _isLoading = false;
        notifyListeners();
      }
    }

    // Fall back to file-based analysis for ONNX
    final tempDir = await getTemporaryDirectory();
    final tempFile = File('${tempDir.path}/temp_image.jpg');
    await tempFile.writeAsBytes(bytes);
    return analyze(tempFile);
  }

  /// Preprocess image for model input
  Future<OrtValue> _preprocessImage(img.Image image) async {
    final size = inputSize;

    // Resize to model input size
    final resized = img.copyResize(
      image,
      width: size,
      height: size,
      interpolation: img.Interpolation.linear,
    );

    // Create float tensor [1, 3, H, W] in CHW format
    final data = Float32List(1 * 3 * size * size);

    int idx = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
          final pixel = resized.getPixel(x, y);
          double value;
          switch (c) {
            case 0:
              value = pixel.r / 255.0;
              break;
            case 1:
              value = pixel.g / 255.0;
              break;
            default:
              value = pixel.b / 255.0;
          }

          // Apply normalization based on model type
          if (_currentModel == ModelType.mobileViT) {
            // ImageNet normalization for MobileViT
            data[idx++] = (value - _mobileViTMean[c]) / _mobileViTStd[c];
          } else {
            // SigLIP normalization: [-1, 1]
            data[idx++] = (value - 0.5) / 0.5;
          }
        }
      }
    }

    return OrtValue.fromList(data, [1, 3, size, size]);
  }

  /// Parse model outputs into AnalysisResult
  Future<AnalysisResult> _parseOutputs(Map<String, OrtValue> outputs) async {
    // seg_probs: [1, 3, H, W]
    // plane_pred: [1]
    final segProbsValue = outputs['seg_probs'];
    final planePredValue = outputs['plane_pred'];

    // Get segmentation mask as flattened list
    final segProbs = await segProbsValue!.asFlattenedList();
    final segMask = _processSegmentation(segProbs.cast<double>());

    // Get plane prediction
    final planePred = await planePredValue!.asFlattenedList();
    final planeClass = planePred[0] == 0 ? 'Transperineal' : 'Other';

    // Dispose output tensors
    await segProbsValue.dispose();
    await planePredValue.dispose();

    // Compute clinical metrics
    final metrics = _computeMetrics(segMask);

    return AnalysisResult(
      segmentationMask: segMask,
      planeClass: planeClass,
      planeConfidence: 0.95, // TODO: Get from softmax
      aop: metrics['aop'],
      hsd: metrics['hsd'],
      headCircumference: metrics['hc'],
      headArea: metrics['area'],
    );
  }

  /// Convert segmentation probabilities to class mask
  List<List<int>> _processSegmentation(List<double> probs) {
    final size = inputSize;
    final mask = List.generate(
      size,
      (_) => List.filled(size, 0),
    );

    final pixelCount = size * size;

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final idx = y * size + x;
        int maxClass = 0;
        double maxProb = probs[idx]; // Class 0

        for (int c = 1; c < 3; c++) {
          final prob = probs[c * pixelCount + idx];
          if (prob > maxProb) {
            maxProb = prob;
            maxClass = c;
          }
        }
        mask[y][x] = maxClass;
      }
    }

    return mask;
  }

  /// Compute clinical metrics from segmentation mask
  Map<String, double?> _computeMetrics(List<List<int>> mask) {
    // Count pixels per class for debugging
    int class0 = 0, class1 = 0, class2 = 0;
    for (int y = 0; y < mask.length; y++) {
      for (int x = 0; x < mask[0].length; x++) {
        switch (mask[y][x]) {
          case 0: class0++; break;
          case 1: class1++; break;
          case 2: class2++; break;
        }
      }
    }
    debugPrint('Segmentation class distribution:');
    debugPrint('  Class 0 (background): $class0 pixels');
    debugPrint('  Class 1 (fetal head): $class1 pixels');
    debugPrint('  Class 2 (symphysis): $class2 pixels');

    // NOTE: Model class outputs are swapped due to training data processing
    // Class 1 = actually fetal head, Class 2 = actually pubic symphysis
    final headPoints = _findContour(mask, 1);      // Head is class 1
    final symphysisPoints = _findContour(mask, 2); // Symphysis is class 2

    debugPrint('Contour points found:');
    debugPrint('  Head contour: ${headPoints.length} points');
    debugPrint('  Symphysis contour: ${symphysisPoints.length} points');

    if (symphysisPoints.length < 10 || headPoints.length < 10) {
      debugPrint('WARNING: Not enough contour points for metrics calculation');
      return {'aop': null, 'hsd': null, 'hc': null, 'area': null};
    }

    return {
      'aop': _computeAoP(symphysisPoints, headPoints),
      'hsd': _computeHSD(symphysisPoints, headPoints),
      'hc': _computeCircumference(headPoints),
      'area': _computeArea(headPoints),
    };
  }

  /// Find contour points for a class
  List<Point> _findContour(List<List<int>> mask, int classId) {
    final points = <Point>[];
    final h = mask.length;
    final w = mask[0].length;

    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        if (mask[y][x] == classId) {
          // Check if boundary pixel
          if (mask[y - 1][x] != classId ||
              mask[y + 1][x] != classId ||
              mask[y][x - 1] != classId ||
              mask[y][x + 1] != classId) {
            points.add(Point(x.toDouble(), y.toDouble()));
          }
        }
      }
    }

    return points;
  }

  /// Compute Angle of Progression
  double? _computeAoP(List<Point> symphysis, List<Point> head) {
    if (symphysis.length < 10 || head.length < 10) return null;

    // Find lowest points
    final symphysisLowest = symphysis.reduce(
      (a, b) => a.y > b.y ? a : b,
    );
    final headLowest = head.reduce(
      (a, b) => a.y > b.y ? a : b,
    );

    // Fit line to symphysis (simple direction using covariance)
    final meanX =
        symphysis.map((p) => p.x).reduce((a, b) => a + b) / symphysis.length;
    final meanY =
        symphysis.map((p) => p.y).reduce((a, b) => a + b) / symphysis.length;

    double covXX = 0, covXY = 0;
    for (final p in symphysis) {
      covXX += (p.x - meanX) * (p.x - meanX);
      covXY += (p.x - meanX) * (p.y - meanY);
    }

    final slope = covXY / (covXX + 1e-6);
    final lineDir = Point(1, slope);

    // Head vector from symphysis to head
    final headVector = Point(
      headLowest.x - symphysisLowest.x,
      headLowest.y - symphysisLowest.y,
    );

    // Calculate angle using dot product
    final dot = lineDir.x * headVector.x + lineDir.y * headVector.y;
    final mag1 = sqrt(lineDir.x * lineDir.x + lineDir.y * lineDir.y);
    final mag2 = sqrt(headVector.x * headVector.x + headVector.y * headVector.y);

    var angle = acos(dot.abs() / (mag1 * mag2)) * 180 / pi;

    // Adjust for head position
    if (headLowest.y > symphysisLowest.y) {
      angle = 90 + (90 - angle);
    }

    return angle;
  }

  /// Compute Head-Symphysis Distance
  double? _computeHSD(List<Point> symphysis, List<Point> head) {
    if (symphysis.isEmpty || head.isEmpty) return null;

    final symphysisLowest = symphysis.reduce(
      (a, b) => a.y > b.y ? a : b,
    );

    double minDist = double.infinity;
    for (final p in head) {
      final dist = sqrt(
        pow(p.x - symphysisLowest.x, 2) + pow(p.y - symphysisLowest.y, 2),
      );
      if (dist < minDist) minDist = dist;
    }

    return minDist.isFinite ? minDist : null;
  }

  /// Compute contour circumference
  double? _computeCircumference(List<Point> contour) {
    if (contour.length < 10) return null;

    // Sort points by angle from centroid for proper perimeter calculation
    final cx = contour.map((p) => p.x).reduce((a, b) => a + b) / contour.length;
    final cy = contour.map((p) => p.y).reduce((a, b) => a + b) / contour.length;

    final sorted = List<Point>.from(contour)
      ..sort((a, b) {
        final angleA = atan2(a.y - cy, a.x - cx);
        final angleB = atan2(b.y - cy, b.x - cx);
        return angleA.compareTo(angleB);
      });

    double perimeter = 0;
    for (int i = 0; i < sorted.length; i++) {
      final next = sorted[(i + 1) % sorted.length];
      perimeter += sqrt(
        pow(sorted[i].x - next.x, 2) + pow(sorted[i].y - next.y, 2),
      );
    }

    return perimeter;
  }

  /// Compute area using shoelace formula
  double? _computeArea(List<Point> contour) {
    if (contour.length < 10) return null;

    // Sort points
    final cx = contour.map((p) => p.x).reduce((a, b) => a + b) / contour.length;
    final cy = contour.map((p) => p.y).reduce((a, b) => a + b) / contour.length;

    final sorted = List<Point>.from(contour)
      ..sort((a, b) {
        final angleA = atan2(a.y - cy, a.x - cx);
        final angleB = atan2(b.y - cy, b.x - cx);
        return angleA.compareTo(angleB);
      });

    double area = 0;
    for (int i = 0; i < sorted.length; i++) {
      final j = (i + 1) % sorted.length;
      area += sorted[i].x * sorted[j].y;
      area -= sorted[j].x * sorted[i].y;
    }

    return area.abs() / 2;
  }

  /// Dispose resources
  @override
  void dispose() {
    _session?.close();
    super.dispose();
  }
}

/// Simple 2D point class
class Point {
  final double x, y;
  const Point(this.x, this.y);
}
