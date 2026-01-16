import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:cactus/cactus.dart';

/// Service for clinical AI interpretation using Cactus on-device
/// Uses vision-capable model for ultrasound analysis
class MedGemmaService extends ChangeNotifier {
  CactusLM? _lm;
  bool _isInitialized = false;
  bool _isLoading = false;
  double _downloadProgress = 0.0;
  String _statusMessage = 'Not initialized';
  String? _errorMessage;

  // Using vision-capable model for image analysis
  static const String _modelSlug = 'lfm2-vl-450m';

  bool get isInitialized => _isInitialized;
  bool get isLoading => _isLoading;
  double get downloadProgress => _downloadProgress;
  String get statusMessage => _statusMessage;
  String? get errorMessage => _errorMessage;

  /// Initialize Cactus with vision model
  Future<void> initialize() async {
    if (_isInitialized || _isLoading) return;

    _isLoading = true;
    _errorMessage = null;
    _statusMessage = 'Initializing...';
    notifyListeners();

    try {
      _lm = CactusLM();

      _statusMessage = 'Downloading model...';
      notifyListeners();

      // Download vision model
      await _lm!.downloadModel(
        model: _modelSlug,
        downloadProcessCallback: (progress, status, isError) {
          if (isError) {
            _errorMessage = status;
            debugPrint('Download error: $status');
          } else {
            _downloadProgress = progress ?? 0.0;
            _statusMessage = status;
            notifyListeners();
          }
        },
      );

      _statusMessage = 'Loading model...';
      notifyListeners();

      // Initialize the model
      await _lm!.initializeModel(
        params: CactusInitParams(
          model: _modelSlug,
          contextSize: 2048,
        ),
      );

      _isInitialized = true;
      _statusMessage = 'AI Ready';
      debugPrint('Cactus vision model initialized successfully');
    } catch (e) {
      _errorMessage = e.toString();
      _statusMessage = 'Failed to initialize';
      debugPrint('Cactus init error: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Analyze ultrasound image and measurements
  Future<String?> analyzeUltrasound(File imageFile, {
    double? aop,
    double? hsd,
    String? planeClass,
  }) async {
    if (!_isInitialized || _lm == null) {
      debugPrint('Cactus not initialized');
      return null;
    }

    try {
      // Build clinical context from segmentation measurements
      final measurements = StringBuffer();
      if (aop != null) {
        measurements.writeln('- Angle of Progression (AoP): ${aop.toStringAsFixed(1)}°');
      }
      if (hsd != null) {
        measurements.writeln('- Head-Symphysis Distance (HSD): ${hsd.toStringAsFixed(1)} pixels');
      }
      if (planeClass != null) {
        measurements.writeln('- Ultrasound plane: $planeClass');
      }

      final prompt = '''You are an expert obstetrician analyzing transperineal ultrasound measurements during labor.

MEASUREMENTS FROM AUTOMATED SEGMENTATION:
${measurements.toString().isNotEmpty ? measurements.toString() : 'No clear measurements - segmentation could not identify structures.'}

CLINICAL CONTEXT:
- AoP (Angle of Progression) >120° suggests good labor progress
- AoP <110° may indicate slow progress requiring attention
- HSD measures proximity of fetal head to pubic symphysis

Provide a brief (3-4 sentences) clinical interpretation including:
1. Assessment of the measurements
2. Labor progress status (Normal/Monitor/Concern)
3. Recommendation''';

      final result = await _lm!.generateCompletion(
        messages: [
          ChatMessage(
            content: 'You are a helpful medical AI assistant specialized in obstetric ultrasound analysis.',
            role: 'system',
          ),
          ChatMessage(
            content: prompt,
            role: 'user',
            images: [imageFile.path],
          ),
        ],
        params: CactusCompletionParams(
          maxTokens: 300,
        ),
      );

      if (result.success) {
        return result.response.trim();
      } else {
        debugPrint('Cactus completion failed');
        return null;
      }
    } catch (e) {
      debugPrint('Cactus analysis error: $e');
      return 'Error: $e';
    }
  }

  /// Generate a clinical report
  Future<String?> generateReport({
    required File imageFile,
    double? aop,
    double? hsd,
    double? headCircumference,
    double? headArea,
    String? planeClass,
    String? laborProgress,
  }) async {
    if (!_isInitialized || _lm == null) return null;

    try {
      final measurements = StringBuffer();
      if (aop != null) measurements.writeln('- Angle of Progression: ${aop.toStringAsFixed(1)}°');
      if (hsd != null) measurements.writeln('- Head-Symphysis Distance: ${hsd.toStringAsFixed(1)} px');
      if (headCircumference != null) measurements.writeln('- Head Circumference: ${headCircumference.toStringAsFixed(0)} px');
      if (headArea != null) measurements.writeln('- Head Area: ${headArea.toStringAsFixed(0)} px²');
      if (laborProgress != null) measurements.writeln('- Automated Assessment: $laborProgress');

      final prompt = '''Generate a brief clinical report for transperineal ultrasound examination.

MEASUREMENTS:
${measurements.toString()}

Format as:
IMPRESSION: [1-2 sentences]
RECOMMENDATION: [Action if needed]''';

      final result = await _lm!.generateCompletion(
        messages: [
          ChatMessage(
            content: 'You are a medical report generator for obstetric ultrasound.',
            role: 'system',
          ),
          ChatMessage(
            content: prompt,
            role: 'user',
            images: [imageFile.path],
          ),
        ],
        params: CactusCompletionParams(
          maxTokens: 200,
        ),
      );

      if (result.success) {
        return result.response.trim();
      }
      return null;
    } catch (e) {
      debugPrint('Cactus report error: $e');
      return null;
    }
  }

  /// Unload model and free resources
  Future<void> unload() async {
    if (_lm != null) {
      _lm!.unload();
      _lm = null;
      _isInitialized = false;
      _statusMessage = 'Not initialized';
      notifyListeners();
    }
  }

  @override
  void dispose() {
    unload();
    super.dispose();
  }
}
