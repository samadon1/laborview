import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import '../models/analysis_result.dart';

/// CoreML-based inference service for iOS
/// Uses Neural Engine for optimized on-device inference
class CoreMLService {
  static const MethodChannel _channel = MethodChannel('com.laborview/coreml');

  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// Check if CoreML is available (iOS only)
  static Future<bool> isAvailable() async {
    if (!Platform.isIOS) return false;

    try {
      final result = await _channel.invokeMethod<bool>('isAvailable');
      return result ?? false;
    } catch (e) {
      debugPrint('CoreML availability check failed: $e');
      return false;
    }
  }

  /// Initialize and load the CoreML model
  Future<bool> initialize() async {
    if (_isInitialized) return true;

    try {
      final success = await _channel.invokeMethod<bool>('loadModel');
      _isInitialized = success ?? false;
      return _isInitialized;
    } catch (e) {
      debugPrint('CoreML initialization failed: $e');
      return false;
    }
  }

  /// Analyze an image from file path
  Future<AnalysisResult?> analyzeFromPath(String imagePath) async {
    if (!_isInitialized) {
      final success = await initialize();
      if (!success) return null;
    }

    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'analyzeFromPath',
        {'imagePath': imagePath},
      );

      if (result == null) return null;

      if (result.containsKey('error')) {
        debugPrint('CoreML analysis error: ${result['error']}');
        return null;
      }

      return _parseResult(result);
    } catch (e) {
      debugPrint('CoreML analysis failed: $e');
      return null;
    }
  }

  /// Analyze an image from bytes
  Future<AnalysisResult?> analyzeBytes(Uint8List imageData) async {
    if (!_isInitialized) {
      final success = await initialize();
      if (!success) return null;
    }

    try {
      final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
        'analyze',
        {'imageData': imageData},
      );

      if (result == null) return null;

      if (result.containsKey('error')) {
        debugPrint('CoreML analysis error: ${result['error']}');
        return null;
      }

      return _parseResult(result);
    } catch (e) {
      debugPrint('CoreML analysis failed: $e');
      return null;
    }
  }

  /// Analyze a File
  Future<AnalysisResult?> analyze(File imageFile) async {
    return analyzeFromPath(imageFile.path);
  }

  /// Parse the result from native code
  AnalysisResult _parseResult(Map<dynamic, dynamic> result) {
    // Parse segmentation mask
    List<List<int>> segMask = [];
    if (result['segmentation'] != null) {
      final rawSeg = result['segmentation'] as List<dynamic>;
      segMask = rawSeg.map((row) {
        return (row as List<dynamic>).map((v) => v as int).toList();
      }).toList();
    }

    // Parse metrics
    final aop = result['aop'] as double?;
    final hsd = result['hsd'] as double?;
    final hc = result['headCircumference'] as double?;
    final planeClass = result['planeClass'] as String? ?? 'Unknown';

    return AnalysisResult(
      segmentationMask: segMask,
      planeClass: planeClass,
      planeConfidence: 0.95,
      aop: aop,
      hsd: hsd,
      headCircumference: hc,
      headArea: null,
    );
  }
}
