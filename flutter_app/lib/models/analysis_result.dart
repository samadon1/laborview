class AnalysisResult {
  final List<List<int>> segmentationMask;
  final String planeClass;
  final double planeConfidence;
  final double? aop;
  final double? hsd;
  final double? headCircumference;
  final double? headArea;
  final DateTime timestamp;

  AnalysisResult({
    required this.segmentationMask,
    required this.planeClass,
    required this.planeConfidence,
    this.aop,
    this.hsd,
    this.headCircumference,
    this.headArea,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// AoP interpretation based on clinical thresholds
  String get aopInterpretation {
    if (aop == null) return 'Unable to measure';
    if (aop! < 100) return 'Very early labor - high fetal station';
    if (aop! < 110) return 'Early labor - fetal head not engaged';
    if (aop! < 120) return 'Active labor - head descending';
    if (aop! < 130) return 'Good progress - head well descended';
    if (aop! < 140) return 'Advanced labor - delivery approaching';
    return 'Late labor - delivery imminent';
  }

  /// HSD interpretation
  String get hsdInterpretation {
    if (hsd == null) return 'Unable to measure';
    // Normalized to typical image size
    final ratio = hsd! / 448;
    if (ratio > 0.3) return 'Large distance - head not engaged';
    if (ratio > 0.2) return 'Moderate distance - early descent';
    if (ratio > 0.1) return 'Small distance - good descent';
    return 'Minimal distance - head at symphysis level';
  }

  /// Overall labor progress assessment
  LaborProgress get laborProgress {
    if (aop == null) return LaborProgress.unknown;

    int score = 0;

    if (aop! >= 120) {
      score += 2;
    } else if (aop! >= 110) {
      score += 1;
    } else {
      score -= 1;
    }

    if (hsd != null) {
      final ratio = hsd! / 448;
      if (ratio < 0.15) {
        score += 1;
      } else if (ratio > 0.25) {
        score -= 1;
      }
    }

    if (score >= 2) return LaborProgress.normal;
    if (score >= 0) return LaborProgress.monitor;
    return LaborProgress.concern;
  }

  /// Clinical recommendation based on assessment
  String get recommendation {
    switch (laborProgress) {
      case LaborProgress.normal:
        return 'Labor progressing well. Continue routine monitoring.';
      case LaborProgress.monitor:
        return 'Continue close monitoring and reassess in 30-60 minutes.';
      case LaborProgress.concern:
        return 'Slow progress noted. Consider clinical examination and evaluate need for intervention.';
      case LaborProgress.unknown:
        return 'Unable to assess - ensure proper ultrasound view and retry.';
    }
  }

  /// Convert to map for storage/export
  Map<String, dynamic> toJson() => {
        'planeClass': planeClass,
        'planeConfidence': planeConfidence,
        'aop': aop,
        'aopInterpretation': aopInterpretation,
        'hsd': hsd,
        'hsdInterpretation': hsdInterpretation,
        'headCircumference': headCircumference,
        'headArea': headArea,
        'laborProgress': laborProgress.name,
        'recommendation': recommendation,
        'timestamp': timestamp.toIso8601String(),
      };
}

enum LaborProgress {
  normal,
  monitor,
  concern,
  unknown;

  String get displayName {
    switch (this) {
      case LaborProgress.normal:
        return 'NORMAL';
      case LaborProgress.monitor:
        return 'MONITOR';
      case LaborProgress.concern:
        return 'CONCERN';
      case LaborProgress.unknown:
        return 'UNKNOWN';
    }
  }

  String get color {
    switch (this) {
      case LaborProgress.normal:
        return '#4CAF50';
      case LaborProgress.monitor:
        return '#FF9800';
      case LaborProgress.concern:
        return '#F44336';
      case LaborProgress.unknown:
        return '#9E9E9E';
    }
  }
}
