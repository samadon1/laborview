import 'package:flutter/material.dart';

class SegmentationOverlay extends StatelessWidget {
  final List<List<int>> mask;
  final double opacity;

  const SegmentationOverlay({
    super.key,
    required this.mask,
    this.opacity = 0.5,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _SegmentationPainter(mask: mask, opacity: opacity),
      size: Size.infinite,
    );
  }
}

class _SegmentationPainter extends CustomPainter {
  final List<List<int>> mask;
  final double opacity;

  _SegmentationPainter({required this.mask, required this.opacity});

  // Medical imaging colors
  // NOTE: Model outputs are swapped due to training data processing
  // Class 1 = actually fetal head (large structure)
  // Class 2 = actually pubic symphysis (small structure)
  static const Map<int, Color> classColors = {
    0: Colors.transparent,
    1: Color(0xFFFF0080), // Magenta - Fetal head (model outputs as class 1)
    2: Color(0xFF00D4FF), // Cyan - Pubic symphysis (model outputs as class 2)
  };

  @override
  void paint(Canvas canvas, Size size) {
    if (mask.isEmpty) return;

    final maskHeight = mask.length;
    final maskWidth = mask[0].length;

    final scaleX = size.width / maskWidth;
    final scaleY = size.height / maskHeight;

    // Draw filled regions with low opacity
    for (int y = 0; y < maskHeight; y++) {
      for (int x = 0; x < maskWidth; x++) {
        final classId = mask[y][x];
        if (classId == 0) continue;

        final color = classColors[classId] ?? Colors.grey;
        final paint = Paint()
          ..color = color.withOpacity(opacity * 0.4)
          ..style = PaintingStyle.fill;

        canvas.drawRect(
          Rect.fromLTWH(
            x * scaleX,
            y * scaleY,
            scaleX + 0.5,
            scaleY + 0.5,
          ),
          paint,
        );
      }
    }

    // Draw contours with higher opacity
    _drawContours(canvas, maskWidth, maskHeight, scaleX, scaleY);
  }

  void _drawContours(
    Canvas canvas,
    int maskWidth,
    int maskHeight,
    double scaleX,
    double scaleY,
  ) {
    for (int classId = 1; classId <= 2; classId++) {
      final color = classColors[classId]!;
      final contourPaint = Paint()
        ..color = color.withOpacity(0.9)
        ..style = PaintingStyle.fill;

      for (int y = 1; y < maskHeight - 1; y++) {
        for (int x = 1; x < maskWidth - 1; x++) {
          if (mask[y][x] == classId) {
            // Check if boundary pixel
            if (mask[y - 1][x] != classId ||
                mask[y + 1][x] != classId ||
                mask[y][x - 1] != classId ||
                mask[y][x + 1] != classId) {
              canvas.drawCircle(
                Offset(x * scaleX, y * scaleY),
                1.2,
                contourPaint,
              );
            }
          }
        }
      }
    }
  }

  @override
  bool shouldRepaint(_SegmentationPainter oldDelegate) {
    return mask != oldDelegate.mask || opacity != oldDelegate.opacity;
  }
}
