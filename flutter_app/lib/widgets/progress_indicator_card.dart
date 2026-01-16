import 'package:flutter/material.dart';
import '../models/analysis_result.dart';

class ProgressIndicatorCard extends StatelessWidget {
  final AnalysisResult result;

  const ProgressIndicatorCard({
    super.key,
    required this.result,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final progress = result.laborProgress;

    return Card(
      color: _getBackgroundColor(progress, theme),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              children: [
                _buildStatusIcon(progress),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Labor Progress',
                        style: theme.textTheme.titleSmall?.copyWith(
                          color: _getTextColor(progress),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        progress.displayName,
                        style: theme.textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: _getTextColor(progress),
                        ),
                      ),
                    ],
                  ),
                ),
                _buildProgressGauge(progress),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.9),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 20,
                    color: _getAccentColor(progress),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      result.recommendation,
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: Colors.black87,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusIcon(LaborProgress progress) {
    IconData icon;
    switch (progress) {
      case LaborProgress.normal:
        icon = Icons.check_circle;
        break;
      case LaborProgress.monitor:
        icon = Icons.schedule;
        break;
      case LaborProgress.concern:
        icon = Icons.warning;
        break;
      case LaborProgress.unknown:
        icon = Icons.help_outline;
        break;
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.3),
        shape: BoxShape.circle,
      ),
      child: Icon(
        icon,
        size: 32,
        color: _getTextColor(progress),
      ),
    );
  }

  Widget _buildProgressGauge(LaborProgress progress) {
    double value;
    switch (progress) {
      case LaborProgress.normal:
        value = 0.85;
        break;
      case LaborProgress.monitor:
        value = 0.5;
        break;
      case LaborProgress.concern:
        value = 0.25;
        break;
      case LaborProgress.unknown:
        value = 0.0;
        break;
    }

    return SizedBox(
      width: 60,
      height: 60,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: value,
            strokeWidth: 6,
            backgroundColor: Colors.white.withOpacity(0.3),
            valueColor: AlwaysStoppedAnimation<Color>(
              _getTextColor(progress),
            ),
          ),
          Text(
            '${(value * 100).toInt()}%',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: _getTextColor(progress),
            ),
          ),
        ],
      ),
    );
  }

  Color _getBackgroundColor(LaborProgress progress, ThemeData theme) {
    switch (progress) {
      case LaborProgress.normal:
        return const Color(0xFF4CAF50);
      case LaborProgress.monitor:
        return const Color(0xFFFF9800);
      case LaborProgress.concern:
        return const Color(0xFFF44336);
      case LaborProgress.unknown:
        return const Color(0xFF9E9E9E);
    }
  }

  Color _getTextColor(LaborProgress progress) {
    return Colors.white;
  }

  Color _getAccentColor(LaborProgress progress) {
    switch (progress) {
      case LaborProgress.normal:
        return const Color(0xFF2E7D32);
      case LaborProgress.monitor:
        return const Color(0xFFE65100);
      case LaborProgress.concern:
        return const Color(0xFFC62828);
      case LaborProgress.unknown:
        return const Color(0xFF616161);
    }
  }
}
