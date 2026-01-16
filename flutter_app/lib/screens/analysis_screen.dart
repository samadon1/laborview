import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/analysis_result.dart';
import '../services/laborview_service.dart';
import '../services/medgemma_service.dart';
import '../widgets/segmentation_overlay.dart';

class AnalysisScreen extends StatefulWidget {
  final File imageFile;

  const AnalysisScreen({super.key, required this.imageFile});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  AnalysisResult? _result;
  bool _isAnalyzing = true;
  String? _errorMessage;
  bool _showOverlay = true;

  // MedGemma AI interpretation
  String? _aiInterpretation;
  bool _isLoadingAI = false;
  bool _aiAvailable = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runAnalysis();
      _checkAIAvailability();
    });
  }

  void _checkAIAvailability() {
    final medgemma = context.read<MedGemmaService>();
    setState(() {
      _aiAvailable = medgemma.isInitialized;
    });
  }

  Future<void> _runAnalysis() async {
    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    try {
      final service = context.read<LaborViewService>();
      final result = await service.analyze(widget.imageFile);

      if (mounted) {
        setState(() {
          _result = result;
          _isAnalyzing = false;
          if (result == null) {
            _errorMessage = 'Analysis failed';
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
          _errorMessage = e.toString();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D1117),
        title: const Text('Analysis'),
        actions: [
          if (_result != null)
            IconButton(
              icon: Icon(
                _showOverlay ? Icons.layers : Icons.layers_outlined,
                color: _showOverlay ? const Color(0xFF58A6FF) : Colors.grey,
              ),
              onPressed: () => setState(() => _showOverlay = !_showOverlay),
              tooltip: 'Toggle overlay',
            ),
        ],
      ),
      body: _isAnalyzing
          ? _buildLoadingView()
          : _errorMessage != null
              ? _buildErrorView()
              : _buildResultsView(),
    );
  }

  Widget _buildLoadingView() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: 32,
            height: 32,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          SizedBox(height: 16),
          Text(
            'Analyzing ultrasound...',
            style: TextStyle(color: Colors.grey, fontSize: 14),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 48, color: Color(0xFFF85149)),
            const SizedBox(height: 16),
            const Text(
              'Analysis Failed',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.grey, fontSize: 13),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _runAnalysis,
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsView() {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Image with overlay
          _buildImageSection(),

          // Results panel
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Labor Progress Assessment
                _buildProgressCard(),

                const SizedBox(height: 16),

                // Clinical Measurements
                _buildMeasurementsCard(),

                const SizedBox(height: 16),

                // Clinical Interpretation
                _buildInterpretationCard(),

                const SizedBox(height: 16),

                // AI Interpretation (MedGemma)
                _buildAIInterpretationCard(),

                const SizedBox(height: 16),

                // Actions
                _buildActions(),

                const SizedBox(height: 12),

                // Timestamp
                Center(
                  child: Text(
                    'Analyzed at ${_formatTimestamp(_result!.timestamp)}',
                    style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageSection() {
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      clipBehavior: Clip.antiAlias,
      child: AspectRatio(
        aspectRatio: 1.0, // Model output is 256x256, keep square
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.file(
              widget.imageFile,
              fit: BoxFit.fill, // Stretch to match model preprocessing (copyResize)
              width: double.infinity,
              height: double.infinity,
            ),
            if (_showOverlay && _result != null)
              SegmentationOverlay(
                mask: _result!.segmentationMask,
                opacity: 0.5,
              ),
          // Plane label
          Positioned(
            top: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                _result!.planeClass,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
          // Legend
          Positioned(
            bottom: 8,
            right: 8,
            child: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(width: 10, height: 10, color: const Color(0xFF00D4FF)),
                  const SizedBox(width: 4),
                  const Text('Symphysis', style: TextStyle(fontSize: 9, color: Colors.white)),
                  const SizedBox(width: 8),
                  Container(width: 10, height: 10, color: const Color(0xFFFF0080)),
                  const SizedBox(width: 4),
                  const Text('Head', style: TextStyle(fontSize: 9, color: Colors.white)),
                ],
              ),
            ),
          ),
          ],
        ),
      ),
    );
  }

  Widget _buildProgressCard() {
    final progress = _result!.laborProgress;
    Color color;
    IconData icon;

    switch (progress) {
      case LaborProgress.normal:
        color = const Color(0xFF3FB950);
        icon = Icons.check_circle;
        break;
      case LaborProgress.monitor:
        color = const Color(0xFFD29922);
        icon = Icons.schedule;
        break;
      case LaborProgress.concern:
        color = const Color(0xFFF85149);
        icon = Icons.warning;
        break;
      case LaborProgress.unknown:
        color = Colors.grey;
        icon = Icons.help_outline;
        break;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 24),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Labor Progress',
                      style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      progress.displayName,
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        color: color,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Row(
              children: [
                Icon(Icons.lightbulb_outline, size: 16, color: Colors.grey[500]),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _result!.recommendation,
                    style: const TextStyle(fontSize: 12, color: Colors.white),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMeasurementsCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Clinical Measurements',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[300],
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _buildMetricTile(
                  'AoP',
                  _result!.aop != null ? '${_result!.aop!.toStringAsFixed(1)}°' : '—',
                  'Angle of Progression',
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildMetricTile(
                  'HSD',
                  _result!.hsd != null ? '${_result!.hsd!.toStringAsFixed(0)} px' : '—',
                  'Head-Symphysis Dist.',
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildMetricTile(
                  'HC',
                  _result!.headCircumference != null
                      ? '${_result!.headCircumference!.toStringAsFixed(0)} px'
                      : '—',
                  'Head Circumference',
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildMetricTile(
                  'Area',
                  _result!.headArea != null
                      ? '${(_result!.headArea! / 1000).toStringAsFixed(1)}K px²'
                      : '—',
                  'Head Area',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricTile(String label, String value, String subtitle) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF21262D),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: Color(0xFF58A6FF),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: TextStyle(fontSize: 10, color: Colors.grey[600]),
          ),
        ],
      ),
    );
  }

  Widget _buildInterpretationCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Clinical Interpretation',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[300],
            ),
          ),
          const SizedBox(height: 12),
          if (_result!.aop != null) ...[
            _buildInterpretationRow('Angle of Progression', _result!.aopInterpretation),
            const Divider(color: Color(0xFF30363D), height: 20),
          ],
          if (_result!.hsd != null) ...[
            _buildInterpretationRow('Head-Symphysis Distance', _result!.hsdInterpretation),
          ],
          if (_result!.aop == null && _result!.hsd == null)
            Text(
              'Insufficient segmentation data for clinical interpretation. Ensure the image shows a clear transperineal view with visible pubic symphysis and fetal head.',
              style: TextStyle(fontSize: 12, color: Colors.grey[500]),
            ),
        ],
      ),
    );
  }

  Widget _buildInterpretationRow(String title, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(fontSize: 11, color: Colors.grey[500]),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(fontSize: 13, color: Colors.white),
        ),
      ],
    );
  }

  Widget _buildActions() {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.camera_alt, size: 18),
            label: const Text('New Scan'),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.icon(
            onPressed: _saveResults,
            icon: const Icon(Icons.save, size: 18),
            label: const Text('Save'),
          ),
        ),
      ],
    );
  }

  String _formatTimestamp(DateTime dt) {
    return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')} · ${dt.day}/${dt.month}/${dt.year}';
  }

  void _saveResults() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Results saved')),
    );
  }

  Future<void> _getAIInterpretation() async {
    final medgemma = context.read<MedGemmaService>();

    if (!medgemma.isInitialized) {
      // Show dialog to download MedGemma
      final shouldDownload = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          backgroundColor: const Color(0xFF161B22),
          title: const Text('Download MedGemma AI?'),
          content: const Text(
            'MedGemma provides AI-powered clinical interpretation. '
            'This requires a one-time download of ~2.5GB.\n\n'
            'The model runs entirely on-device for privacy.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('Download'),
            ),
          ],
        ),
      );

      if (shouldDownload != true) return;

      // Initialize MedGemma
      setState(() => _isLoadingAI = true);
      await medgemma.initialize();
      setState(() {
        _aiAvailable = medgemma.isInitialized;
        _isLoadingAI = false;
      });

      if (!medgemma.isInitialized) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to initialize: ${medgemma.errorMessage}')),
          );
        }
        return;
      }
    }

    // Run AI interpretation
    setState(() => _isLoadingAI = true);

    final interpretation = await medgemma.analyzeUltrasound(
      widget.imageFile,
      aop: _result?.aop,
      hsd: _result?.hsd,
      planeClass: _result?.planeClass,
    );

    if (mounted) {
      setState(() {
        _aiInterpretation = interpretation;
        _isLoadingAI = false;
      });
    }
  }

  Widget _buildAIInterpretationCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: _aiInterpretation != null
              ? const Color(0xFF58A6FF).withOpacity(0.3)
              : const Color(0xFF30363D),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.auto_awesome, size: 16, color: Color(0xFF58A6FF)),
              const SizedBox(width: 8),
              Text(
                'AI Clinical Interpretation',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[300],
                ),
              ),
              const Spacer(),
              if (!_aiAvailable && _aiInterpretation == null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF21262D),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    '~2.5GB',
                    style: TextStyle(fontSize: 10, color: Colors.grey[500]),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          if (_isLoadingAI) ...[
            Consumer<MedGemmaService>(
              builder: (context, service, _) {
                return Column(
                  children: [
                    if (service.downloadProgress > 0 && service.downloadProgress < 1)
                      ClipRRect(
                        borderRadius: BorderRadius.circular(2),
                        child: LinearProgressIndicator(
                          value: service.downloadProgress,
                          minHeight: 3,
                          backgroundColor: const Color(0xFF21262D),
                        ),
                      ),
                    const SizedBox(height: 8),
                    Text(
                      service.statusMessage,
                      style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                    ),
                  ],
                );
              },
            ),
          ] else if (_aiInterpretation != null) ...[
            Text(
              _aiInterpretation!,
              style: const TextStyle(fontSize: 13, color: Colors.white, height: 1.5),
            ),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: _getAIInterpretation,
                icon: const Icon(Icons.refresh, size: 16),
                label: const Text('Regenerate'),
                style: TextButton.styleFrom(
                  foregroundColor: Colors.grey[500],
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                ),
              ),
            ),
          ] else ...[
            Text(
              'Get AI-powered clinical interpretation using MedGemma, '
              'Google\'s medical AI model running entirely on your device.',
              style: TextStyle(fontSize: 12, color: Colors.grey[500]),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: _getAIInterpretation,
                icon: const Icon(Icons.auto_awesome, size: 16),
                label: Text(_aiAvailable ? 'Get AI Interpretation' : 'Download & Analyze'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFF58A6FF),
                  side: const BorderSide(color: Color(0xFF58A6FF)),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
