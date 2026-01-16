import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import '../services/laborview_service.dart';
import 'analysis_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LaborViewService>().initialize();
    });
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 90,
    );

    if (image != null && mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => AnalysisScreen(imageFile: File(image.path)),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Consumer<LaborViewService>(
          builder: (context, service, child) {
            return SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 16),

                  // Header
                  const Text(
                    'LaborView',
                    style: TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'AI-powered intrapartum ultrasound analysis',
                    style: TextStyle(fontSize: 13, color: Colors.grey[500]),
                  ),

                  const SizedBox(height: 28),

                  // Status
                  _buildStatusCard(service),

                  const SizedBox(height: 20),

                  // Action buttons
                  _buildActions(service),

                  const SizedBox(height: 28),

                  // Measurements info
                  _buildMeasurementsInfo(),

                  const SizedBox(height: 20),

                  // How it works
                  _buildHowItWorks(),

                  const SizedBox(height: 20),

                  // Disclaimer
                  _buildDisclaimer(),

                  const SizedBox(height: 16),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildStatusCard(LaborViewService service) {
    Color statusColor;
    String statusText;
    IconData statusIcon;

    if (service.isLoading) {
      statusColor = const Color(0xFFD29922);
      statusText = service.statusMessage;
      statusIcon = Icons.hourglass_top;
    } else if (service.isInitialized) {
      statusColor = const Color(0xFF3FB950);
      statusText = 'Model ready';
      statusIcon = Icons.check_circle;
    } else {
      statusColor = Colors.grey;
      statusText = 'Initializing...';
      statusIcon = Icons.circle_outlined;
    }

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Row(
        children: [
          Icon(statusIcon, color: statusColor, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  statusText,
                  style: TextStyle(
                    fontSize: 14,
                    color: statusColor,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                if (service.isLoading && service.downloadProgress > 0) ...[
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(2),
                    child: LinearProgressIndicator(
                      value: service.downloadProgress,
                      minHeight: 3,
                      backgroundColor: const Color(0xFF21262D),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActions(LaborViewService service) {
    final isReady = service.isInitialized && !service.isLoading;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        FilledButton.icon(
          onPressed: isReady ? () => _pickImage(ImageSource.camera) : null,
          icon: const Icon(Icons.camera_alt, size: 20),
          label: const Text('Capture Ultrasound'),
          style: FilledButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          onPressed: isReady ? () => _pickImage(ImageSource.gallery) : null,
          icon: const Icon(Icons.photo_library, size: 20),
          label: const Text('Select from Gallery'),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
        ),
      ],
    );
  }

  Widget _buildMeasurementsInfo() {
    return Container(
      padding: const EdgeInsets.all(14),
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
          const SizedBox(height: 12),
          _measurementRow('AoP', 'Angle of Progression', 'Labor stage indicator'),
          _measurementRow('HSD', 'Head-Symphysis Distance', 'Fetal descent'),
          _measurementRow('HC', 'Head Circumference', 'Head size measurement'),
          _measurementRow('Area', 'Head Area', 'Cross-sectional area'),
        ],
      ),
    );
  }

  Widget _measurementRow(String abbr, String name, String desc) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Container(
            width: 44,
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
            decoration: BoxDecoration(
              color: const Color(0xFF58A6FF).withOpacity(0.15),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              abbr,
              style: const TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: Color(0xFF58A6FF),
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: const TextStyle(fontSize: 13, color: Colors.white),
                ),
                Text(
                  desc,
                  style: TextStyle(fontSize: 11, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHowItWorks() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF30363D)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'How It Works',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[300],
            ),
          ),
          const SizedBox(height: 12),
          _stepRow('1', 'Capture transperineal ultrasound image'),
          _stepRow('2', 'AI segments pubic symphysis & fetal head'),
          _stepRow('3', 'Computes clinical measurements'),
          _stepRow('4', 'Provides labor progress assessment'),
        ],
      ),
    );
  }

  Widget _stepRow(String num, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [
          Container(
            width: 22,
            height: 22,
            decoration: BoxDecoration(
              color: const Color(0xFF21262D),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Center(
              child: Text(
                num,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[400],
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(fontSize: 12, color: Colors.grey[500]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8514933),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFF8514966)),
      ),
      child: Row(
        children: [
          const Icon(Icons.warning_amber, size: 18, color: Color(0xFFF85149)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Research prototype only. Not for clinical diagnosis. Always consult qualified healthcare professionals.',
              style: TextStyle(fontSize: 11, color: Colors.grey[400]),
            ),
          ),
        ],
      ),
    );
  }
}
