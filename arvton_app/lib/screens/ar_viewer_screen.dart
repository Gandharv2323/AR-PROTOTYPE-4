// ARVTON — Screen 5: ARViewerScreen
// AR view with plane detection, GLB model placement,
// scale slider, rotation, and 3D fallback viewer.

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';
import '../providers/tryon_provider.dart';

class ARViewerScreen extends ConsumerStatefulWidget {
  const ARViewerScreen({super.key});

  @override
  ConsumerState<ARViewerScreen> createState() => _ARViewerScreenState();
}

class _ARViewerScreenState extends ConsumerState<ARViewerScreen>
    with SingleTickerProviderStateMixin {
  bool _planeDetected = false;
  bool _modelPlaced = false;
  bool _isRotating = false;
  double _scale = 1.0;
  Timer? _fallbackTimer;
  late AnimationController _scanPulse;
  bool _showFallbackBanner = false;

  @override
  void initState() {
    super.initState();

    _scanPulse = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    // Start fallback timer — 12 seconds
    _fallbackTimer = Timer(const Duration(seconds: 12), () {
      if (!_planeDetected && mounted) {
        setState(() => _showFallbackBanner = true);
      }
    });

    // Simulate plane detection for demo mode
    Timer(const Duration(seconds: 3), () {
      if (mounted && !_planeDetected) {
        setState(() {
          _planeDetected = true;
          _modelPlaced = true;
        });
      }
    });
  }

  @override
  void dispose() {
    _scanPulse.dispose();
    _fallbackTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final tryonState = ref.watch(tryonProvider);
    final glbUrl =
        tryonState.glbUrl ??
        'https://modelviewer.dev/shared-assets/models/Astronaut.glb';

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // 3D Model Viewer (used as primary viewer — AR plugin requires
          // physical device, so model_viewer_plus gives universal support)
          Positioned.fill(
            child: ModelViewer(
              src: glbUrl,
              alt: 'Virtual try-on result',
              ar: true,
              arModes: const ['scene-viewer', 'webxr', 'quick-look'],
              autoPlay: true,
              autoRotate: _isRotating,
              cameraControls: true,
              backgroundColor: const Color(0xFF0D0D2B),
              scale: '$_scale $_scale $_scale',
              shadowIntensity: 1.0,
              shadowSoftness: 0.5,
              exposure: 1.0,
            ),
          ),

          // Scanning overlay (before plane detected)
          if (!_planeDetected) _buildScanOverlay(),

          // Scale-in animation when model is first placed
          if (_modelPlaced && !_planeDetected) _buildPlacementAnimation(),

          // Top bar
          _buildTopBar(),

          // Bottom controls
          _buildBottomControls(),

          // Scale slider (right edge)
          _buildScaleSlider(),

          // Fallback banner
          if (_showFallbackBanner) _buildFallbackBanner(),
        ],
      ),
    );
  }

  Widget _buildScanOverlay() {
    return AnimatedBuilder(
      animation: _scanPulse,
      builder: (context, _) {
        return Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 200,
                height: 200,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: const Color(
                      0xFF6C63FF,
                    ).withOpacity(0.3 + _scanPulse.value * 0.3),
                    width: 3,
                  ),
                ),
                child: Icon(
                  Icons.local_airport_rounded,
                  size: 64,
                  color: const Color(
                    0xFF6C63FF,
                  ).withOpacity(0.5 + _scanPulse.value * 0.3),
                ),
              ),
              const SizedBox(height: 24),
              Text(
                'Move your phone slowly\nto detect the floor',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withOpacity(0.7),
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildPlacementAnimation() {
    return const SizedBox.shrink();
  }

  Widget _buildTopBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // Back button
            _glassButton(
              icon: Icons.arrow_back_ios_new,
              onTap: () => context.pop(),
            ),

            // Title
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.4),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.white.withOpacity(0.1)),
              ),
              child: const Text(
                'AR View',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            // Share button
            _glassButton(
              icon: Icons.share_rounded,
              onTap: () => context.push('/share'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomControls() {
    return Positioned(
      bottom: 40,
      left: 20,
      right: 20,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // View in 3D button
          _controlButton(
            icon: Icons.view_in_ar,
            label: 'View in 3D',
            onTap: () {
              final tryonState = ref.read(tryonProvider);
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => ModelViewerFallbackScreen(
                    glbUrl:
                        tryonState.glbUrl ??
                        'https://modelviewer.dev/shared-assets/models/Astronaut.glb',
                  ),
                ),
              );
            },
          ),

          // Rotate toggle
          _controlButton(
            icon: _isRotating
                ? Icons.rotate_right
                : Icons.rotate_90_degrees_ccw,
            label: _isRotating ? 'Stop' : 'Rotate',
            isActive: _isRotating,
            onTap: () => setState(() => _isRotating = !_isRotating),
          ),
        ],
      ),
    );
  }

  Widget _buildScaleSlider() {
    return Positioned(
      right: 16,
      top: MediaQuery.of(context).size.height * 0.25,
      bottom: MediaQuery.of(context).size.height * 0.25,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.add, color: Colors.white54, size: 18),
          Expanded(
            child: RotatedBox(
              quarterTurns: 3,
              child: SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  trackHeight: 3,
                  thumbShape: const RoundSliderThumbShape(
                    enabledThumbRadius: 8,
                  ),
                  activeTrackColor: const Color(0xFF6C63FF),
                  inactiveTrackColor: Colors.white.withOpacity(0.1),
                  thumbColor: Colors.white,
                  overlayColor: const Color(0xFF6C63FF).withOpacity(0.2),
                ),
                child: Slider(
                  value: _scale,
                  min: 0.5,
                  max: 2.0,
                  onChanged: (v) => setState(() => _scale = v),
                ),
              ),
            ),
          ),
          const Icon(Icons.remove, color: Colors.white54, size: 18),
          const SizedBox(height: 4),
          Text(
            '${_scale.toStringAsFixed(1)}x',
            style: const TextStyle(color: Colors.white54, fontSize: 11),
          ),
        ],
      ),
    );
  }

  Widget _buildFallbackBanner() {
    return Positioned(
      bottom: 100,
      left: 32,
      right: 32,
      child: GestureDetector(
        onTap: () {
          final tryonState = ref.read(tryonProvider);
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => ModelViewerFallbackScreen(
                glbUrl:
                    tryonState.glbUrl ??
                    'https://modelviewer.dev/shared-assets/models/Astronaut.glb',
              ),
            ),
          );
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          decoration: BoxDecoration(
            color: Colors.amber.shade900.withOpacity(0.9),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.amber.shade900.withOpacity(0.3),
                blurRadius: 12,
              ),
            ],
          ),
          child: const Row(
            children: [
              Icon(Icons.warning_amber, color: Colors.white, size: 20),
              SizedBox(width: 10),
              Expanded(
                child: Text(
                  'AR not working? Tap to view in 3D instead',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              Icon(Icons.arrow_forward_ios, color: Colors.white70, size: 14),
            ],
          ),
        ),
      ),
    );
  }

  Widget _glassButton({required IconData icon, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.3),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.15)),
        ),
        child: Icon(icon, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _controlButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isActive = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: isActive
              ? const Color(0xFF6C63FF).withOpacity(0.8)
              : Colors.black.withOpacity(0.4),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: isActive
                ? const Color(0xFF6C63FF)
                : Colors.white.withOpacity(0.15),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 18),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Fallback 360° 3D model viewer (no AR required)
class ModelViewerFallbackScreen extends StatefulWidget {
  final String glbUrl;

  const ModelViewerFallbackScreen({super.key, required this.glbUrl});

  @override
  State<ModelViewerFallbackScreen> createState() =>
      _ModelViewerFallbackScreenState();
}

class _ModelViewerFallbackScreenState extends State<ModelViewerFallbackScreen> {
  double _scale = 1.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D2B),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0D0D2B),
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          '3D Viewer',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          // Model viewer
          Positioned.fill(
            child: ModelViewer(
              src: widget.glbUrl,
              alt: 'Try-on 3D model',
              autoRotate: true,
              cameraControls: true,
              backgroundColor: const Color(0xFF0D0D2B),
              scale: '$_scale $_scale $_scale',
              shadowIntensity: 1.0,
              exposure: 1.2,
            ),
          ),

          // Scale control
          Positioned(
            bottom: 40,
            left: 40,
            right: 40,
            child: Column(
              children: [
                Text(
                  'Scale: ${_scale.toStringAsFixed(1)}x',
                  style: const TextStyle(color: Colors.white54, fontSize: 12),
                ),
                const SizedBox(height: 8),
                SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    trackHeight: 3,
                    thumbShape: const RoundSliderThumbShape(
                      enabledThumbRadius: 8,
                    ),
                    activeTrackColor: const Color(0xFF6C63FF),
                    inactiveTrackColor: Colors.white.withOpacity(0.1),
                    thumbColor: Colors.white,
                  ),
                  child: Slider(
                    value: _scale,
                    min: 0.5,
                    max: 2.0,
                    onChanged: (v) => setState(() => _scale = v),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
