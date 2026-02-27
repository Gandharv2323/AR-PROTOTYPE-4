// ARVTON — Screen 4: ProcessingScreen
// Full-screen dark background with Lottie animation and dynamic status text.
// Polls GET /result/{jobId} every 2 seconds.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../providers/tryon_provider.dart';

class ProcessingScreen extends ConsumerStatefulWidget {
  const ProcessingScreen({super.key});

  @override
  ConsumerState<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends ConsumerState<ProcessingScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseScale;

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _pulseScale = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  String _getStatusText(TryonState state) {
    final progress = state.progress?.toLowerCase() ?? '';

    if (state.status == TryonStatus.uploading) return 'Uploading images...';

    if (progress.contains('queued') || state.status == TryonStatus.queued) {
      return 'Getting in line...';
    }
    if (progress.contains('segment')) return 'Removing backgrounds...';
    if (progress.contains('generat') || progress.contains('tryon')) {
      return 'Dressing you up...';
    }
    if (progress.contains('3d') || progress.contains('reconstruct')) {
      return 'Shaping your 3D avatar...';
    }
    if (progress.contains('export')) return 'Almost ready...';
    if (progress.contains('demo')) return 'Demo mode — using sample model';

    return 'Processing...';
  }

  String _getSubText(TryonState state) {
    if (state.status == TryonStatus.queued) {
      return 'This usually takes 30-90 seconds';
    }
    if (state.status == TryonStatus.processing) {
      return 'AI is working its magic';
    }
    return '';
  }

  IconData _getStatusIcon(TryonState state) {
    final progress = state.progress?.toLowerCase() ?? '';

    if (progress.contains('segment')) return Icons.content_cut;
    if (progress.contains('generat') || progress.contains('tryon')) {
      return Icons.checkroom;
    }
    if (progress.contains('3d') || progress.contains('reconstruct')) {
      return Icons.view_in_ar;
    }
    if (progress.contains('export')) return Icons.file_download;
    return Icons.auto_awesome;
  }

  @override
  Widget build(BuildContext context) {
    final tryonState = ref.watch(tryonProvider);

    // Navigate on completion
    ref.listen<TryonState>(tryonProvider, (prev, next) {
      if (next.status == TryonStatus.done && next.glbUrl != null) {
        context.go('/ar-viewer');
      }
    });

    // Full-screen error state
    if (tryonState.status == TryonStatus.failed) {
      return _buildErrorScreen(tryonState);
    }

    return Scaffold(
      backgroundColor: const Color(0xFF0D0D2B),
      body: Stack(
        children: [
          // Animated background particles
          ...List.generate(6, (i) => _FloatingOrb(index: i)),

          // Content
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Animated icon
                AnimatedBuilder(
                  animation: _pulseController,
                  builder: (context, child) {
                    return Transform.scale(
                      scale: _pulseScale.value,
                      child: child,
                    );
                  },
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: RadialGradient(
                        colors: [
                          const Color(0xFF6C63FF).withOpacity(0.3),
                          const Color(0xFF6C63FF).withOpacity(0.05),
                        ],
                      ),
                    ),
                    child: Icon(
                      _getStatusIcon(tryonState),
                      size: 48,
                      color: const Color(0xFF6C63FF),
                    ),
                  ),
                ),

                const SizedBox(height: 40),

                // Progress indicator
                SizedBox(
                  width: 200,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: LinearProgressIndicator(
                      backgroundColor: Colors.white.withOpacity(0.08),
                      color: const Color(0xFF6C63FF),
                      minHeight: 4,
                    ),
                  ),
                ),

                const SizedBox(height: 32),

                // Status text
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 400),
                  child: Text(
                    _getStatusText(tryonState),
                    key: ValueKey(tryonState.progress),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.w600,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),

                const SizedBox(height: 8),

                Text(
                  _getSubText(tryonState),
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.5),
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),

          // Cancel button
          Positioned(
            top: MediaQuery.of(context).padding.top + 12,
            left: 16,
            child: GestureDetector(
              onTap: () {
                ref.read(tryonProvider.notifier).cancelJob();
                context.pop();
              },
              child: Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.08),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.close, color: Colors.white70, size: 22),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorScreen(TryonState state) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D2B),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.red.shade900.withOpacity(0.3),
                ),
                child: const Icon(
                  Icons.error_outline,
                  color: Colors.redAccent,
                  size: 56,
                ),
              ),

              const SizedBox(height: 24),

              const Text(
                'Something went wrong',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                ),
              ),

              const SizedBox(height: 12),

              Text(
                state.errorMessage ?? 'Unknown error occurred',
                style: TextStyle(
                  color: Colors.white.withOpacity(0.5),
                  fontSize: 14,
                ),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 32),

              // Try Again
              ElevatedButton.icon(
                onPressed: () {
                  ref.read(tryonProvider.notifier).submitTryon();
                },
                icon: const Icon(Icons.refresh),
                label: const Text('Try Again'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF6C63FF),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 14,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
              ),

              const SizedBox(height: 12),

              // Go Back
              TextButton(
                onPressed: () => context.pop(),
                child: const Text(
                  'Go Back',
                  style: TextStyle(color: Colors.white70),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Floating orb for background ambience
class _FloatingOrb extends StatefulWidget {
  final int index;
  const _FloatingOrb({required this.index});

  @override
  State<_FloatingOrb> createState() => _FloatingOrbState();
}

class _FloatingOrbState extends State<_FloatingOrb>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _position;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: Duration(seconds: 4 + widget.index * 2),
    )..repeat(reverse: true);

    final dx = (widget.index % 3 - 1) * 0.15;
    final dy = (widget.index % 2 == 0 ? -1 : 1) * 0.1;

    _position = Tween<Offset>(
      begin: Offset(dx - 0.05, dy - 0.05),
      end: Offset(dx + 0.05, dy + 0.05),
    ).animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = 60.0 + widget.index * 30;
    final opacity = 0.04 + widget.index * 0.01;

    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Positioned(
          left:
              MediaQuery.of(context).size.width *
              (0.1 + widget.index * 0.15 + _position.value.dx),
          top:
              MediaQuery.of(context).size.height *
              (0.1 + widget.index * 0.12 + _position.value.dy),
          child: Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: const Color(0xFF6C63FF).withOpacity(opacity),
            ),
          ),
        );
      },
    );
  }
}
