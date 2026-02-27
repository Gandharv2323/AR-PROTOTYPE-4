// ARVTON — Screen 6: ShareScreen
// Screenshot capture of AR view, share, and save to gallery.

import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import '../providers/tryon_provider.dart';

class ShareScreen extends ConsumerStatefulWidget {
  const ShareScreen({super.key});

  @override
  ConsumerState<ShareScreen> createState() => _ShareScreenState();
}

class _ShareScreenState extends ConsumerState<ShareScreen>
    with SingleTickerProviderStateMixin {
  final GlobalKey _screenshotKey = GlobalKey();
  Uint8List? _screenshotBytes;
  bool _isSaving = false;
  bool _isSharing = false;
  late AnimationController _fadeController;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..forward();

    // Capture screenshot after build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _captureScreenshot();
    });
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  Future<void> _captureScreenshot() async {
    try {
      RenderRepaintBoundary? boundary =
          _screenshotKey.currentContext?.findRenderObject()
              as RenderRepaintBoundary?;

      if (boundary == null) return;

      ui.Image image = await boundary.toImage(pixelRatio: 3.0);
      ByteData? byteData = await image.toByteData(
        format: ui.ImageByteFormat.png,
      );

      if (byteData != null) {
        setState(() {
          _screenshotBytes = byteData.buffer.asUint8List();
        });
      }
    } catch (e) {
      debugPrint('Screenshot error: $e');
    }
  }

  Future<void> _shareImage() async {
    if (_screenshotBytes == null) return;

    setState(() => _isSharing = true);

    try {
      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/arvton_tryon.png');
      await file.writeAsBytes(_screenshotBytes!);

      await Share.shareXFiles([
        XFile(file.path),
      ], text: 'Check out my virtual try-on with ARVTON! 👗✨');
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Share failed: $e'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSharing = false);
    }
  }

  Future<void> _saveToGallery() async {
    if (_screenshotBytes == null) return;

    setState(() => _isSaving = true);

    try {
      // Save to app documents dir (gallery access requires platform-specific
      // plugins that may not be available)
      final dir = await getApplicationDocumentsDirectory();
      final fileName = 'arvton_${DateTime.now().millisecondsSinceEpoch}.png';
      final file = File('${dir.path}/$fileName');
      await file.writeAsBytes(_screenshotBytes!);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Text('Saved to ${file.path}'),
              ],
            ),
            backgroundColor: Colors.green.shade700,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Save failed: $e'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final tryonState = ref.watch(tryonProvider);

    return Scaffold(
      backgroundColor: const Color(0xFF0D0D2B),
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeController,
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () => context.pop(),
                      child: Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.arrow_back_ios_new,
                          color: Colors.white,
                          size: 18,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'Share Your Look',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // Screenshot preview
              Expanded(
                child: RepaintBoundary(
                  key: _screenshotKey,
                  child: Container(
                    margin: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(24),
                      color: const Color(0xFF1A1A40),
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF6C63FF).withOpacity(0.15),
                          blurRadius: 30,
                          spreadRadius: 5,
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          // Placeholder / preview
                          if (tryonState.personImage != null)
                            Image.file(
                              tryonState.personImage!,
                              fit: BoxFit.cover,
                            )
                          else
                            const Center(
                              child: Icon(
                                Icons.checkroom,
                                size: 80,
                                color: Color(0xFF6C63FF),
                              ),
                            ),

                          // ARVTON watermark
                          Positioned(
                            bottom: 16,
                            right: 16,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.black.withOpacity(0.5),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: const Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(
                                    Icons.checkroom,
                                    color: Color(0xFF6C63FF),
                                    size: 14,
                                  ),
                                  SizedBox(width: 4),
                                  Text(
                                    'ARVTON',
                                    style: TextStyle(
                                      color: Colors.white70,
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      letterSpacing: 2,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              // Action bar
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFF1A1A40),
                  border: Border(
                    top: BorderSide(color: Colors.white.withOpacity(0.06)),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    // Share
                    _ActionButton(
                      icon: Icons.share_rounded,
                      label: 'Share',
                      color: const Color(0xFF6C63FF),
                      isLoading: _isSharing,
                      onTap: _shareImage,
                    ),

                    // Save
                    _ActionButton(
                      icon: Icons.save_alt_rounded,
                      label: 'Save',
                      color: Colors.greenAccent.shade700,
                      isLoading: _isSaving,
                      onTap: _saveToGallery,
                    ),

                    // Try Another (keep person)
                    _ActionButton(
                      icon: Icons.checkroom,
                      label: 'Try Another',
                      color: Colors.amber.shade700,
                      onTap: () {
                        ref.read(tryonProvider.notifier).resetKeepPerson();
                        context.go('/garment-picker');
                      },
                    ),

                    // New Photo
                    _ActionButton(
                      icon: Icons.camera_alt,
                      label: 'New Photo',
                      color: Colors.pinkAccent,
                      onTap: () {
                        ref.read(tryonProvider.notifier).reset();
                        context.go('/camera');
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;
  final bool isLoading;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
    this.isLoading = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isLoading ? null : onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              shape: BoxShape.circle,
              border: Border.all(color: color.withOpacity(0.3)),
            ),
            child: isLoading
                ? Padding(
                    padding: const EdgeInsets.all(14),
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: color,
                    ),
                  )
                : Icon(icon, color: color, size: 22),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
