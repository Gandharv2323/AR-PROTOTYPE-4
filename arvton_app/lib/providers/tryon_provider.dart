// ARVTON — Tryon State Management (Riverpod)
// Manages the entire try-on flow: person selection → garment selection →
// job submission → polling → result display.

import 'dart:async';
import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/api_service.dart';

/// Status enum for the try-on flow
enum TryonStatus { idle, uploading, queued, processing, done, failed }

/// Immutable state for the try-on flow
class TryonState {
  final File? personImage;
  final File? garmentImage;
  final String? selectedGarmentId;
  final String? jobId;
  final TryonStatus status;
  final String? progress;
  final String? glbUrl;
  final String? errorMessage;

  const TryonState({
    this.personImage,
    this.garmentImage,
    this.selectedGarmentId,
    this.jobId,
    this.status = TryonStatus.idle,
    this.progress,
    this.glbUrl,
    this.errorMessage,
  });

  TryonState copyWith({
    File? personImage,
    File? garmentImage,
    String? selectedGarmentId,
    String? jobId,
    TryonStatus? status,
    String? progress,
    String? glbUrl,
    String? errorMessage,
  }) {
    return TryonState(
      personImage: personImage ?? this.personImage,
      garmentImage: garmentImage ?? this.garmentImage,
      selectedGarmentId: selectedGarmentId ?? this.selectedGarmentId,
      jobId: jobId ?? this.jobId,
      status: status ?? this.status,
      progress: progress ?? this.progress,
      glbUrl: glbUrl ?? this.glbUrl,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }
}

/// Notifier that manages TryonState transitions
class TryonNotifier extends StateNotifier<TryonState> {
  final ApiService _api;
  Timer? _pollTimer;

  TryonNotifier(this._api) : super(const TryonState());

  /// Select the person photo
  void selectPerson(File image) {
    state = state.copyWith(
      personImage: image,
      status: TryonStatus.idle,
      errorMessage: null,
      glbUrl: null,
      jobId: null,
    );
  }

  /// Select a garment
  void selectGarment(String garmentId, File image) {
    state = state.copyWith(selectedGarmentId: garmentId, garmentImage: image);
  }

  /// Submit try-on job to backend
  Future<void> submitTryon({String quality = 'auto'}) async {
    if (state.personImage == null || state.garmentImage == null) return;

    state = state.copyWith(
      status: TryonStatus.uploading,
      errorMessage: null,
      glbUrl: null,
    );

    try {
      final jobId = await _api.postTryon(
        personImage: state.personImage!,
        garmentImage: state.garmentImage!,
        quality: quality,
      );

      state = state.copyWith(
        jobId: jobId,
        status: TryonStatus.queued,
        progress: 'queued',
      );

      // Start polling
      _startPolling();
    } catch (e) {
      // Demo mode fallback: use mock GLB URL
      state = state.copyWith(
        status: TryonStatus.done,
        glbUrl: 'https://modelviewer.dev/shared-assets/models/Astronaut.glb',
        progress: 'Demo mode — backend offline',
      );
    }
  }

  /// Poll GET /result/{jobId} every 2 seconds
  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      await pollResult();
    });
  }

  /// Single poll attempt
  Future<void> pollResult() async {
    final jobId = state.jobId;
    if (jobId == null) return;

    try {
      final result = await _api.getResult(jobId);

      switch (result.status) {
        case 'queued':
          state = state.copyWith(
            status: TryonStatus.queued,
            progress: result.progress ?? 'queued',
          );
          break;

        case 'processing':
          state = state.copyWith(
            status: TryonStatus.processing,
            progress: result.progress ?? 'processing',
          );
          break;

        case 'done':
          _pollTimer?.cancel();
          state = state.copyWith(
            status: TryonStatus.done,
            glbUrl: result.glbUrl,
            progress: 'done',
          );
          break;

        case 'failed':
          _pollTimer?.cancel();
          state = state.copyWith(
            status: TryonStatus.failed,
            errorMessage: result.error ?? 'Unknown error',
          );
          break;
      }
    } catch (e) {
      // Don't stop polling on transient errors
    }
  }

  /// Cancel the current job
  Future<void> cancelJob() async {
    _pollTimer?.cancel();
    final jobId = state.jobId;
    if (jobId != null) {
      try {
        await _api.deleteJob(jobId);
      } catch (_) {}
    }
    state = state.copyWith(
      status: TryonStatus.idle,
      jobId: null,
      progress: null,
    );
  }

  /// Reset to initial state (new photo)
  void reset() {
    _pollTimer?.cancel();
    state = const TryonState();
  }

  /// Reset but keep person image (try another garment)
  void resetKeepPerson() {
    _pollTimer?.cancel();
    state = TryonState(personImage: state.personImage);
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }
}

/// Riverpod providers
final apiServiceProvider = Provider<ApiService>((ref) {
  return ApiService();
});

final tryonProvider = StateNotifierProvider<TryonNotifier, TryonState>((ref) {
  return TryonNotifier(ref.watch(apiServiceProvider));
});

/// Backend reachability provider (auto-refresh)
final backendReachableProvider = FutureProvider<bool>((ref) async {
  return ref.watch(apiServiceProvider).isBackendReachable();
});
