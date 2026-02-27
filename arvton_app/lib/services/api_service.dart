// ARVTON — API Service
// Dio HTTP client for communicating with the FastAPI backend.
// Supports retry on 5xx with exponential backoff (1s, 2s, 4s, max 3 attempts).

import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

/// Result model from GET /result/{jobId}
class TryonResult {
  final String status; // queued, processing, done, failed
  final String? glbUrl;
  final String? progress;
  final String? error;
  final int? durationMs;

  TryonResult({
    required this.status,
    this.glbUrl,
    this.progress,
    this.error,
    this.durationMs,
  });

  factory TryonResult.fromJson(Map<String, dynamic> json) {
    return TryonResult(
      status: json['status'] as String,
      glbUrl: json['glb_url'] as String?,
      progress: json['progress'] as String?,
      error: json['error'] as String?,
      durationMs: json['duration_ms'] as int?,
    );
  }
}

/// Health check response
class HealthStatus {
  final String status;
  final String gpuName;
  final double gpuMemoryUsedGb;
  final double gpuMemoryTotalGb;
  final int queueLength;
  final Map<String, bool> modelsLoaded;

  HealthStatus({
    required this.status,
    required this.gpuName,
    required this.gpuMemoryUsedGb,
    required this.gpuMemoryTotalGb,
    required this.queueLength,
    required this.modelsLoaded,
  });

  factory HealthStatus.fromJson(Map<String, dynamic> json) {
    return HealthStatus(
      status: json['status'] as String? ?? 'unknown',
      gpuName: json['gpu_name'] as String? ?? 'N/A',
      gpuMemoryUsedGb: (json['gpu_memory_used_gb'] as num?)?.toDouble() ?? 0,
      gpuMemoryTotalGb: (json['gpu_memory_total_gb'] as num?)?.toDouble() ?? 0,
      queueLength: json['queue_length'] as int? ?? 0,
      modelsLoaded: (json['models_loaded'] as Map<String, dynamic>?)
              ?.map((k, v) => MapEntry(k, v as bool)) ??
          {},
    );
  }
}

class ApiService {
  late final Dio _dio;
  static ApiService? _instance;

  ApiService._() {
    final baseUrl = dotenv.env['BACKEND_URL'] ?? 'http://localhost:8000';

    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 10),
      receiveTimeout: const Duration(seconds: 60),
      sendTimeout: const Duration(seconds: 60),
      headers: {
        'Accept': 'application/json',
      },
    ));

    // Add retry interceptor for 5xx errors
    _dio.interceptors.add(_RetryInterceptor(_dio));
  }

  /// Singleton
  factory ApiService() {
    _instance ??= ApiService._();
    return _instance!;
  }

  /// POST /tryon — submit try-on job
  /// Returns job_id string.
  Future<String> postTryon({
    required File personImage,
    required File garmentImage,
    String quality = 'auto',
  }) async {
    final formData = FormData.fromMap({
      'person_image': await MultipartFile.fromFile(
        personImage.path,
        filename: 'person.jpg',
      ),
      'garment_image': await MultipartFile.fromFile(
        garmentImage.path,
        filename: 'garment.jpg',
      ),
      'quality': quality,
    });

    final response = await _dio.post('/tryon', data: formData);

    if (response.statusCode == 202 || response.statusCode == 200) {
      return response.data['job_id'] as String;
    }

    throw DioException(
      requestOptions: response.requestOptions,
      response: response,
      message: 'Unexpected status: ${response.statusCode}',
    );
  }

  /// GET /result/{jobId} — poll for job status
  Future<TryonResult> getResult(String jobId) async {
    final response = await _dio.get('/result/$jobId');
    return TryonResult.fromJson(response.data as Map<String, dynamic>);
  }

  /// DELETE /job/{jobId} — cancel a job
  Future<void> deleteJob(String jobId) async {
    await _dio.delete('/job/$jobId');
  }

  /// GET /health — check backend health
  Future<HealthStatus> getHealth() async {
    final response = await _dio.get('/health');
    return HealthStatus.fromJson(response.data as Map<String, dynamic>);
  }

  /// Check if backend is reachable
  Future<bool> isBackendReachable() async {
    try {
      final response = await _dio.head(
        '/health',
        options: Options(
          receiveTimeout: const Duration(seconds: 5),
        ),
      );
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}

/// Retry interceptor with exponential backoff for 5xx errors.
/// Retries: 1s, 2s, 4s — max 3 attempts.
class _RetryInterceptor extends Interceptor {
  final Dio _dio;
  static const int _maxRetries = 3;
  static const List<int> _backoffMs = [1000, 2000, 4000];

  _RetryInterceptor(this._dio);

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    final statusCode = err.response?.statusCode ?? 0;

    if (statusCode >= 500 && statusCode < 600) {
      final retryCount =
          (err.requestOptions.extra['_retryCount'] as int?) ?? 0;

      if (retryCount < _maxRetries) {
        final delay = _backoffMs[retryCount];
        await Future.delayed(Duration(milliseconds: delay));

        final options = err.requestOptions;
        options.extra['_retryCount'] = retryCount + 1;

        try {
          final response = await _dio.fetch(options);
          return handler.resolve(response);
        } catch (e) {
          return handler.next(err);
        }
      }
    }

    handler.next(err);
  }
}
