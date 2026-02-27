# ARVTON — Flutter AR Frontend

Virtual try-on experience with AR/3D visualization.

## Prerequisites

- Flutter 3.22+ / Dart 3.4+
- Xcode 15+ (iOS builds)
- Android Studio with NDK (Android builds)
- ARVTON backend running (local or remote)

## Setup

### 1. Environment
```bash
cp .env.example .env
# Edit .env:
#   BACKEND_URL=http://localhost:8000
#   API_KEY=your-api-key
```

### 2. Install Dependencies
```bash
flutter pub get
```

### 3. iOS Setup
```bash
cd ios && pod install && cd ..
```

Add to `ios/Runner/Info.plist`:
```xml
<key>NSCameraUsageDescription</key>
<string>Required for virtual try-on AR view</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Select your photo to try on clothes</string>
<key>NSPhotoLibraryAddUsageDescription</key>
<string>Save your try-on result</string>
```

### 4. Android Setup
Add to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-feature android:name="android.hardware.camera.ar" android:required="true"/>
```

Set `minSdkVersion 24` in `android/app/build.gradle`.

## Run

### Development
```bash
flutter run
```

### Local Backend with ngrok
```bash
# Terminal 1: Start backend
cd .. && python -m pipeline.server

# Terminal 2: Tunnel
ngrok http 8000

# Update .env with ngrok URL
BACKEND_URL=https://xxxx.ngrok.io
```

### Production Build
```bash
# iOS
flutter build ios --release

# Android
flutter build apk --release
```

## Architecture

```
lib/
├── main.dart                 # Entry point, GoRouter, theme
├── services/
│   └── api_service.dart      # Dio HTTP client with retry
├── providers/
│   └── tryon_provider.dart   # Riverpod state management
└── screens/
    ├── splash_screen.dart    # Animated splash + health check
    ├── camera_screen.dart    # Camera capture + gallery
    ├── garment_picker_screen.dart  # 16-item catalog grid
    ├── processing_screen.dart     # Status polling + animations
    ├── ar_viewer_screen.dart      # AR/3D model viewer
    └── share_screen.dart          # Screenshot + share + save
```

## Screens

| # | Screen | Description |
|---|--------|-------------|
| 1 | Splash | Logo animation, backend health check, demo mode fallback |
| 2 | Camera | Full-screen preview, capture, gallery, portrait validation |
| 3 | Garment Picker | 2-column grid, category tabs, floating "Try On" bar |
| 4 | Processing | Dynamic status text, cancel support, error screen |
| 5 | AR Viewer | model_viewer_plus, scale slider, rotate, fallback 3D |
| 6 | Share | Screenshot capture, share_plus, save, try another |

## Troubleshooting

### AR Not Working on Android
- Ensure device supports ARCore (check [supported devices](https://developers.google.com/ar/devices))
- Install Google Play Services for AR from Play Store
- The app will show a "View in 3D" fallback after 12 seconds

### Camera Black Screen
- Check camera permissions in device settings
- Restart the app after granting permissions

### Backend Connection Issues
- Verify `BACKEND_URL` in `.env` is correct
- Check if backend is running: `curl http://localhost:8000/health`
- App will run in demo mode if backend is unreachable

## Known Limitations

- AR features require a physical device (not available in simulators)
- GLB model quality depends on the reconstruction backend
- Demo mode uses a sample astronaut model for testing
- Gallery save goes to app documents directory (not system gallery)
