# AR/VR Virtual Try-On Pipeline
## Claude Opus 4.6 — AI Agent Prompts
> Feed each prompt as a **separate, independent conversation** with Claude Opus 4.6.  
> Each prompt is fully self-contained. Execute them in order: Prompt 01 → 08.

---

## PROMPT 01 — Environment Setup & SAM 2 Segmentation

```
TASK IDENTITY: You are an expert ML engineer building an AR virtual try-on pipeline on Google Colab. This is Task 01 of 08.

OBJECTIVE: Set up the complete Python environment on Google Colab and implement a production-quality segmentation module using SAM 2.

PLATFORM: Google Colab (T4 or A100). Mount Google Drive to /content/drive/MyDrive/arvton/.

---

STEP 1 — Install all dependencies in a single pip cell:
  torch>=2.3 (CUDA 11.8), torchvision, diffusers, transformers, accelerate,
  segment-anything-2, opencv-python-headless, pillow, trimesh, pymeshlab,
  fastapi, uvicorn, python-multipart, requests.

STEP 2 — Download the sam2-hiera-large checkpoint from Meta's Hugging Face
  repository (facebook/sam2-hiera-large).
  Save to: /content/drive/MyDrive/arvton/checkpoints/sam2/

STEP 3 — Write a Python module named segment.py containing:
  - Function: segment_garment(image_path: str) -> PIL.Image
      Removes background, returns RGBA PNG.
  - Function: segment_person(image_path: str) -> PIL.Image
      Isolates human body region, returns RGBA PNG.
  - Both functions must handle edge cases: non-square images,
    low-contrast garments, partial occlusion.
  - Add docstrings and inline comments throughout.

STEP 4 — Write a test cell that:
  - Runs both functions on a sample garment image and a sample person image.
  - Displays the outputs side by side using matplotlib.
  - Prints VRAM usage after inference.

SUCCESS CRITERIA:
  - Both functions produce clean RGBA masks with no background bleed.
  - VRAM usage stays under 10GB.
  - Code is modular and importable as a Python module.

OUTPUT: Provide the complete, runnable Colab notebook in Python code blocks.
Do not truncate any code. Include all import statements.
```

---

## PROMPT 02 — 2D Virtual Try-On with Leffa & CatVTON

```
TASK IDENTITY: You are an expert ML engineer building an AR virtual try-on pipeline. This is Task 02 of 08.

OBJECTIVE: Implement the 2D virtual try-on stage with two execution paths —
  Leffa (quality path) and CatVTON (fast fallback path).

PREREQUISITE: segment.py from Task 01 is available at:
  /content/drive/MyDrive/arvton/segment.py

---

STEP 1 — Implement the Leffa quality path:
  - Load Leffa from Hugging Face (franciszzj/Leffa).
  - Use float32 on A100, float16 on T4.
  - Target output resolution: 768x1024.
  - Save result to: /content/drive/MyDrive/arvton/outputs/tryon_leffa.png

STEP 2 — Implement the CatVTON fallback path:
  - Load CatVTON from Hugging Face (zhengchong/CatVTON).
  - Enable attention slicing and VAE tiling.
  - Use torch.float16.
  - Target output resolution: 512x768.
  - Save result to: /content/drive/MyDrive/arvton/outputs/tryon_catvton.png

STEP 3 — Write a unified module named tryon.py containing:
  - Function: run_tryon(person_rgba: PIL.Image, garment_rgba: PIL.Image, quality: str = "fast") -> PIL.Image
  - quality="quality" uses Leffa. quality="fast" uses CatVTON.
  - Function auto-detects available VRAM and falls back if needed.
  - Includes proper error handling, logging, and VRAM monitoring.

STEP 4 — Write a comparison test cell that:
  - Runs both paths on identical inputs.
  - Displays outputs side by side.
  - Prints inference time (seconds) and VRAM usage (GB) below each output.

SUCCESS CRITERIA:
  - Leffa produces photorealistic fabric with no warping artefacts.
  - CatVTON produces acceptable output in under 8 seconds on T4.
  - Both paths callable from a single run_tryon() function call.

OUTPUT: Complete tryon.py module + Colab test cells. No truncation.
```

---

## PROMPT 03 — 3D Body Reconstruction (HMR 2.0 + ECON)

```
TASK IDENTITY: You are an expert ML engineer building an AR virtual try-on pipeline. This is Task 03 of 08.

OBJECTIVE: Implement 3D body reconstruction from the 2D try-on composite image
  using HMR 2.0 for body pose/shape and ECON for clothed surface geometry.

PREREQUISITES:
  - segment.py is available at /content/drive/MyDrive/arvton/segment.py
  - tryon.py is available at /content/drive/MyDrive/arvton/tryon.py
  - Input to this stage is the PIL.Image output from run_tryon()

---

STEP 1 — HMR 2.0 (Human Mesh Recovery):
  - Install: pip install 4D-Humans (gzb1-lab/4D-Humans)
  - Load the hmr2-0b checkpoint.
  - Write function: estimate_body(composite_image: PIL.Image) -> dict
      Returns SMPL theta, beta, camera_translation as numpy arrays.
  - Visualize the SMPL mesh overlay on the input image for debugging.

STEP 2 — ECON (Explicit Clothed humans Obtained from Normals):
  - Clone ECON from its GitHub repo and install all dependencies.
  - Write function: reconstruct_clothed(composite_image: PIL.Image, smpl_params: dict) -> str
      Returns path to output .obj file.
  - Handle the full ECON pipeline: normal estimation → IF-Net → SMPL-D fitting.
  - Save output .obj to: /content/drive/MyDrive/arvton/outputs/clothed_mesh.obj

STEP 3 — Write a module named reconstruct3d.py containing:
  - Function: build_clothed_mesh(composite_image: PIL.Image) -> str (path to .obj)
  - Include UV unwrapping using xatlas for texture mapping readiness.
  - Include progress logging at each sub-step (normal estimation, IF-Net, fitting).

STEP 4 — Validate the output .obj using trimesh:
  - Print: vertex count, face count.
  - Check and print watertightness status.
  - Raise an AssertionError if vertex count < 10,000.

SUCCESS CRITERIA:
  - Output .obj opens correctly in trimesh with no degenerate faces.
  - UV coordinates are present.
  - Vertex count > 10,000.
  - Code is AMD ROCm compatible (replace any CUDA-specific calls with equivalents).

OUTPUT: Complete reconstruct3d.py module + validation cell. No truncation.
```

---

## PROMPT 04 — .glb Export (TripoSR Fast Path + SyncHuman Quality Path)

```
TASK IDENTITY: You are an expert ML engineer building an AR virtual try-on pipeline. This is Task 04 of 08.

OBJECTIVE: Implement the final 3D export stage producing a .glb file
  suitable for direct rendering in the Flutter AR app.

PREREQUISITE: reconstruct3d.py is available. Input is a .obj file path
  from build_clothed_mesh().

---

STEP 1 — TripoSR fast path:
  - Install: pip install tsr (VAST-AI-Research/TripoSR)
  - Write function: export_glb_fast(composite_image: PIL.Image) -> str (path to .glb)
  - Runs TripoSR directly on the 2D composite image.
  - Target latency: under 15 seconds on T4.
  - Apply texture from the composite image onto the mesh before export.

STEP 2 — SyncHuman quality path:
  - Clone SyncHuman from its research repository.
  - Install dependencies carefully. Document any CUDA version requirements
    and provide the ROCm equivalent commands as inline comments.
  - Write function: export_glb_quality(clothed_obj_path: str, reference_image: PIL.Image) -> str
  - Document expected runtime: on A100 and on AMD MI250.

STEP 3 — Write a module named export3d.py containing:
  - Function: export_to_glb(composite: PIL.Image, obj_path: str = None, quality: str = "fast") -> str
  - "fast" calls TripoSR, "quality" calls SyncHuman.
  - Validate output with trimesh: confirm .glb loads, has textures, is under 15MB.
  - Apply Draco mesh compression if trimesh.exchange.gltf supports it.

STEP 4 — Write an inline three.js viewer cell:
  - Uses Colab IPython.display.HTML to render the .glb in the notebook.
  - Allows basic orbit rotation so the mesh can be visually inspected
    without leaving the notebook.

SUCCESS CRITERIA:
  - Fast path .glb completes in under 15 seconds on T4.
  - Quality path .glb opens in Google's model-viewer without errors.
  - Output file size under 15MB.
  - Both paths return a valid file path string.

OUTPUT: Complete export3d.py module + inline three.js viewer cell. No truncation.
```

---

## PROMPT 05 — Fine-Tuning CatVTON on Custom Clothing Dataset

```
TASK IDENTITY: You are an expert ML training engineer. This is Task 05 of 08.

OBJECTIVE: Fine-tune CatVTON on a proprietary clothing catalog to improve
  accuracy for brand-specific garments, textures, and non-standard body types.

PLATFORM: AMD ROCm platform (MI250 or MI300).
  PyTorch must be installed via the ROCm wheel:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

---

STEP 1 — Dataset class:
  Write VTONDataset(torch.utils.data.Dataset) that loads from this structure:
    /dataset/train/
      person/          # 512x768 person photos
      garment/         # flat-lay garment images, white background
      garment_mask/    # binary garment masks (RGBA from SAM 2)
      person_parse/    # body part segmentation maps

  Apply augmentations:
    - Random horizontal flip (p=0.5)
    - Color jitter: brightness=0.2, contrast=0.2, saturation=0.1
    - Random crop then resize to 512x768

  DataLoader settings: batch_size=16 (MI250), num_workers=8, pin_memory=True.

STEP 2 — Attention-only fine-tuning:
  - Load CatVTON base weights from Hugging Face (zhengchong/CatVTON).
  - Freeze all parameters except layers with "attn" in their name.
  - Print: number of trainable parameters vs total parameters.
  - Optimizer: AdamW, lr=1e-5, weight_decay=0.01.
  - LR schedule: cosine decay with 200-step linear warmup.
  - Mixed precision: bf16 preferred on AMD MI250. Fallback to fp16 if unavailable.
    Detect with: torch.cuda.is_bf16_supported()

STEP 3 — Training loop must include:
  - Checkpoint save every 500 steps to: /mnt/storage/arvton_ckpts/step_{N}.pt
  - Resume-from-checkpoint logic that auto-loads the latest checkpoint on startup.
  - Logging: loss and learning rate to both console and TensorBoard.
  - Sample output images saved to /mnt/storage/arvton_samples/ every 1000 steps.
  - ROCm verification: assert torch.cuda.is_available() == True at startup.

STEP 4 — Evaluation after training:
  - Compute SSIM and LPIPS on the validation split.
  - Print a comparison table: base model score vs fine-tuned model score.
  - Display 5 side-by-side validation sample comparisons.

SUCCESS CRITERIA:
  - Fine-tuned SSIM > base model SSIM on validation set.
  - Training runs stably for 5000 steps without OOM error.
  - All checkpoints saved and resumable.
  - Script is fully ROCm compatible — no hardcoded CUDA-only calls.

OUTPUT: Complete finetune.py script + the exact terminal command to launch
  training on the AMD platform. No truncation.
```

---

## PROMPT 06 — FastAPI Backend with Async Job Queue

```
TASK IDENTITY: You are an expert Python backend engineer. This is Task 06 of 08.

OBJECTIVE: Build a production-ready FastAPI backend that exposes the complete
  AR try-on pipeline as a REST API with asynchronous job handling and Docker deployment.

PREREQUISITES: The following modules are available in /app/pipeline/:
  segment.py, tryon.py, reconstruct3d.py, export3d.py

---

STEP 1 — Create this exact project structure:
  /app
    main.py              # FastAPI app entry point
    pipeline/            # all pipeline modules from prior tasks
      __init__.py
      segment.py
      tryon.py
      reconstruct3d.py
      export3d.py
    jobs/
      __init__.py
      store.py           # thread-safe job state management
      worker.py          # pipeline execution worker
    storage/
      __init__.py
      local.py           # local file serving
      s3.py              # S3 upload stub (TODO comments)
    Dockerfile
    requirements.txt
    .env.example

STEP 2 — Implement these four endpoints in main.py:

  POST /tryon
    - Accepts: multipart/form-data with fields:
        person_image (UploadFile), garment_image (UploadFile), quality (str, default="fast")
    - Validates: file size < 10MB, MIME type must be image/jpeg or image/png
    - Returns: {"job_id": str} immediately (HTTP 202)

  GET /result/{job_id}
    - Returns: {"status": "queued"|"processing"|"done"|"failed", "glb_url": str|null, "error": str|null}

  GET /health
    - Returns: {"gpu_memory_used_gb": float, "gpu_memory_total_gb": float, "queue_length": int, "model_status": dict}

  DELETE /job/{job_id}
    - Cancels a queued job, returns HTTP 204

STEP 3 — Async architecture:
  - Development: use FastAPI BackgroundTasks with asyncio.Lock for thread-safe job state.
  - Run pipeline in a ThreadPoolExecutor to avoid blocking the async event loop.
  - Add a clearly marked comment block explaining how to upgrade to Celery + Redis for production.

STEP 4 — File serving:
  - Serve completed .glb files via FastAPI StaticFiles at /outputs/.
  - Return full URL in glb_url field: http://{host}/outputs/{job_id}.glb
  - s3.py should contain a stubbed upload_to_s3(file_path, bucket) function with TODO comments.

STEP 5 — Dockerfile:
  - Base image: pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
  - Copy app code, install requirements.txt
  - Expose port 8000
  - CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

STEP 6 — Write test_api.sh:
  A bash script using curl that tests every endpoint:
  - POST /tryon with real image files
  - Poll GET /result/{job_id} until status == "done"
  - Verify .glb URL is downloadable
  - GET /health and print GPU stats
  - DELETE a job

SUCCESS CRITERIA:
  - API handles concurrent requests without deadlock or race conditions.
  - /health returns correct GPU memory figures.
  - End-to-end curl test produces a downloadable .glb URL.
  - Docker container builds cleanly with docker build.

OUTPUT: All files fully written — main.py, store.py, worker.py, local.py,
  s3.py, Dockerfile, requirements.txt, .env.example, test_api.sh. No truncation.
```

---

## PROMPT 07 — Flutter AR Frontend (6 Screens + ARKit/ARCore)

```
TASK IDENTITY: You are an expert Flutter developer. This is Task 07 of 08.

OBJECTIVE: Build a complete, production-quality Flutter mobile app with 6 screens,
  world-anchored AR .glb viewing, and full integration with the FastAPI backend.

PLATFORM: Flutter 3.22+. Targets: iOS (ARKit) and Android (ARCore).

---

STEP 1 — Project bootstrap:
  Run: flutter create arvton_app --org com.retailiq --template app

  Add to pubspec.yaml dependencies:
    ar_flutter_plugin: ^0.7.3
    model_viewer_plus: ^1.7.0
    camera: ^0.10.5
    image_picker: ^1.0.7
    dio: ^5.4.3
    riverpod: ^2.5.1
    flutter_riverpod: ^2.5.1
    go_router: ^13.2.0
    cached_network_image: ^3.3.1
    share_plus: ^9.0.0
    flutter_dotenv: ^5.1.0

  iOS Info.plist — add keys:
    NSCameraUsageDescription: "Required for AR try-on"
    NSPhotoLibraryUsageDescription: "Select your photo for try-on"
    NSPhotoLibraryAddUsageDescription: "Save your try-on result"

  Android AndroidManifest.xml — add:
    <uses-permission android:name="android.permission.CAMERA"/>
    <uses-feature android:name="android.hardware.camera.ar" android:required="true"/>

STEP 2 — State management with Riverpod:
  Create lib/providers/tryon_provider.dart:
    TryonState: { personImage, garmentImage, jobId, status, glbUrl, errorMessage }
    TryonNotifier extends AsyncNotifier<TryonState>
    Methods: selectPerson(), selectGarment(), submitTryon(), pollResult()

  Create lib/services/api_service.dart:
    - Dio client reading BACKEND_URL from .env
    - postTryon(File person, File garment, String quality) -> String jobId
    - getResult(String jobId) -> Map<String, dynamic>

STEP 3 — Implement all 6 screens:

  Screen 1 — SplashScreen (lib/screens/splash_screen.dart)
    - Animated logo fade-in over 1.5 seconds
    - Check internet connectivity
    - Navigate to CameraScreen after animation completes

  Screen 2 — CameraScreen (lib/screens/camera_screen.dart)
    - Full-screen live camera preview using camera package
    - Circular capture button at bottom center
    - Gallery button (bottom left) using image_picker
    - On image captured/selected: validate it contains a person (aspect ratio check),
      update TryonNotifier.selectPerson(), navigate to GarmentPickerScreen

  Screen 3 — GarmentPickerScreen (lib/screens/garment_picker_screen.dart)
    - 2-column grid of garment cards from a hardcoded JSON list (minimum 12 items)
    - Each card: garment image (CachedNetworkImage), name, price
    - Selected garment shows a blue border highlight
    - Floating "Try On" button appears when a garment is selected
    - On tap: call apiService.postTryon(), navigate to ProcessingScreen

  Screen 4 — ProcessingScreen (lib/screens/processing_screen.dart)
    - Full-screen with animated circular progress indicator
    - Status text updates: "Segmenting..." → "Generating try-on..." → "Building 3D model..." → "Almost done..."
    - Poll GET /result/{jobId} every 2 seconds using Timer.periodic
    - On status=="done": navigate to ARViewerScreen
    - On status=="failed": show error dialog with Retry and Cancel buttons

  Screen 5 — ARViewerScreen (lib/screens/ar_viewer_screen.dart)
    - ArFlutterPlugin.buildArView as the main widget
    - On first horizontal plane detected: load .glb from glbUrl and anchor to floor
    - Scale slider (0.5x to 2.0x) — vertical slider on right edge
    - "Rotate" FAB — toggles 0.5 degrees/frame Y-axis auto-rotation
    - "View in 3D" button — fallback to ModelViewerScreen (non-AR)
    - "Retry" button if AR initialisation fails after 10 seconds

  Screen 6 — ShareScreen (lib/screens/share_screen.dart)
    - Screenshot of AR view captured using RepaintBoundary
    - Share button using share_plus
    - Save to gallery button
    - "Try Another" button returns to GarmentPickerScreen

STEP 4 — Error handling:
  - Network errors: show SnackBar with "Retry" option, exponential backoff (1s, 2s, 4s, max 3 retries)
  - AR not supported (device check): graceful automatic fallback to model_viewer_plus
  - Job failed: dedicated error screen with retry and contact support buttons
  - All errors logged with error code and timestamp

STEP 5 — Write README.md including:
  - Prerequisites and Flutter version requirement
  - How to set BACKEND_URL in .env
  - Build commands for iOS and Android
  - How to run against local FastAPI (ngrok or LAN IP instructions)
  - Known limitations on Android ARCore

SUCCESS CRITERIA:
  - App builds without errors on both iOS and Android.
  - AR plane detection successfully anchors the .glb to a floor surface.
  - Scale slider smoothly adjusts avatar size.
  - All 6 screens navigate correctly in both happy path and error scenarios.

OUTPUT: Complete Flutter project — every Dart file fully written with no placeholder
  TODO comments in functional code. Include README.md. No truncation.
```

---

## PROMPT 08 — Integration Testing & Production Hardening

```
TASK IDENTITY: You are a senior ML systems engineer. This is Task 08 of 08 — the final
  integration, performance profiling, and production hardening task.

OBJECTIVE: Write a complete integration test suite, profile the pipeline performance,
  harden the FastAPI backend for production, and produce a deployment runbook.

PREREQUISITES: All 7 prior tasks completed.
  - FastAPI backend deployed and accessible at BACKEND_URL (set via .env)
  - Flutter app built and running on at least one device

---

STEP 1 — Integration test suite using pytest:
  Write these 5 test files in /tests/:

  test_segmentation.py
    - Assert output is RGBA (4 channels)
    - Assert all background pixels have alpha == 0
    - Assert foreground pixel count > 1000
    - Run on 10 different garment images and report pass/fail per image

  test_tryon.py
    - Assert output image shape is 768x1024 (Leffa) or 512x768 (CatVTON)
    - Compute SSIM against a reference output — assert SSIM > 0.70
    - Assert no pixel has all-zero RGB values in the garment region

  test_reconstruction.py
    - Assert output .obj is watertight (trimesh.is_watertight)
    - Assert vertex count > 10,000
    - Assert UV coordinates are present (mesh.visual is TextureVisuals)

  test_export.py
    - Assert .glb file size < 15MB
    - Assert trimesh.load(glb_path) succeeds without exception
    - Assert mesh has at least one material with a texture

  test_api.py
    - Assert POST /tryon returns HTTP 202 with a job_id
    - Assert GET /result/{job_id} transitions through queued → processing → done
    - Assert entire job completes in under 60 seconds
    - Assert GET /health returns gpu_memory_used_gb as a float
    - Assert DELETE /job/{job_id} returns HTTP 204
    - Assert POST /tryon rejects files > 10MB with HTTP 413
    - Assert POST /tryon rejects non-image files with HTTP 422

STEP 2 — Performance profiling:
  - Use torch.profiler to measure GPU kernel time at each pipeline stage.
  - Run profiling on 5 different input pairs and compute average + std deviation.
  - Print this table:
      Stage          | Avg latency (ms) | Std Dev (ms) | VRAM peak (GB)
      ---------------+------------------+--------------+---------------
      Segmentation   |                  |              |
      2D Try-On      |                  |              |
      3D Reconstruct |                  |              |
      .glb Export    |                  |              |
      Total          |                  |              |
  - Identify the slowest stage.
  - Propose and implement ONE concrete optimization for it
    (e.g., model warm-up caching, resolution reduction, batch norm fusion).

STEP 3 — Production hardening — implement ALL of the following in main.py:

  Rate limiting:
    pip install slowapi
    Limit: 10 requests per minute per API key
    Return HTTP 429 with Retry-After header when exceeded

  Input validation:
    - Reject images > 10MB → HTTP 413
    - Reject non-image MIME types → HTTP 422
    - Reject images with no detected human body
      (use HMR 2.0 confidence score < 0.5 as the rejection threshold → HTTP 422)

  CORS configuration:
    - Allow only specific origins: your Flutter app domain + localhost for dev

  Structured JSON logging:
    Every log line must be valid JSON with these fields:
    { "timestamp": ISO8601, "job_id": str, "stage": str, "latency_ms": float, "level": str, "error": str|null }

  Health monitoring:
    /health endpoint returns:
    - gpu_memory_used_gb
    - gpu_memory_total_gb
    - queue_length
    - last_5_jobs: array of { job_id, status, duration_ms }

STEP 4 — Write DEPLOY.md — a complete deployment runbook covering:

  Section 1: Deploy to AWS EC2 (G4dn.xlarge, T4 GPU)
    - AMI selection, security group setup
    - Docker pull and run commands
    - Environment variables and .env setup

  Section 2: Deploy to GCP (n1-standard-4 + T4 GPU)
    - VM creation command (gcloud)
    - Docker deployment commands

  Section 3: Point Flutter app to production
    - How to update BACKEND_URL in .env
    - How to rebuild and distribute the app

  Section 4: Run the full test suite against production
    - Set BACKEND_URL, run pytest /tests/ -v
    - Expected output format

  Section 5: Rollback procedure
    - How to revert to previous Docker image tag
    - How to restore from a training checkpoint

SUCCESS CRITERIA:
  - All 5 test files pass with 0 failures on a working deployment.
  - Performance table is printed with real measured values.
  - Rate limiting returns HTTP 429 correctly at 11th request/minute.
  - DEPLOY.md is complete enough for a new engineer with no prior context to follow.

OUTPUT: All test files, profiling script, updated main.py with hardening,
  and DEPLOY.md — fully written with no truncation.
```

---

## Quick Reference — Pipeline Summary

| Prompt | Task | Key Models / Tools | Platform |
|--------|------|-------------------|----------|
| 01 | Segmentation | SAM 2 hiera-large | Colab T4/A100 |
| 02 | 2D Try-On | Leffa (quality) / CatVTON (fast) | Colab T4/A100 |
| 03 | 3D Body Mesh | HMR 2.0 + ECON + SMPLitex | Colab A100 / AMD |
| 04 | .glb Export | TripoSR (fast) / SyncHuman (quality) | AMD MI250/MI300 |
| 05 | Fine-Tuning | CatVTON attention fine-tune | AMD MI250/MI300 |
| 06 | Backend | FastAPI + Redis + Docker | Cloud GPU |
| 07 | Frontend | Flutter + ARKit/ARCore | Mobile |
| 08 | Hardening | pytest + slowapi + profiler | All |

---

## Execution Tips for Claude Opus 4.6

- Start each prompt in a **fresh conversation** — do not chain them in one thread
- If the agent stops mid-code, reply: `"Continue from where you left off. Do not repeat previous code."`
- If a model checkpoint URL is outdated, reply: `"Search Hugging Face for the current model card and update the code accordingly."`
- For AMD-specific issues, reply: `"Use the ROCm equivalent. Replace any CUDA-only library calls with their ROCm or HIP counterparts."`
- After receiving code, always ask: `"Now write the unit test for the function you just wrote."`