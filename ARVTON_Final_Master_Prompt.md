# ARVTON — Final Master Prompt
## Claude Opus 4.6 · Complete AR/VR Virtual Try-On Pipeline
> This is the single, definitive prompt. Copy everything inside the code block and paste it into a fresh Claude Opus 4.6 conversation. It covers the entire project from dataset construction to deployed Flutter AR app.

---

```
╔══════════════════════════════════════════════════════════════════════╗
║          ARVTON — AR/VR VIRTUAL TRY-ON PIPELINE                     ║
║          MASTER BUILD PROMPT FOR CLAUDE OPUS 4.6                    ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are a senior ML systems engineer and full-stack developer building
a production-grade, commercially licensed AR/VR Virtual Try-On pipeline
called ARVTON. You will build this project end-to-end: dataset
construction, model training, backend API, and Flutter AR mobile app.

You have no pre-existing dataset or retail catalog.
Every dataset and model checkpoint you use MUST be commercially licensed
(MIT, Apache 2.0, CC BY 4.0, CC0, or equivalent).
NEVER use VITON-HD, DressCode, or any CC BY-NC dataset.
NEVER load any checkpoint pre-trained on non-commercial data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLATFORMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Development and Inference  →  Google Colab (T4 free / A100 Pro+)
Training and Heavy Compute →  AMD ROCm platform (MI250 or MI300)
Mobile Frontend            →  Flutter 3.22+ (iOS ARKit + Android ARCore)
Backend Deployment         →  Docker on cloud GPU (AWS G4dn / GCP T4)

All code must be platform-aware:
- Detect CUDA vs ROCm automatically using torch.version.hip
- Use bf16 on A100/MI250, fp16 on T4, fp32 only when required
- All checkpoints save to Google Drive on Colab,
  to /mnt/storage/arvton_ckpts/ on AMD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT: User photo + Garment image
  ↓
[Stage 0]  SAM 2 (sam2-hiera-large, Apache 2.0)
           → Clean RGBA segmentation masks
  ↓
[Stage 1]  Leffa (franciszzj/Leffa) → quality path (A100/MI250)
           CatVTON (zhengchong/CatVTON) → fast fallback (T4)
           → Photorealistic 2D try-on composite at 768x1024
  ↓
[Stage 2]  HMR 2.0 (gzb1-lab/4D-Humans) → SMPL body parameters
           ECON → clothed surface geometry
           SMPLitex → high-res texture mapping
           → Clothed .obj mesh with UV coordinates
  ↓
[Stage 3]  TripoSR → fast path (.glb in under 15s, T4 compatible)
           SyncHuman → quality path (.glb, A100/MI250 required)
           → Textured .glb file under 15MB
  ↓
[Stage 4]  FastAPI backend → async REST API with job queue
           Docker container → cloud GPU deployment
  ↓
[Stage 5]  Flutter AR app → world-anchored AR viewer
           ARKit (iOS) + ARCore (Android) floor plane detection


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 0 — COMMERCIAL DATASET CONSTRUCTION (DO THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have no existing catalog. Build a commercially clean dataset of
3,000+ training pairs using three parallel sources.

SOURCE A — ViViD Dataset (Primary, Apache 2.0)
  Repo: https://github.com/Zheng-Chong/ViViD
  Contains: 9,700 paired garment and video try-on sequences, 1.2M frames.
  Covers: upper-body, lower-body, dresses.

  Write a Python script vivid_prepare.py that:
  1. Downloads and extracts ViViD to
     /content/drive/MyDrive/arvton/datasets/vivid/
  2. Samples 1 representative frame per video — the sharpest frame
     detected via Laplacian variance score
  3. Runs SAM 2 on each sampled frame to generate person_mask
  4. Runs HMR 2.0 to generate body_parse and densepose maps
  5. Saves each training example as a JSON metadata record:
       {
         "id": "vivid_00001",
         "person_image": "path/to/person.jpg",
         "person_mask": "path/to/person_mask.png",
         "body_parse": "path/to/body_parse.png",
         "densepose": "path/to/densepose.png",
         "garment_image": "path/to/garment.jpg",
         "garment_mask": "path/to/garment_mask.png",
         "category": "upper or lower or dress",
         "license": "Apache-2.0",
         "source": "ViViD"
       }
  6. Writes train_manifest.json (80%) and val_manifest.json (20%)
     to /content/drive/MyDrive/arvton/datasets/

SOURCE B — CC0 Web Scraper (Supplement)
  Licensed sources: Pexels API (CC0), Pixabay API (CC0)
  Target: 2,000 additional pairs

  Write a script cc0_scraper.py that:
  1. Uses the Pexels API (free key from pexels.com/api/) to search:
     ["t-shirt flat lay", "dress flat lay", "jeans flat lay",
      "fashion model full body", "person standing white background",
      "streetwear model", "formal wear model"]
  2. Uses the Pixabay API (free key from pixabay.com/api/docs/)
     for the same queries
  3. Downloads images and classifies each as "person" or "garment"
     using CLIP (openai/clip-vit-base-patch32, MIT license):
       Prompt A: "a person wearing clothes, full body photo"
       Prompt B: "a flat-lay garment product photo, white background"
       Discard images where CLIP confidence is below 0.75
  4. Runs the same SAM 2 and HMR 2.0 labeling as SOURCE A
  5. Randomly pairs person images with garment images (unpaired mode)
  6. Runs CatVTON inference on each pair to generate synthetic
     try-on ground truth — this is the training label
  7. Appends entries to train_manifest.json with "source": "CC0"

  Save all raw downloads to:
    /content/drive/MyDrive/arvton/datasets/cc0/raw/
  Save all processed pairs to:
    /content/drive/MyDrive/arvton/datasets/cc0/processed/

SOURCE C — Synthetic Generation (Diversity Boost)
  Use SDXL and ControlNet to generate diverse synthetic garments
  and person images.

  Write a script synthetic_gen.py that:
  1. Generates 500 garment flat-lay images using SDXL:
     Prompt template: "flat-lay photograph of a {COLOR} {MATERIAL}
     {GARMENT_TYPE}, white background, product photography, studio
     lighting, ultra sharp, no mannequin"
     COLOR: black, white, navy, red, olive, beige, grey,
            forest green, burgundy, mustard
     MATERIAL: cotton, linen, denim, silk, wool, polyester,
               jersey, tweed, canvas
     GARMENT_TYPE: t-shirt, button-down shirt, hoodie, blazer,
                   dress, trousers, jeans, skirt

  2. Generates 500 person images using SDXL + ControlNet-OpenPose:
     Sample pose skeletons from AMASS dataset (MIT license)
     Prompt template: "full body photo of a person standing,
     {BODY_TYPE} body, {ETHNICITY}, neutral background, fashion
     photography, sharp focus, natural lighting"
     BODY_TYPE: slim, athletic, average, plus-size, tall
     ETHNICITY: diverse representation across all generations

  3. Applies the same SAM 2 and HMR 2.0 auto-labeling pipeline
  4. Appends to train_manifest.json with "source": "synthetic"

DATASET VALIDATION
  Write validate_dataset.py that checks the final manifest and prints:
    Total pairs:          [count]
    By source:            ViViD=[n]  CC0=[n]  Synthetic=[n]
    By category:          upper=[n]  lower=[n]  dress=[n]
    Missing files:        [list any broken paths]
    License compliance:   PASS or FAIL (flag any non-commercial sources)
    Ready for training:   YES or NO

  Target: 3,000+ pairs before proceeding to Phase 1.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — SEGMENTATION MODULE (segment.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checkpoint: facebook/sam2-hiera-large (Apache 2.0)
Save to: /content/drive/MyDrive/arvton/checkpoints/sam2/

Write segment.py with these functions:

  segment_garment(image_path: str) -> PIL.Image
    Removes background from a garment flat-lay.
    Returns RGBA PNG with clean edges.
    Handles: white backgrounds, gradient backgrounds, shadows.

  segment_person(image_path: str) -> PIL.Image
    Isolates human body from background.
    Returns RGBA PNG.
    Handles: complex backgrounds, partial occlusion,
    multiple people — keep the largest or most centered person.

  batch_segment(image_dir: str, output_dir: str, mode: str)
    mode = "garment" or "person"
    Processes entire directories in batches of 8.
    Skips already-processed images by checking output_dir.
    Prints progress: "Processed 45/500 images"

Rules:
  TRAINING PATH: skip SAM 2, load masks directly from dataset manifest.
  INFERENCE PATH: always run SAM 2 on new user-uploaded images.
  VRAM usage must stay under 10GB.
  Add try/except around every SAM 2 call with meaningful error messages.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — 2D VIRTUAL TRY-ON MODULE (tryon.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two execution paths based on available VRAM:

QUALITY PATH — Leffa (franciszzj/Leffa)
  Use when: VRAM >= 20GB (A100, MI250)
  Precision: float32 on A100, bfloat16 on MI250
  Output resolution: 768x1024
  Expected latency: 4 to 8 seconds on A100

FAST PATH — CatVTON (zhengchong/CatVTON)
  Use when: VRAM < 20GB (T4, RTX)
  Enable: attention_slicing, VAE tiling
  Precision: float16
  Output resolution: 512x768
  Expected latency: 6 to 12 seconds on T4

Write tryon.py with:

  detect_optimal_path() -> str
    Queries torch.cuda.get_device_properties(0).total_memory
    Returns "quality" or "fast"
    Prints which path was selected and why

  load_model(path: str = "auto") -> model
    path="auto" calls detect_optimal_path()
    Loads model once and caches in module-level variable
    Subsequent calls reuse cached model — no reload overhead

  run_tryon(person_rgba: PIL.Image,
            garment_rgba: PIL.Image,
            quality: str = "auto") -> PIL.Image
    quality = "auto" or "quality" or "fast"
    Validates input image sizes before inference
    Returns composite PIL.Image
    Logs: path used, resolution, latency_ms, vram_peak_gb

  run_tryon_batch(pairs: list, quality: str = "auto") -> list
    Accepts list of (person_rgba, garment_rgba) tuples
    Processes in batches for efficiency
    Returns list of composite PIL.Images


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — 3D BODY RECONSTRUCTION MODULE (reconstruct3d.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three sequential sub-steps:

SUB-STEP A — HMR 2.0 (gzb1-lab/4D-Humans, MIT license)
  Install: pip install 4D-Humans
  Checkpoint: hmr2-0b

  estimate_body(composite_image: PIL.Image) -> dict
    Returns:
      theta: np.ndarray (72,)    SMPL pose parameters
      beta: np.ndarray (10,)     SMPL shape parameters
      camera: np.ndarray (3,)    weak-perspective camera
      confidence: float          detection confidence 0.0 to 1.0
      bbox: list                 [x, y, w, h] of detected person
    Reject images where confidence < 0.5 — return None, log warning

SUB-STEP B — ECON (clothed surface geometry)
  Clone from GitHub, install dependencies.
  Provide ROCm equivalents for any CUDA-only calls as inline comments.

  reconstruct_clothed(composite: PIL.Image, smpl_params: dict) -> str
    Runs: normal estimation → IF-Net → SMPL-D fitting
    Returns: path to output .obj file
    Saves to: /content/drive/MyDrive/arvton/outputs/{job_id}_mesh.obj
    Logs progress at each sub-step

SUB-STEP C — SMPLitex (high-res texture)
  apply_texture(obj_path: str, composite: PIL.Image) -> str
    UV-unwraps the mesh using xatlas
    Projects garment texture from composite image onto mesh
    Returns: path to textured .obj

MASTER FUNCTION:
  build_clothed_mesh(composite_image: PIL.Image,
                     job_id: str) -> str or None
    Chains A → B → C
    Returns .obj path on success, None on failure
    Full error handling with specific messages per sub-step
    Validates output: assert vertex_count > 10,000
    Validates output: assert UV coordinates present
    Logs total wall time


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — .GLB EXPORT MODULE (export3d.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two execution paths:

FAST PATH — TripoSR (VAST-AI-Research/TripoSR, MIT license)
  Install: pip install tsr
  Use when: T4, quick preview needed, latency target under 15 seconds
  Input: composite PIL.Image directly (single image to 3D)

  export_glb_fast(composite: PIL.Image, job_id: str) -> str
    Runs TripoSR inference
    Applies composite image as texture
    Applies Draco compression (target under 15MB)
    Returns: path to .glb file
    Validates with trimesh before returning

QUALITY PATH — SyncHuman
  Use when: A100 or MI250, high-fidelity output needed
  Expected latency: 90 to 180 seconds on A100
  Document ROCm installation steps as inline comments

  export_glb_quality(obj_path: str,
                     composite: PIL.Image,
                     job_id: str) -> str
    Runs SyncHuman multiview generation
    Returns: path to .glb file
    Falls back to TripoSR automatically if SyncHuman fails

MASTER FUNCTION:
  export_to_glb(composite: PIL.Image,
                obj_path: str = None,
                quality: str = "auto",
                job_id: str = None) -> str or None
    "auto" uses VRAM detection to choose path
    Validates output: file size under 15MB
    Validates output: trimesh.load() succeeds without errors
    Returns .glb path or None on failure

INLINE NOTEBOOK VIEWER:
  Write show_glb_in_notebook(glb_path: str):
    Generates an HTML cell using three.js CDN
    Renders the .glb with orbit controls in Colab output
    No external hosting required, works offline


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 5 — FINE-TUNING (finetune.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Platform: AMD ROCm (MI250 or MI300)
PyTorch install:
  pip install torch torchvision --index-url
  https://download.pytorch.org/whl/rocm6.1

Base model: DCI-VTON (MIT license) — code only, train weights from
  scratch on commercial data. DO NOT load any checkpoint pre-trained
  on VITON-HD or DressCode.

DATASET CLASS
  Write ARVTONDataset(torch.utils.data.Dataset):
    Loads from train_manifest.json built in Phase 0
    Returns 6-item dict per sample:
      person_image:  tensor [3, 1024, 768] normalized to [-1, 1]
      person_mask:   tensor [1, 1024, 768] binary float
      body_parse:    tensor [3, 1024, 768]
      densepose:     tensor [3, 1024, 768]
      garment_image: tensor [3, 1024, 768] normalized to [-1, 1]
      garment_mask:  tensor [1, 1024, 768] binary float

    Augmentations (training only, NEVER on validation):
      Random horizontal flip p=0.5 applied identically to all 6 tensors
      Color jitter on person_image and garment_image ONLY:
        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
      NO random crop — images are pre-sized at 1024x768

    mode = "paired" (same person and garment index) or
           "unpaired" (random garment, different index)
    Use paired=80%, unpaired=20% per epoch

TRAINING CONFIGURATION
  Optimizer:   AdamW, lr=1e-5, weight_decay=0.01
  LR Schedule: Cosine decay with 200-step linear warmup
  Precision:   bf16 on MI250 (torch.cuda.is_bf16_supported() check)
               fp16 fallback if bf16 unavailable
  Batch size:  16 on MI250 or 32 on MI300
  DataLoader:  num_workers=8, pin_memory=True, prefetch_factor=2
  Epochs:      5 or 5000 steps, whichever comes first

  Fine-tuning strategy:
    FREEZE all parameters EXCEPT layers with "attn" in their name
    Print trainable vs frozen parameter counts at startup

TRAINING LOOP must include:
  1. ROCm verification at startup:
       assert torch.cuda.is_available(), "ROCm not detected"
       print(f"GPU: {torch.cuda.get_device_name(0)}")

  2. Checkpoint save every 500 steps:
       /mnt/storage/arvton_ckpts/step_{N:06d}.pt
       Save: step, model_state_dict, optimizer_state_dict, loss

  3. Resume-from-checkpoint at startup:
       Find latest checkpoint in /mnt/storage/arvton_ckpts/
       Load and resume training seamlessly

  4. Logging every 50 steps to console (structured JSON):
       {"step": N, "loss": 0.0423, "lr": 1e-5, "vram_gb": 48.2}

  5. TensorBoard logging: loss, lr, sample images every 1000 steps

  6. Sample image saves every 1000 steps:
       Run validation on 4 fixed pairs
       Save grid: [person | garment | prediction | ground_truth]
       to /mnt/storage/arvton_samples/step_{N:06d}.png

EVALUATION after training completes:
  Compute SSIM (skimage.metrics) and LPIPS (lpips, MIT license)
  Print comparison table:
    Metric    | Base Model | Fine-tuned | Delta
    ----------+------------+------------+-------
    SSIM      |            |            |
    LPIPS     |            |            |
  Save 10 side-by-side comparison images to /mnt/storage/arvton_eval/

Launch command (write at bottom of finetune.py):
  torchrun --nproc_per_node=1 finetune.py \
    --manifest /mnt/storage/arvton/datasets/train_manifest.json \
    --output_dir /mnt/storage/arvton_ckpts/ \
    --batch_size 16 \
    --steps 5000 \
    --precision bf16


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 6 — FASTAPI BACKEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a production-ready FastAPI application.

PROJECT STRUCTURE
  /app
    main.py              FastAPI entry point and all endpoints
    pipeline/
      __init__.py
      segment.py         from Phase 1
      tryon.py           from Phase 2
      reconstruct3d.py   from Phase 3
      export3d.py        from Phase 4
    jobs/
      __init__.py
      store.py           thread-safe job state with asyncio.Lock
      worker.py          runs pipeline in ThreadPoolExecutor
    storage/
      __init__.py
      local.py           serve files via StaticFiles
      s3.py              stubbed S3 upload with TODO comments
    Dockerfile
    requirements.txt
    .env.example

ENDPOINTS

  POST /tryon
    Input:  multipart/form-data
              person_image: UploadFile (JPEG/PNG, max 10MB)
              garment_image: UploadFile (JPEG/PNG, max 10MB)
              quality: str = "auto"
    Logic:  Validate MIME type → validate file size →
            validate person detected (HMR 2.0 confidence >= 0.5,
            else return 422 "No person detected in image") →
            enqueue job → return immediately
    Output: HTTP 202, {"job_id": "uuid4-string"}

  GET /result/{job_id}
    Output: {
      "status": "queued" or "processing" or "done" or "failed",
      "glb_url": "http://host/outputs/{job_id}.glb" or null,
      "progress": stage description or null,
      "error": "description" or null,
      "duration_ms": int or null
    }

  GET /health
    Output: {
      "status": "ok",
      "gpu_name": str,
      "gpu_memory_used_gb": float,
      "gpu_memory_total_gb": float,
      "queue_length": int,
      "models_loaded": {"sam2": bool, "tryon": bool, "hmr2": bool},
      "last_5_jobs": [{"job_id": str, "status": str, "duration_ms": int}]
    }

  DELETE /job/{job_id}
    Output: HTTP 204

PRODUCTION HARDENING

  Rate limiting (pip install slowapi):
    10 requests/minute per API key
    Return HTTP 429 with Retry-After header when exceeded

  CORS:
    Allow only: ["http://localhost:*", "https://yourdomain.com"]
    Configurable via ALLOWED_ORIGINS env var

  Structured JSON logging per request:
    {
      "timestamp": "ISO8601",
      "job_id": "uuid",
      "stage": "segmentation or tryon or reconstruct or export",
      "latency_ms": 1240,
      "level": "INFO or ERROR",
      "error": null
    }

  Model warm-up on startup:
    Load all 4 models (SAM2, tryon, HMR2, TripoSR) into GPU memory
    during FastAPI lifespan startup event
    Log: "All models loaded. Ready to accept requests."

DOCKERFILE
  FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt --no-cache-dir
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0",
       "--port", "8000", "--workers", "1"]

TEST SCRIPT (test_api.sh)
  Write a bash script using curl that:
  1. POST /tryon with real test images → capture job_id
  2. Poll GET /result/{job_id} every 2s until status=="done" (timeout 120s)
  3. Download the .glb URL and verify it is a valid binary file
  4. GET /health and print GPU stats
  5. POST /tryon with >10MB file → assert HTTP 413
  6. POST /tryon with a .pdf file → assert HTTP 422
  7. DELETE the job → assert HTTP 204
  Print PASS/FAIL for each test.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 7 — FLUTTER AR FRONTEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
flutter create arvton_app --org com.retailiq --template app

PUBSPEC.YAML DEPENDENCIES
  ar_flutter_plugin: ^0.7.3
  model_viewer_plus: ^1.7.0
  camera: ^0.10.5
  image_picker: ^1.0.7
  dio: ^5.4.3
  flutter_riverpod: ^2.5.1
  go_router: ^13.2.0
  cached_network_image: ^3.3.1
  share_plus: ^9.0.0
  flutter_dotenv: ^5.1.0
  lottie: ^3.1.0

PERMISSIONS
  iOS Info.plist:
    NSCameraUsageDescription: "Required for virtual try-on AR view"
    NSPhotoLibraryUsageDescription: "Select your photo to try on clothes"
    NSPhotoLibraryAddUsageDescription: "Save your try-on result"

  Android AndroidManifest.xml:
    <uses-permission android:name="android.permission.CAMERA"/>
    <uses-feature android:name="android.hardware.camera.ar"
                  android:required="true"/>

STATE MANAGEMENT (Riverpod)
  lib/providers/tryon_provider.dart
    TryonState fields:
      personImage, garmentImage, selectedGarmentId, jobId,
      status (idle/uploading/queued/processing/done/failed),
      progress, glbUrl, errorMessage

    TryonNotifier methods:
      selectPerson(File image)
      selectGarment(String garmentId, File image)
      submitTryon(String quality)   calls POST /tryon
      pollResult()                  calls GET /result, updates state
      reset()

  lib/services/api_service.dart
    Dio client reading BACKEND_URL from .env
    postTryon(File person, File garment, String quality) -> String jobId
    getResult(String jobId) -> TryonResult
    Retry on 5xx: exponential backoff 1s, 2s, 4s, max 3 attempts

6 SCREENS — fully implement all of them

  SCREEN 1 — SplashScreen
  lib/screens/splash_screen.dart
    Lottie animation logo
    Check internet with dio HEAD request to BACKEND_URL/health
    Navigate to CameraScreen after 2.0 seconds
    If backend unreachable: show "Backend offline" SnackBar,
    allow continue in demo mode (mock .glb URL)

  SCREEN 2 — CameraScreen
  lib/screens/camera_screen.dart
    Full-screen live camera preview (camera package)
    Shutter button (bottom center): capture person photo
    Gallery button (bottom left): image_picker from gallery
    Flip camera button (top right)
    After capture: validate aspect ratio is portrait (1:1.2 to 1:2)
    If landscape: show dialog "Please use a portrait photo"
    Update TryonNotifier.selectPerson()
    Navigate to GarmentPickerScreen

  SCREEN 3 — GarmentPickerScreen
  lib/screens/garment_picker_screen.dart
    Header: small circular person photo thumbnail from state
    2-column grid using GridView.builder
    Hardcode a JSON garment catalog of 16 items minimum:
      Each item: { "id", "name", "price", "category", "image_url" }
      Use placeholder URLs from picsum.photos
      Categories: T-Shirts, Shirts, Dresses, Trousers
    Category filter tabs at top: All / T-Shirts / Shirts / Dresses / Trousers
    Each garment card: image (CachedNetworkImage), name, price
    Selected card: 3px accent blue border + checkmark overlay
    Floating bottom bar when garment is selected:
      Shows garment thumbnail and "Try On" button
      Tap: calls submitTryon("auto"), navigates to ProcessingScreen

  SCREEN 4 — ProcessingScreen
  lib/screens/processing_screen.dart
    Full-screen dark background
    Lottie loading animation
    Dynamic status text based on TryonState.progress:
      "queued"       → "Getting in line..."
      "Segmenting"   → "Removing backgrounds..."
      "Generating"   → "Dressing you up..."
      "Building 3D"  → "Shaping your 3D avatar..."
      "Exporting"    → "Almost ready..."
    Poll GET /result/{jobId} every 2 seconds using Timer.periodic
    On status=="done": navigate to ARViewerScreen (replace, not push)
    On status=="failed": full-screen error with error message,
      "Try Again" button, and "Go Back" button
    Cancel button (top left X): DELETE /job/{jobId}, go back to Garment

  SCREEN 5 — ARViewerScreen
  lib/screens/ar_viewer_screen.dart
    ArFlutterPlugin.buildArView() fills entire screen

    PHASE A — Scanning:
      Pulsing floor scan animation overlay
      "Move your phone slowly to detect the floor" instruction text
      Hide overlay once plane detected

    PHASE B — Avatar shown:
      On first horizontal plane detected: place .glb at floor position
      Avatar appears with scale-in animation (0.0 to 1.0 over 0.5s)

    UI Controls (semi-transparent overlay):
      Scale slider (right edge, vertical): 0.5x to 2.0x
        Adjusts node scale in real time
      Rotate FAB (bottom right): toggles 0.5 degrees per frame Y-axis rotation
      "View in 3D" button (bottom left): push ModelViewerFallbackScreen
      "Share" button (top right): navigate to ShareScreen

    Fallback timeout: if no plane detected after 12 seconds,
      show banner "AR not working? View in 3D instead"
      Tap banner → push ModelViewerFallbackScreen automatically

    ModelViewerFallbackScreen: model_viewer_plus widget in a non-AR
      360-degree viewer with same scale controls

  SCREEN 6 — ShareScreen
  lib/screens/share_screen.dart
    Capture AR screenshot using RepaintBoundary + RenderObject.toImage
    Display screenshot preview full screen
    Bottom action bar:
      Share (share_plus) — shares image + "Try ARVTON" text
      Save to Gallery (image_gallery_saver_plus)
      Try Another → pop back to GarmentPickerScreen (keep person photo)
      New Photo → pop back to CameraScreen

README.md — write a complete README including:
  Prerequisites: Flutter 3.22+, Dart 3.4+, Xcode 15+, Android Studio
  .env setup: BACKEND_URL and API_KEY
  Build commands for iOS and Android
  How to point app at local FastAPI via ngrok tunnel
  How to point app at production URL
  Troubleshooting AR not working on Android
  Known limitations


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 8 — INTEGRATION TESTING AND DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PYTEST INTEGRATION TESTS in /tests/

  tests/test_segmentation.py
    Assert output mode == "RGBA"
    Assert alpha channel has zeros (background removed)
    Assert foreground pixel count > 1000
    Run on 10 different test images, report per-image pass/fail

  tests/test_tryon.py
    Assert output size == (768, 1024) for Leffa or (512, 768) for CatVTON
    Compute SSIM vs reference image — assert > 0.70
    Assert no fully-black pixels in garment region

  tests/test_reconstruction.py
    Assert .obj is watertight (trimesh.is_watertight)
    Assert vertex_count > 10,000
    Assert UV coordinates present

  tests/test_export.py
    Assert .glb size < 15MB
    Assert trimesh.load(path) raises no exception
    Assert at least one material with texture in the scene

  tests/test_api.py
    Assert POST /tryon returns HTTP 202 and job_id
    Assert status transitions: queued → processing → done
    Assert full job completes in under 90 seconds
    Assert GET /health returns gpu_memory_used_gb as float
    Assert DELETE /job returns HTTP 204
    Assert POST /tryon with >10MB file returns HTTP 413
    Assert POST /tryon with .pdf returns HTTP 422
    Assert POST /tryon with no person detected returns HTTP 422

  Run all: pytest /tests/ -v --tb=short --timeout=120

PERFORMANCE PROFILING (profile_pipeline.py)
  Use torch.profiler over 5 input pairs.
  Print this exact table:

    Stage            | Avg (ms) | Std (ms) | VRAM Peak (GB)
    -----------------+----------+----------+---------------
    SAM2 Segment     |          |          |
    2D Try-On        |          |          |
    3D Reconstruct   |          |          |
    .glb Export      |          |          |
    API Overhead     |          |          |
    -----------------+----------+----------+---------------
    TOTAL            |          |          |

  Identify the slowest stage.
  Implement ONE concrete optimization:
    If SAM2 is slowest:        pre-compute and cache masks at upload time
    If Try-On is slowest:      reduce inference steps from 50 to 30
    If Reconstruct is slowest: cache HMR2 body params per user session
    If Export is slowest:      run TripoSR at half resolution then upsample

DEPLOY.md — write a complete runbook with five sections:

  Section 1: AWS EC2 G4dn.xlarge (T4 16GB, $0.526/hr)
    AMI selection, security group setup
    Exact commands to pull Docker image and run container
    How to set environment variables

  Section 2: GCP N1 + T4 GPU ($0.35/hr spot)
    gcloud compute instances create command
    Docker deployment commands

  Section 3: Update Flutter app to production
    Edit .env BACKEND_URL
    flutter build ios --release
    flutter build apk --release

  Section 4: Run test suite against production
    export BACKEND_URL=https://your-production-url
    pytest /tests/test_api.py -v

  Section 5: Rollback procedure
    Roll back Docker image: docker run previous-tag
    Roll back model: load step_{N}.pt checkpoint
    Emergency: disable GPU route, return HTTP 503


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIVERSAL RULES — APPLY TO EVERY FILE YOU WRITE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER truncate code. Write every file completely.

2. NEVER use placeholder comments like "# add logic here" or
   "# TODO: implement this". Every function must be fully implemented.

3. NEVER load a checkpoint trained on VITON-HD or DressCode.
   Every base model must be trained from scratch on commercial data.

4. ALL library installs on AMD must include --break-system-packages.

5. EVERY GPU call must be wrapped in try/except with a meaningful
   error message that includes stage name and VRAM usage at failure.

6. EVERY Python file must begin with this header block:
     # Module: [name]
     # License: MIT (ARVTON project)
     # Description: [one line]
     # Platform: Colab or AMD ROCm or Both
     # Dependencies: [list]

7. When writing for AMD ROCm, always add a comment on the first GPU
   call: # ROCm compatible: YES or NO or PARTIAL - reason

8. Output each Phase with this header:
     ===================================
     PHASE N — [NAME]
     File: [filename]
     ===================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTION ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write the phases in this exact order.
After each phase print:
  Phase N complete. Files written: [list]
  Reply "continue" to proceed to Phase N+1.

  Phase 0 → Dataset Construction
  Phase 1 → segment.py
  Phase 2 → tryon.py
  Phase 3 → reconstruct3d.py
  Phase 4 → export3d.py
  Phase 5 → finetune.py
  Phase 6 → FastAPI backend (all files)
  Phase 7 → Flutter app (all screens)
  Phase 8 → Tests, profiling, DEPLOY.md

Begin with Phase 0 now.
```

---

## How to Use

| Step | Action |
|---|---|
| **1** | Open a **fresh** Claude Opus 4.6 conversation |
| **2** | Copy **everything inside the code block** above |
| **3** | Paste it and send |
| **4** | Claude starts with Phase 0 — dataset construction |
| **5** | Reply `continue` after each phase to move forward |

## Recovery Commands

If Claude stops mid-code:
```
Continue from where you left off. Do not repeat any previous code.
```

If a Hugging Face URL is outdated:
```
Search Hugging Face for the current model card and update the code.
```

If an AMD/ROCm error occurs:
```
Use the ROCm equivalent. Add the original CUDA version as an inline comment.
```

If output is too short or truncated:
```
You truncated your response. Continue writing from the exact line where
you stopped. Do not summarise or skip any functions.
```

---

## What the Prompt Builds

| Phase | Deliverable | Platform | License |
|---|---|---|---|
| 0 | 3,000+ pair commercial dataset | Colab | Apache 2.0 / CC0 |
| 1 | SAM 2 segmentation module | Colab T4 / A100 | Apache 2.0 |
| 2 | Leffa + CatVTON try-on (auto path) | Colab T4 / A100 | MIT |
| 3 | HMR 2.0 + ECON 3D reconstruction | A100 / AMD | MIT |
| 4 | TripoSR + SyncHuman .glb export | AMD MI250/MI300 | MIT |
| 5 | Full fine-tuning with checkpoint resume | AMD MI250/MI300 | MIT |
| 6 | FastAPI + rate limiting + Docker | Cloud GPU | — |
| 7 | Flutter 6-screen AR app | iOS + Android | — |
| 8 | pytest suite + profiler + DEPLOY.md | All | — |
