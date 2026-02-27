# ARVTON — Deployment Runbook

## Section 1: AWS EC2 G4dn.xlarge (T4 16GB, ~$0.526/hr)

### AMI & Instance Setup
```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-0c94855ba95c71c99 \          # Deep Learning AMI (Ubuntu 22.04)
  --instance-type g4dn.xlarge \                # 1x T4 (16GB), 4 vCPU, 16GB RAM
  --key-name your-keypair \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=arvton-api}]'
```

### Security Group
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0   # API
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxxxx \
  --protocol tcp --port 22 --cidr YOUR_IP/32     # SSH
```

### Docker Deployment
```bash
# SSH into instance
ssh -i your-keypair.pem ubuntu@<public-ip>

# Install Docker (if not on DLAMI)
sudo apt-get update && sudo apt-get install -y docker.io nvidia-docker2
sudo systemctl restart docker

# Clone and build
git clone https://github.com/yourorg/arvton.git
cd arvton

# Build Docker image
sudo docker build -f app/Dockerfile -t arvton-api .

# Run with GPU access
sudo docker run -d \
  --name arvton \
  --gpus all \
  -p 8000:8000 \
  -v /home/ubuntu/arvton/outputs:/app/outputs \
  -e BASE_URL=http://<public-ip>:8000 \
  -e ALLOWED_ORIGINS="*" \
  -e CUDA_VISIBLE_DEVICES=0 \
  arvton-api

# Verify
curl http://localhost:8000/health
```

### Environment Variables
```bash
sudo docker exec arvton env
# Set via -e flags or .env file:
#   BASE_URL=https://api.yourdomain.com
#   ALLOWED_ORIGINS=https://yourdomain.com
#   ARVTON_S3_BUCKET=your-bucket (optional)
```

---

## Section 2: GCP N1 + T4 GPU (~$0.35/hr spot)

### Create Instance
```bash
gcloud compute instances create arvton-api \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --preemptible                                  # Spot pricing
```

### Firewall
```bash
gcloud compute firewall-rules create arvton-api \
  --allow tcp:8000 \
  --source-ranges 0.0.0.0/0 \
  --target-tags arvton
```

### Docker Deployment
```bash
gcloud compute ssh arvton-api --zone=us-central1-a

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Build and run
git clone https://github.com/yourorg/arvton.git && cd arvton
sudo docker build -f app/Dockerfile -t arvton-api .
sudo docker run -d --gpus all -p 8000:8000 --name arvton \
  -e BASE_URL=http://$(curl -s ifconfig.me):8000 \
  arvton-api

curl http://localhost:8000/health
```

---

## Section 3: Update Flutter App to Production

### Edit `.env`
```bash
cd arvton_app
# Update the backend URL
echo "BACKEND_URL=https://api.yourdomain.com" > .env
echo "API_KEY=your-production-key" >> .env
```

### Build Releases
```bash
# iOS (requires macOS + Xcode 15+)
flutter build ios --release
# Open Xcode → Product → Archive → Distribute

# Android
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk

# Android App Bundle (for Play Store)
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

### Deploy to Stores
- **App Store**: Upload `.ipa` via Xcode or Transporter
- **Play Store**: Upload `.aab` via Google Play Console

---

## Section 4: Run Test Suite Against Production

```bash
# Set production URL
export BACKEND_URL=https://api.yourdomain.com

# Run pytest
pytest tests/test_api.py -v --timeout=120

# Run curl-based test script
bash app/test_api.sh $BACKEND_URL

# Expected output:
#   Test 1: POST /tryon → 202                    ✓ PASS
#   Test 2: Poll until done                       ✓ PASS
#   Test 3: Download GLB                          ✓ PASS
#   Test 4: GET /health                           ✓ PASS
#   Test 5: Oversized file → 413                  ✓ PASS
#   Test 6: PDF file → 422                        ✓ PASS
#   Test 7: DELETE job → 204                      ✓ PASS
```

---

## Section 5: Rollback Procedure

### Roll Back Docker Image
```bash
# List available images
sudo docker images arvton-api

# Stop current container
sudo docker stop arvton && sudo docker rm arvton

# Run previous version
sudo docker run -d --gpus all -p 8000:8000 --name arvton \
  arvton-api:previous-tag

# Verify
curl http://localhost:8000/health
```

### Roll Back Model Checkpoint
```bash
# Load a previous checkpoint
sudo docker exec arvton python -c "
import torch
from pipeline.refine import load_refinement_model
model = load_refinement_model('checkpoints/gan/gan_epoch_040.pt')
print('Rolled back to epoch 40')
"
```

### Emergency: Disable GPU Route
```bash
# Return HTTP 503 for all inference requests
sudo docker exec arvton bash -c '
cat > /app/maintenance.py << EOF
from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE"])
async def maintenance(path: str):
    if path == "health":
        return {"status": "maintenance", "message": "Service temporarily unavailable"}
    raise HTTPException(503, "Service under maintenance. Please try again later.")
EOF
'

# Restart with maintenance mode
sudo docker stop arvton
sudo docker run -d --gpus all -p 8000:8000 --name arvton \
  arvton-api \
  uvicorn app.maintenance:app --host 0.0.0.0 --port 8000
```

### Recovery Checklist
1. ✅ Verify health endpoint responds
2. ✅ Run `bash app/test_api.sh` — all 7 tests pass
3. ✅ Check GPU utilization: `nvidia-smi`
4. ✅ Monitor logs: `sudo docker logs -f arvton`
5. ✅ Run one manual try-on through the Flutter app
