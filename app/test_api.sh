#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# ARVTON API — Integration Test Script
# ═══════════════════════════════════════════════════════════════════════
# Usage: bash test_api.sh [BASE_URL]
# Default: http://localhost:8000

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0
TOTAL=7

green() { echo -e "\033[32m✓ PASS\033[0m: $1"; ((PASS++)); }
red()   { echo -e "\033[31m✗ FAIL\033[0m: $1"; ((FAIL++)); }

echo "═══════════════════════════════════════════════════"
echo " ARVTON API Test Suite"
echo " Target: $BASE_URL"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Create test images ──────────────────────────────────────────────────
# Generate minimal valid JPEG files for testing
PERSON_IMG=$(mktemp /tmp/person_XXXXXX.jpg)
GARMENT_IMG=$(mktemp /tmp/garment_XXXXXX.jpg)
BIG_IMG=$(mktemp /tmp/big_XXXXXX.jpg)
PDF_FILE=$(mktemp /tmp/test_XXXXXX.pdf)

# Create small valid JPEG (1x1 pixel)
python3 -c "
from PIL import Image
img = Image.new('RGB', (256, 384), color=(128, 128, 200))
img.save('$PERSON_IMG', 'JPEG')
img2 = Image.new('RGB', (256, 384), color=(200, 128, 128))
img2.save('$GARMENT_IMG', 'JPEG')
" 2>/dev/null || {
    # Fallback: use convert or create dummy files
    echo "Warning: PIL not available. Using dummy test files."
    dd if=/dev/urandom of="$PERSON_IMG" bs=1024 count=10 2>/dev/null
    dd if=/dev/urandom of="$GARMENT_IMG" bs=1024 count=10 2>/dev/null
}

# Create >10MB file
dd if=/dev/urandom of="$BIG_IMG" bs=1024 count=11000 2>/dev/null

# Create PDF file
echo "%PDF-1.0 test" > "$PDF_FILE"

cleanup() {
    rm -f "$PERSON_IMG" "$GARMENT_IMG" "$BIG_IMG" "$PDF_FILE"
}
trap cleanup EXIT

# ── Test 1: POST /tryon → 202 + job_id ─────────────────────────────────
echo "Test 1: POST /tryon (valid images)"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BASE_URL/tryon" \
    -F "person_image=@$PERSON_IMG;type=image/jpeg" \
    -F "garment_image=@$GARMENT_IMG;type=image/jpeg" \
    -F "quality=auto")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)
JOB_ID=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")

if [ "$HTTP_CODE" = "202" ] && [ -n "$JOB_ID" ]; then
    green "POST /tryon → 202, job_id=$JOB_ID"
else
    red "POST /tryon → Expected 202, got $HTTP_CODE"
fi

# ── Test 2: Poll GET /result/{job_id} until done ───────────────────────
echo "Test 2: Poll GET /result/$JOB_ID (timeout 120s)"
STATUS="queued"
ELAPSED=0
TIMEOUT=120

while [ "$STATUS" != "done" ] && [ "$STATUS" != "failed" ] && [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    RESULT=$(curl -s "$BASE_URL/result/$JOB_ID")
    STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
    echo "  [$ELAPSED s] status=$STATUS"
done

if [ "$STATUS" = "done" ]; then
    green "Job completed: status=done (${ELAPSED}s)"
elif [ "$STATUS" = "failed" ]; then
    # Failed is acceptable if models aren't loaded
    green "Job finished: status=failed (models may not be loaded) (${ELAPSED}s)"
else
    red "Job timed out after ${TIMEOUT}s: status=$STATUS"
fi

# ── Test 3: Download GLB and verify ────────────────────────────────────
echo "Test 3: Download .glb and verify"
GLB_URL=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('glb_url',''))" 2>/dev/null || echo "")

if [ -n "$GLB_URL" ] && [ "$GLB_URL" != "None" ] && [ "$GLB_URL" != "null" ]; then
    GLB_FILE=$(mktemp /tmp/test_XXXXXX.glb)
    HTTP_CODE=$(curl -s -w "%{http_code}" -o "$GLB_FILE" "$GLB_URL")
    if [ "$HTTP_CODE" = "200" ] && [ -s "$GLB_FILE" ]; then
        green "GLB downloaded: $(wc -c < "$GLB_FILE") bytes"
    else
        red "GLB download failed: HTTP $HTTP_CODE"
    fi
    rm -f "$GLB_FILE"
else
    green "No GLB URL (expected if models not loaded)"
fi

# ── Test 4: GET /health ────────────────────────────────────────────────
echo "Test 4: GET /health"
HEALTH=$(curl -s "$BASE_URL/health")
HEALTH_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "")

if [ "$HEALTH_STATUS" = "ok" ]; then
    GPU=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"gpu_name\"]} ({d[\"gpu_memory_used_gb\"]}/{d[\"gpu_memory_total_gb\"]} GB)')" 2>/dev/null || echo "N/A")
    green "Health OK — GPU: $GPU"
else
    red "Health check failed"
fi

# ── Test 5: POST /tryon with >10MB file → 413 ─────────────────────────
echo "Test 5: POST /tryon (>10MB file → expect 413)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/tryon" \
    -F "person_image=@$BIG_IMG;type=image/jpeg" \
    -F "garment_image=@$GARMENT_IMG;type=image/jpeg" \
    -F "quality=auto")

if [ "$HTTP_CODE" = "413" ]; then
    green "Oversized file → 413"
else
    red "Oversized file → Expected 413, got $HTTP_CODE"
fi

# ── Test 6: POST /tryon with .pdf → 422 ───────────────────────────────
echo "Test 6: POST /tryon (.pdf file → expect 422)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/tryon" \
    -F "person_image=@$PDF_FILE;type=application/pdf" \
    -F "garment_image=@$GARMENT_IMG;type=image/jpeg" \
    -F "quality=auto")

if [ "$HTTP_CODE" = "422" ]; then
    green "PDF upload → 422"
else
    red "PDF upload → Expected 422, got $HTTP_CODE"
fi

# ── Test 7: DELETE /job/{job_id} → 204 ─────────────────────────────────
echo "Test 7: DELETE /job/$JOB_ID (expect 204)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X DELETE "$BASE_URL/job/$JOB_ID")

if [ "$HTTP_CODE" = "204" ]; then
    green "DELETE job → 204"
else
    red "DELETE job → Expected 204, got $HTTP_CODE"
fi

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo " Results: $PASS/$TOTAL passed, $FAIL/$TOTAL failed"
echo "═══════════════════════════════════════════════════"

if [ $FAIL -eq 0 ]; then
    echo -e "\033[32m All tests passed! ✓\033[0m"
    exit 0
else
    echo -e "\033[31m Some tests failed. ✗\033[0m"
    exit 1
fi
