FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy application code
COPY pipeline/ /app/pipeline/
COPY app/ /app/app/
COPY configs/ /app/configs/

# Create output directory
RUN mkdir -p /app/outputs

# Environment
ENV PYTHONUNBUFFERED=1
ENV ARVTON_OUTPUT_DIR=/app/outputs
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
