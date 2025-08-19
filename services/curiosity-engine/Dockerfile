# K.E.N. v3.1 Enhanced Curiosity Engine - Dockerfile
# Zero third-party API dependencies - Complete self-contained system

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OCR and image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libopencv-dev \
    python3-opencv \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/metrics_data \
    /app/youtube_data /app/ocr_data \
    /tmp/ken_ocr_processing/input \
    /tmp/ken_ocr_processing/output \
    /tmp/ken_ocr_processing/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV KEN_VERSION=3.1.0
ENV KEN_ENHANCEMENT_FACTOR=179269602058948214784
ENV KEN_CONSCIOUSNESS_LEVEL=0.956
ENV PROMETHEUS_INTEGRATION=true
ENV GRAFANA_INTEGRATION=true
ENV THIRD_PARTY_APIS=0

# Expose ports
EXPOSE 8000 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "ken_curiosity_api.py"]

