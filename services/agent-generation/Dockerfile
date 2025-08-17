FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/agents \
    /app/data/models \
    /app/data/training \
    /app/logs \
    /app/cache

# Set permissions
RUN chmod +x /app/*.py

# Create non-root user
RUN useradd -m -u 1000 kenagent && \
    chown -R kenagent:kenagent /app
USER kenagent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8004/api/health || exit 1

# Expose port
EXPOSE 8004

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV KEN_ENVIRONMENT=production

# Start command
CMD ["python", "ken_agent_api.py"]

