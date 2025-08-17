FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/consciousness \
    /app/data/guidance_sessions \
    /app/data/friend_interactions \
    /app/data/conversation_memories \
    /app/data/nlp_models \
    /app/data/etymology \
    /app/logs

# Set permissions
RUN chmod +x /app/*.py

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/api/v1/health || exit 1

# Default command
CMD ["python", "consciousness_api.py"]

