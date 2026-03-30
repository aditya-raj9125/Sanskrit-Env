FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py .
COPY client.py .
COPY openenv.yaml .
COPY graders/ ./graders/
COPY data/ ./data/
COPY server/ ./server/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# HF Spaces uses port 7860 by default
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
