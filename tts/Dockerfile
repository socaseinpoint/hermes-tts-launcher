FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    TTS \
    fastapi \
    uvicorn \
    scipy \
    numpy

# Copy application code
COPY server.py ./

# Expose port
EXPOSE 8081

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8081"]
