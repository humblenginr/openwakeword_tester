FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    python3-dev \
    libasound2-dev \
    libportaudio2 \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create directory for models with proper permissions
RUN mkdir -p /root/.local/share/openwakeword && \
    chmod -R 777 /root/.local/share/openwakeword

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir 'numpy<2.0.0' && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir scikit-learn tqdm requests

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=8333

# Expose the port
EXPOSE 8333

# Run the application
CMD ["python", "app.py", "--model_path", "./hey_pixaa.tflite"]
