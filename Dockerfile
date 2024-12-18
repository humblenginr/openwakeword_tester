FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    speex \
    speexdsp-utils \
    libspeexdsp-dev \
    gcc \
    g++ \
    python3-dev \
    libasound2-dev \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables
ENV PORT=8333

# Expose the port
EXPOSE 8333

# Run the application
CMD gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:$PORT app:app
