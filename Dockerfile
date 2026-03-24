FROM python:3.10-slim

# Install system dependencies (ffmpeg and OpenCV requirements)
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CPU version of PyTorch for smaller footprint and broader compatibility
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the whole project structure
COPY . .

# Expose Flask port
EXPOSE 5000

# Start server
WORKDIR /app/audio_visual
CMD ["python", "app.py"]
