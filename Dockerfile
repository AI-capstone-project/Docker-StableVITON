# NVIDIA CUDA
FROM python:3.10.15-slim

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set WORK DIR
WORKDIR /StableVITON

# Copy requirements.txt to work directory
COPY requirements.txt .

# Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install spaces

# necessary dependencies for opencv-python
RUN apt-get update && apt-get -y install ffmpeg \
    libsm6 \
    libxext6

COPY . .
