# NVIDIA CUDA
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
 
# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
 
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
 
CMD [ "python3", "app.py" ]