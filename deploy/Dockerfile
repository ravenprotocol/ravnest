FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
# torchpippy may not be on PyPI anymore; skip it for inference
RUN pip install --no-cache-dir $(grep -v torchpippy requirements.txt) \
    && pip install --no-cache-dir fastapi uvicorn transformers accelerate

# Copy the project
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
