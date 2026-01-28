FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY agent/ ./agent/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Start command
CMD ["python", "-m", "uvicorn", "agent.serving.model_server:app", "--host", "0.0.0.0", "--port", "8000"]
