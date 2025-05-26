# Use Python 3.11 as base image
FROM python:3.11-slim
LABEL authors="Samuel Nussbaumer"

# Install curl for Ollama installation
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory structure
COPY . .

# Start Ollama server and pull required models
RUN ollama serve & \
    sleep 5 && \
    ollama pull mxbai-embed-large && \
    ollama pull llama3.2

# Command to run the script
CMD ["python", "src/search.py"]