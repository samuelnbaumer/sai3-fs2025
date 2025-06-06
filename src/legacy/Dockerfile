# Use an official Python 3.11 image as a base
FROM python:3.11-slim

# Install curl and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Start Ollama server and pull models
RUN OLLAMA_DEBUG=0 ollama serve & \
    sleep 10 && \
    ollama pull llama3.2 && \
    ollama pull mxbai-embed-large

# Expose the Flask port
EXPOSE 8000
EXPOSE 1337

# Start Ollama server and run the Flask application
CMD OLLAMA_DEBUG=0 ollama serve > /dev/null 2>&1 & sleep 5 && python src/webserver.py