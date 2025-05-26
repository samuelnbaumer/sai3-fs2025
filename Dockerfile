# Use an official Python 3.11 image as a base
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port (if needed)
EXPOSE 8000

# Set environment variable for Ollama host
ENV OLLAMA_HOST=host.docker.internal

# Run the search.py script in interactive mode
CMD ["python", "-i", "src/search.py"]