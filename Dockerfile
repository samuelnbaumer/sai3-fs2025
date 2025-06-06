FROM python:3.11-slim
LABEL authors="Samuel Nussbaumer"

RUN apt-get update && apt-get install -y curl

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["python", "src/webserver.py"]