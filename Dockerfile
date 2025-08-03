FROM python:3.11-slim-buster

WORKDIR /application
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]
