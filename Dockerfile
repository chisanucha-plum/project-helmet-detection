FROM python:3.13-alpine

WORKDIR /application
COPY . .

RUN apk update && \
    apk add --no-cache \
        build-base \
        curl \
        tzdata && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "9"]
