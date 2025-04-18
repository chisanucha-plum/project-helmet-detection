FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install ultralytics opencv-python matplotlib numpy

CMD ["python", "main.py"]
