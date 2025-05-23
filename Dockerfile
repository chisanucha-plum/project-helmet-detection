FROM python:3.12.9-slim

WORKDIR /application
COPY . .

# ติดตั้ง system dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-tha \
    fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# ถ้าต้องการรัน FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ถ้าต้องการรัน main.py (เช่นรัน batch หรือทดสอบ)
# CMD ["python", "main.py"]
