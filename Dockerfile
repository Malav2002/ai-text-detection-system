FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

# Copy kaggle credentials to the expected location
RUN mkdir -p /root/.config/kaggle
COPY .kaggle/kaggle.json /root/.config/kaggle/kaggle.json
RUN chmod 600 /root/.config/kaggle/kaggle.json

ENV PYTHONPATH=/app

RUN mkdir -p /app/data/{raw,processed,models} /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
