FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         gcc         libjpeg-dev zlib1g-dev         && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY kb ./kb
COPY models ./models
RUN mkdir -p /app/media

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
