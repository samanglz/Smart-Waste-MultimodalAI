FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/retrieval_index ./data/retrieval_index
COPY exports ./exports

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[generation]"

EXPOSE 8000

CMD ["uvicorn", "smart_waste_multimodal.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
