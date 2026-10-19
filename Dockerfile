FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg and spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir nemoguardrails==0.24.1 --ignore-requires-python

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUTF8=1
