FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY marketing_orchestrator/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY marketing_orchestrator /app/marketing_orchestrator

WORKDIR /app/marketing_orchestrator

CMD ["python", "main.py"]
