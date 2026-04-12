FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install system deps (for lxml and other wheels if needed)

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libgomp1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get purge -y --auto-remove gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Default to showing help. Override with docker run command.
CMD ["python3", "main.py", "--help"]
