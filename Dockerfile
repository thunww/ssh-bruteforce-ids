FROM python:3.10-slim

# system deps for iptables + journalctl (systemd)
RUN apt-get update && apt-get install -y --no-install-recommends \
        iptables \
        systemd \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY models/ models/

# Tạo thư mục output
RUN mkdir -p outputs/alerts

ENV PYTHONPATH=/app
ENV IDS_ALERTS_PATH=/app/outputs/alerts.jsonl

# Real-time detector cần chạy với quyền root để gọi iptables
CMD ["python", "scripts/09_realtime_detector.py"]
