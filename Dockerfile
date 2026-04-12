FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run.sh .
RUN chmod +x run.sh

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
