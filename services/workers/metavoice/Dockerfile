# MetaVoice-1B worker. Upstream pinned to a commit SHA.
# torch 2.1.0 matches upstream's lockfile (audiocraft 1.2 and
# xformers 0.0.22.post7 both require it). build-essential is needed at
# runtime: TTS() runs torch.compile (Triton), which invokes a C compiler.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

ARG METAVOICE_COMMIT=de3fa211ac4621e03a5f990651aeecc64da418f5
RUN git clone https://github.com/metavoiceio/metavoice-src.git /opt/metavoice \
    && cd /opt/metavoice \
    && git checkout ${METAVOICE_COMMIT} \
    && pip3 install --no-cache-dir -e /opt/metavoice \
    && pip3 install --no-cache-dir soundfile

COPY shared/python /shared_python
RUN pip3 install --no-cache-dir /shared_python

# Upstream's telemetry opt-out check is `bool(os.getenv("ANONYMIZED_TELEMETRY",
# True))` — truthy for ANY non-empty string, so only the empty string disables it.
ENV ANONYMIZED_TELEMETRY=

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/metavoice/model.py services/workers/metavoice/worker.py /app/
COPY services/workers/metavoice/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
