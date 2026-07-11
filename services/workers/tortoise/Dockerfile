# Tortoise-TTS worker. Upstream pinned to a commit SHA, never a branch.
# torch 2.3.1+cu121 chosen explicitly (upstream leaves torch unpinned).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

ARG TORTOISE_COMMIT=8a2563ecabe93c4fb626f876dd0c52c966edef2f
RUN pip3 install --no-cache-dir \
        "git+https://github.com/neonbjb/tortoise-tts.git@${TORTOISE_COMMIT}"

COPY shared/python /shared_python
RUN pip3 install --no-cache-dir /shared_python

# Tortoise ignores HF_HOME for its own weights (api.py passes
# cache_dir=models_dir to hf_hub_download); point its models_dir at the
# mounted /data cache. Other components (transformers) do use HF_HOME.
ENV TORTOISE_MODELS_DIR=/data/hf_cache/expertasd_models/tortoise

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/tortoise/model.py services/workers/tortoise/worker.py /app/
COPY services/workers/tortoise/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
