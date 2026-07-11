# Tortoise-TTS worker. Upstream pinned to a commit SHA, never a branch.
# torch 2.3.1+cu121 chosen explicitly (upstream leaves torch unpinned).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Tortoise's setup.py is internally self-contradictory: it pins BOTH
# transformers==4.31.0 AND tokenizers==0.14.0, but transformers 4.31.0 forbids
# tokenizers>=0.14 — so the package is uninstallable via pip's resolver. Install
# a coherent dependency set explicitly, then install tortoise with --no-deps.
# tokenizers 0.13.3 satisfies transformers 4.31.0 (>=0.11.1,!=0.11.3,<0.14),
# which is the API tortoise's code actually targets. torch/torchaudio are
# re-pinned here so an unconstrained transitive dep (rotary_embedding_torch)
# cannot silently upgrade the cu121 torch installed above.
RUN pip3 install --no-cache-dir \
        torch==2.3.1 torchaudio==2.3.1 \
        transformers==4.31.0 tokenizers==0.13.3 scipy==1.13.1 \
        tqdm rotary_embedding_torch inflect progressbar einops unidecode librosa

ARG TORTOISE_COMMIT=8a2563ecabe93c4fb626f876dd0c52c966edef2f
RUN pip3 install --no-cache-dir --no-deps \
        "git+https://github.com/neonbjb/tortoise-tts.git@${TORTOISE_COMMIT}"

# Shared library: runtime deps installed explicitly, package placed on
# PYTHONPATH via COPY (not built as a wheel) — the base image's older pip
# mis-builds the local pyproject as an empty "UNKNOWN" wheel otherwise.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

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
