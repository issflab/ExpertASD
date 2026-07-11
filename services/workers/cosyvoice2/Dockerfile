# CosyVoice2 worker. Upstream pinned to a commit SHA; --recursive clone is
# required for the third_party/Matcha-TTS submodule.
# requirements.txt is trimmed to an inference-only subset: deepspeed,
# tensorrt-cu12*, and gradio are training/demo/acceleration extras that are
# NOT imported at module level by cosyvoice.cli (verified against source;
# tensorrt/vllm imports are lazy inside load_trt/load_vllm, which we don't call).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg sox curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
RUN git clone https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && cd /opt/CosyVoice \
    && git checkout ${COSYVOICE_COMMIT} \
    && git submodule update --init --recursive

RUN grep -vE '^(deepspeed|tensorrt|gradio|vllm|wandb)' /opt/CosyVoice/requirements.txt \
        > /tmp/requirements-inference.txt \
    && pip3 install --no-cache-dir -r /tmp/requirements-inference.txt

# Shared library: runtime deps explicit, package on PYTHONPATH via COPY (not a
# wheel) — the base image's older pip mis-builds the local pyproject otherwise.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:/app
ENV COSYVOICE_MODEL_DIR=/data/hf_cache/expertasd_models/CosyVoice2-0.5B

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/cosyvoice2/model.py services/workers/cosyvoice2/worker.py /app/
COPY services/workers/cosyvoice2/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
