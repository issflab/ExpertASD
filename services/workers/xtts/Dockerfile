# XTTS-v2 worker. Upstream is idiap/coqui-ai-TTS — the actively maintained
# fork of the now-unmaintained coqui-ai/TTS (PyPI package: coqui-tts).
# Weights are Coqui Public Model License (CPML) 1.0.0 — non-commercial
# research use only, see docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# torch 2.5.1 (not 2.3.1 like the other workers): coqui-tts declares only
# "torch>=2.2", but it transitively pulls transformers>=4.57, and pip resolves
# that to transformers 5.x, which itself requires torch>=2.4 and silently
# disables its torch backend (not a hard error) if that's unmet — TTS's own
# import then fails with a misleading "PyTorch was not found". Verified by
# building with 2.3.1 first and hitting exactly that failure.
RUN pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

# coqui-tts is a proper PyPI release (unlike tortoise/metavoice, which need
# git installs); pinning the exact version is the release-based equivalent of
# pinning a commit SHA. torch/torchaudio already satisfy its ">=2.2" pin so
# this does not upgrade them.
ARG XTTS_PKG_VERSION=0.27.5
RUN pip3 install --no-cache-dir coqui-tts==${XTTS_PKG_VERSION}

# coqui-tts declares only "transformers>=4.57" (no upper bound), so pip
# resolves the newest transformers (5.x) — but XTTS's autoregressive layer
# (copied from Tortoise's GPT code) imports
# transformers.pytorch_utils.isin_mps_friendly, which transformers 5.x
# removed. Re-pin to a 4.x release the floor allows; verified this name
# still exists there.
RUN pip3 install --no-cache-dir transformers==4.57.1

# Shared library: same pattern as the other workers (see tortoise/Dockerfile
# for why this is a COPY onto PYTHONPATH rather than `pip install /shared_python`).
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# COQUI_TOS_AGREED=1 accepts the CPML non-interactively — without it, the
# first weight download blocks on a y/n prompt, which hangs forever in a
# container. TTS_HOME redirects the model cache onto the shared /data cache;
# coqui-tts does not honor HF_HOME despite the weights being HF-hosted.
ENV COQUI_TOS_AGREED=1
ENV TTS_HOME=/data/hf_cache/expertasd_models/xtts

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/xtts/model.py services/workers/xtts/worker.py /app/
COPY services/workers/xtts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
