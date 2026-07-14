# F5-TTS worker. Upstream (SWivid/F5-TTS) is a proper, actively-maintained
# PyPI release (f5-tts) — unlike StyleTTS2, this installs cleanly via pip.
# Code is MIT; pretrained weights are CC-BY-NC-4.0 (non-commercial, due to
# training on the Emilia dataset) — see docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# torch 2.5.1 (not 2.3.1 like Tortoise/CosyVoice2): f5-tts leaves transformers
# unpinned, which pip resolves to 5.x, and transformers 5.x silently disables
# its torch backend under torch<2.4 — same failure mode hit debugging the
# XTTS and StyleTTS2 workers; pre-applying the fix here.
RUN pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

ARG F5TTS_PKG_VERSION=1.1.21
RUN pip3 install --no-cache-dir f5-tts==${F5TTS_PKG_VERSION}

# f5-tts declares an unpinned `torchcodec` dependency; pip resolves that to
# the latest (0.14.0), which requires torch>=2.11 and a CUDA-13-linked
# libnvrtc. That's incompatible with our torch==2.5.1+cu121 pin: newer
# torchaudio prefers torchcodec as an I/O backend when present, and it
# crashed loading some (not all) reference WAVs with "Could not load
# libtorchcodec" / "libnvrtc.so.13: cannot open shared object file" —
# verified empirically. f5-tts's own code never calls torchcodec directly,
# so removing it lets torchaudio fall back to its standard backend.
RUN pip3 uninstall -y torchcodec

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# f5-tts downloads weights via `cached_path("hf://SWivid/F5-TTS/...")`; our
# model.py passes hf_cache_dir explicitly (an F5TTS() constructor arg) rather
# than relying on env vars, but HF_HOME is still set for consistency/anything
# that falls back to it.
ENV HF_HOME=/data/hf_cache

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/f5tts/model.py services/workers/f5tts/worker.py /app/
COPY services/workers/f5tts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
