# SSR-Speech worker. Upstream (WangHelin1997/SSR-Speech) is a speech-editing
# model repurposed for zero-shot TTS, not pip-installable — cloned into
# /opt/SSR-Speech and run with cwd there (same pattern as StyleTTS2/MaskGCT).
# Shares MetaVoice's torch 2.1.0/xformers 0.0.22.post7/audiocraft stack.
# Code is MIT; pretrained weights are CC-BY-NC-SA-4.0 — see docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# libav*-dev: faster-whisper's own av dependency has no prebuilt wheel for
# this environment across the versions pip resolves here, so it must build
# from source against these headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        espeak-ng libespeak-ng1 build-essential pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip so it matches modern manylinux wheels (e.g. av/PyAV) — same
# fix as MetaVoice's Dockerfile, same underlying base-image pip (22.0.2) bug.
RUN pip3 install --no-cache-dir -U pip setuptools wheel

RUN pip3 install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

ARG SSRSPEECH_COMMIT=097c3c4f8c08acd7b431f095fa0f10e3151075c1
RUN git clone https://github.com/WangHelin1997/SSR-Speech.git /opt/SSR-Speech \
    && cd /opt/SSR-Speech && git checkout ${SSRSPEECH_COMMIT}

# SSR-Speech bundles its own audiocraft fork (subdirectory of its repo) with
# av/torch/xformers all left unpinned in that fork's requirements.txt —
# unlike MetaVoice's separate PyPI audiocraft==1.3.0 (which hard-pins
# av==11.0.0 and needed a patched sdist), this installs cleanly with the pip
# upgrade above.
RUN pip3 install --no-cache-dir /opt/SSR-Speech/audiocraft

# Remaining upstream deps (README "Environment setup"), skipping gradio (no
# WebUI here). WhisperX + faster-whisper + ctranslate2 implement the mandatory
# reference-transcription/forced-alignment step SSR-Speech always runs
# internally (see model.py) — this is not optional, unlike F5-TTS's ASR
# fallback.
#
# whisperx bumped from the README's pinned 3.1.5: that version's VAD model
# loader downloads from a hardcoded S3 URL that now 403s (a widely-reported,
# still-unfixed upstream bug — github.com/m-bain/whisperX#943 and friends).
# 3.7.5 bundles the VAD weights as a package asset instead of downloading
# them, avoiding the dead URL entirely. Verified the load_align_model/
# load_model/align/load_audio functions our model.py calls kept the same
# signatures across this range.
#
# ctranslate2, faster-whisper, and huggingface_hub left unpinned (README's
# ==4.4.0/==1.0.1/==0.22.2 each conflict with whisperx 3.7.5's own transitive
# floors) — let pip resolve coherent versions. openai-whisper dropped
# entirely: its old build script does `import pkg_resources`, which current
# setuptools no longer vendors, breaking its build in an isolated build env;
# it's only used by inference_v2.py's alternate WhisperModel class, which
# this port doesn't call (we use whisperx.load_model, matching the CLI's
# actual TTS code path, not the unused legacy one).
RUN pip3 install --no-cache-dir \
        tensorboard==2.16.2 phonemizer==3.2.1 datasets==2.16.0 \
        torchmetrics==0.11.1 huggingface_hub nltk \
        whisperx==3.7.5 faster-whisper \
        num2words==0.5.13 opencc-python-reimplemented ctranslate2

# Every install above declares only unpinned/loose torch floors, and between
# them pip has repeatedly drifted the resolved torch version (observed
# torch 2.0.1 after audiocraft+xformers, then torch 2.8.0 after whisperx's
# own pyannote-audio dependency) — verified empirically, each time silently,
# with pip only warning rather than refusing. Force the final state back to
# the tested-working combination as the last word: torch 2.1.0 (matches
# xformers 0.0.22.post7 and SSR-Speech's own model code), numpy<2 (the
# whisperx/pyannote chain pulls numpy 2.x, which breaks torch's own compiled
# extensions with an ABI mismatch against modules built for numpy 1.x), and
# transformers==4.40.2 (whisperx pulled in a recent transformers whose
# ModelOutput registration unconditionally calls
# torch.utils._pytree.register_pytree_node, a public API torch only added in
# 2.2+ — torch 2.1.0 only has the private _register_pytree_node. 4.40.2 is
# the exact version the metavoice worker already runs successfully against
# this same torch 2.1.0, in this same project — reusing that proven pin
# rather than guessing a new one).
RUN pip3 install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.22.post7 \
    && pip3 install --no-cache-dir "numpy<2" transformers==4.40.2

RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# Not pip-installable — repo-root-relative imports (models.ssr, data.tokenizer,
# etc.), so PYTHONPATH points at the checkout rather than a site-packages
# install. HF_HOME set for consistency; the whisper/whisperx/align models
# download through their own libraries' default HF/torch caches under it.
ENV PYTHONPATH=/opt/SSR-Speech
ENV HF_HOME=/data/hf_cache

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/ssrspeech/model.py services/workers/ssrspeech/worker.py /app/
COPY services/workers/ssrspeech/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
