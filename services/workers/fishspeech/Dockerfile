# Fish-Speech worker. Deliberately pinned to the v1.5.1 tag rather than
# current main: fishaudio/fish-speech has since rebranded to "Fish Audio S2"
# (main, March 2026), whose license restricts the CODE itself to
# research/non-commercial use — a stricter category than anything else in
# this pipeline. v1.5.1 is the version selected_tts_systems.csv actually
# refers to: Apache-2.0 code, CC-BY-NC-SA-4.0 weights (still live on HF) —
# see docs/licensing.md. Not pip-installable as a published package, but
# does have a real pyproject.toml, so `pip install /opt/fish-speech` works
# cleanly (unlike StyleTTS2/MaskGCT's raw sys.path vendoring).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# portaudio19-dev: pyaudio (a hard dependency, used only for mic input in
# fish-speech's interactive CLI tools we never call) needs portaudio.h to
# build its C extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        build-essential portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# torch==2.4.1: satisfies v1.5.1's own ceiling (torch<=2.4.1 in its "stable"
# extra) AND transformers>=4.45.2's unpinned floor, which otherwise resolves
# to transformers 5.x and hits the same torch<2.4-disables-backend failure
# mode as XTTS/StyleTTS2/F5-TTS — pinning exactly at the ceiling dodges both
# problems in one shot.
RUN pip3 install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# The base image's pip (22.0.2) mis-builds fish-speech's pyproject.toml-based
# local install into an empty "UNKNOWN-0.0.0" wheel (same bug hit with
# shared/python earlier in this project) — upgrade pip first.
RUN pip3 install --no-cache-dir --upgrade pip

ARG FISHSPEECH_COMMIT=58046eaa1a4cefb0c8cc3a3a667b34186ea02dde
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech \
    && cd /opt/fish-speech && git checkout ${FISHSPEECH_COMMIT}

# Base dependency list (pyproject.toml `dependencies`) deliberately excludes
# torch/torchaudio (those only live in the optional "stable" extra), so this
# won't touch the pins installed above.
RUN pip3 install --no-cache-dir /opt/fish-speech

# fish-speech's own code calls pyrootutils.setup_root(__file__, indicator=
# ".project-root") at import time, searching upward from wherever the module
# file actually lives for that marker — which only exists at the repo root,
# not anywhere above the site-packages copy pip just installed. Point
# PYTHONPATH at the repo checkout so imports resolve there instead, where
# .project-root is a sibling of the fish_speech/ package.
ENV PYTHONPATH=/opt/fish-speech

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# Checkpoints load via huggingface_hub.hf_hub_download with an explicit
# local_dir (see model.py) — HF_HOME set for consistency/anything that falls
# back to it (e.g. transformers' own auxiliary downloads, if any).
ENV HF_HOME=/data/hf_cache

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/fishspeech/model.py services/workers/fishspeech/worker.py /app/
COPY services/workers/fishspeech/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
