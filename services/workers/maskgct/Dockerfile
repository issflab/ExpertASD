# MaskGCT worker. Upstream (open-mmlab/Amphion) is a large multi-model
# monorepo, not pip-installable — MaskGCT's own code cross-imports from the
# repo root (utils.util, models.codec.*), so this clones the pinned commit
# into /opt/Amphion and runs with cwd there (same pattern as StyleTTS2, one
# level bigger). Code is MIT; pretrained weights are CC-BY-NC-4.0 (trained on
# Emilia, same as F5-TTS) — see docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# cmake + build-essential: MaskGCT's g2p module unconditionally imports a
# Japanese cleaner (even for English-only use) which pulls in pyopenjtalk,
# which compiles a C++ extension and its dictionary at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        espeak-ng libespeak-ng1 build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Exact upstream pin (models/tts/maskgct/requirements.txt) — unlike
# XTTS/StyleTTS2/F5-TTS, MaskGCT pins torch==2.0.1 and transformers==4.41.2
# precisely, so there's no unpinned-floor drift to fight here. cu118 is the
# newest CUDA wheel build torch 2.0.1 shipped; runs fine against this host's
# driver via CUDA forward compatibility, same as every other worker's newer
# cu121 wheels do.
RUN pip3 install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

ARG AMPHION_COMMIT=26f6883110181f1dbfe95c70a7c7dbaf4de5f42a
RUN git clone https://github.com/open-mmlab/Amphion.git /opt/Amphion \
    && cd /opt/Amphion && git checkout ${AMPHION_COMMIT}

# pyopenjtalk (a transitive dep, only for Japanese g2p we never use) bundles
# hts_engine_API with a CMakeLists.txt whose ancient `cmake_minimum_required`
# floor modern CMake (>=4, from the isolated build env's own pinned cmake
# wheel) refuses to honor without an explicit override.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip3 install --no-cache-dir -r /opt/Amphion/models/tts/maskgct/requirements.txt

# LangSegment 0.2.0 (the only version ever published to PyPI) ships a broken
# __init__.py: it imports setLangfilters/getLangfilters, names that don't
# exist anywhere in its own LangSegment.py (only setfilters/getfilters do) —
# so the package can't even be imported to introspect its own path. This
# breaks every consumer, not just us — confirmed via open-mmlab/Amphion#450,
# still unfixed upstream. Patch the installed file directly.
RUN LANGSEG_INIT=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/LangSegment/__init__.py \
    && echo "from .LangSegment import LangSegment, getTexts, classify, getCounts, printList, setfilters, getfilters" > "$LANGSEG_INIT"

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# All checkpoints (amphion/MaskGCT's 5 safetensors files, plus
# facebook/w2v-bert-2.0) load via huggingface_hub, which honors HF_HOME
# natively — no explicit cache_dir plumbing needed, unlike F5-TTS/XTTS.
ENV HF_HOME=/data/hf_cache

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/maskgct/model.py services/workers/maskgct/worker.py /app/
COPY services/workers/maskgct/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
