# StyleTTS2 worker. Upstream (yl4579/StyleTTS2) is not pip-installable — it's
# a "run from repo root" research codebase, not a package — so this clones the
# pinned commit into /opt/StyleTTS2 and imports its modules directly (same
# approach as Tortoise). Code is MIT; pretrained LibriTTS weights have no
# separate license file but carry a disclosure condition — see
# docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        espeak-ng libespeak-ng1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# torch 2.5.1 (not 2.3.1 like Tortoise/CosyVoice2): StyleTTS2's requirements.txt
# leaves transformers unpinned, which pip resolves to 5.x, and transformers 5.x
# silently disables its torch backend if torch<2.4 — hit the identical failure
# mode debugging the XTTS worker; same fix applies here.
RUN pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

ARG STYLETTS2_COMMIT=5cedc71c333f8d8b8551ca59378bdcc7af4c9529
RUN git clone https://github.com/yl4579/StyleTTS2.git /opt/StyleTTS2 \
    && cd /opt/StyleTTS2 && git checkout ${STYLETTS2_COMMIT}

# requirements.txt (upstream, verified against the actual file) omits
# `phonemizer` despite Utils code importing it directly, and pins nothing —
# install it explicitly plus a coherent numpy/cython pair needed to build the
# monotonic_align Cython extension. transformers is left to resolve (used only
# for the long-stable AlbertModel API in PLBERT, unlike XTTS's fragile
# internal import — verified this doesn't need pinning against 5.x).
RUN pip3 install --no-cache-dir cython numpy \
    && pip3 install --no-cache-dir -r /opt/StyleTTS2/requirements.txt \
    && pip3 install --no-cache-dir phonemizer huggingface_hub

# nltk's word_tokenize needs the punkt tokenizer data; bake it into the image
# (tiny) rather than downloading at every cold start. Newer nltk renamed the
# resource to punkt_tab, so fetch both for compatibility.
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/styletts2/model.py services/workers/styletts2/worker.py /app/
COPY services/workers/styletts2/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
