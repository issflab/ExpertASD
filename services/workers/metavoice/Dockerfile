# MetaVoice-1B worker. Upstream pinned to a commit SHA.
# torch 2.1.0 matches upstream's lockfile (audiocraft 1.2 and
# xformers 0.0.22.post7 both require it). build-essential is needed at
# runtime: TTS() runs torch.compile (Triton), which invokes a C compiler.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip so it matches modern manylinux wheels (e.g. av/PyAV 11.0.0). The
# base image's pip 22.0.2 falls back to the av sdist, which then fails to build
# against Ubuntu 22.04's ffmpeg. A modern pip installs the prebuilt wheel.
RUN pip3 install --no-cache-dir -U pip setuptools wheel

RUN pip3 install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu121

# audiocraft (a MetaVoice dependency) hard-pins av==11.0.0, which no longer
# ships a PyPI wheel and requires ffmpeg 6 to build from source (this base image
# has ffmpeg 4.4). Install audiocraft from a patched sdist that relaxes the pin
# to a wheeled av 12.x (whose wheels bundle their own ffmpeg), so MetaVoice then
# resolves against an already-satisfied, wheel-installable av.
RUN pip3 download audiocraft==1.3.0 --no-deps --no-binary :all: -d /tmp/ac \
    && cd /tmp/ac && tar xzf audiocraft-1.3.0.tar.gz \
    && sed -i 's/^av==11.0.0/av>=12.0.0,<13.0.0/' audiocraft-1.3.0/requirements.txt \
    && pip3 install --no-cache-dir ./audiocraft-1.3.0 \
    && rm -rf /tmp/ac

ARG METAVOICE_COMMIT=de3fa211ac4621e03a5f990651aeecc64da418f5
# MetaVoice sets torch._inductor.config.fx_graph_cache at import, but that attr
# only exists in torch >= 2.2 while its own lockfile pins torch 2.1.0 (which
# xformers 0.0.22 requires). It is an optional compile-time cache flag ("on by
# default in future"), the only torch-2.2+ API used, so we drop the 3-line
# assignment to keep the tested torch 2.1.0 stack intact.
RUN git clone https://github.com/metavoiceio/metavoice-src.git /opt/metavoice \
    && cd /opt/metavoice \
    && git checkout ${METAVOICE_COMMIT} \
    && sed -i '/torch\._inductor\.config\.fx_graph_cache = ($/,/^)$/d' \
        fam/llm/fast_inference_utils.py \
    && pip3 install --no-cache-dir -e /opt/metavoice \
    && pip3 install --no-cache-dir soundfile

# Shared library: runtime deps explicit, package on PYTHONPATH via COPY (not a
# wheel) — the base image's older pip mis-builds the local pyproject otherwise.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# Upstream's telemetry opt-out check is `bool(os.getenv("ANONYMIZED_TELEMETRY",
# True))` — truthy for ANY non-empty string, so only the empty string disables it.
ENV ANONYMIZED_TELEMETRY=

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/metavoice/model.py services/workers/metavoice/worker.py /app/
COPY services/workers/metavoice/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
