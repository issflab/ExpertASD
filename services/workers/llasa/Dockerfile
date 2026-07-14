# LLASA worker. Unlike most other systems here, there's no upstream code repo
# to clone — LLaSA_training (github.com/zhenye234/LLaSA_training) is
# training/finetuning scripts only, with no inference code. The actual
# runnable model is a standard HF `transformers.AutoModelForCausalLM`
# checkpoint (HKUSTAudio/Llasa-1B) plus a separate codec package (xcodec2,
# MIT, PyPI) that encodes/decodes speech tokens. Weights (both Llasa-1B and
# xcodec2) are CC-BY-NC-4.0 — see docs/licensing.md.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv python3.10-dev git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Exact upstream pin (xcodec2==0.1.5's own setup.py: torch==2.5.0,
# torchaudio==2.5.0). This also satisfies transformers 5.x's torch>=2.4 floor
# (xcodec2 itself only needs transformers>=4.45.2, unpinned) — same fix
# pattern as XTTS/StyleTTS2/F5-TTS, applied preemptively here.
RUN pip3 install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir xcodec2==0.1.5
# xcodec2's own setup.py pins torch==2.5.0 exactly but leaves torchao
# unpinned (only "torchao>=0.5.0") — pip resolved the latest torchao
# (0.17.0), whose code unconditionally references torch.int1 (a sub-byte
# dtype torch didn't add until well after 2.5.0), breaking the import of
# transformers.PreTrainedModel itself (transformers' quantizer integrations
# eagerly import torchao). Same self-contradictory-pins pattern as Tortoise;
# re-pin to the version at xcodec2's own stated floor.
RUN pip3 install --no-cache-dir torchao==0.5.0

# xcodec2's XCodec2Model.__init__ calls Wav2Vec2BertModel.from_pretrained()
# (loading facebook/w2v-bert-2.0, the same semantic encoder MaskGCT uses)
# *nested inside* the outer XCodec2Model.from_pretrained() call — while
# transformers 5.x's meta-device model-construction context is still active,
# which now hard-rejects that nested pattern ("anti-pattern... wants to load
# existing weights" under a meta-device context). xcodec2's own code
# predates this transformers behavior; pin to a 4.x release from its
# release era where meta-device construction wasn't the aggressive default.
RUN pip3 install --no-cache-dir transformers==4.46.3

# model.py writes output via soundfile directly; nothing else in this
# dependency chain pulls it in transitively (unlike most other workers,
# which get it via librosa).
RUN pip3 install --no-cache-dir soundfile

# Shared library: same pattern as the other workers.
COPY shared/python/requirements.txt /tmp/shared-req.txt
RUN pip3 install --no-cache-dir -r /tmp/shared-req.txt
COPY shared/python/expertasd_common /app/expertasd_common

# Both checkpoints (HKUSTAudio/Llasa-1B, HKUSTAudio/xcodec2) load via
# transformers' standard from_pretrained cache, which honors HF_HOME.
ENV HF_HOME=/data/hf_cache

WORKDIR /app
COPY scripts/check_data_disk.sh /app/check_data_disk.sh
COPY services/workers/llasa/model.py services/workers/llasa/worker.py /app/
COPY services/workers/llasa/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh /app/check_data_disk.sh

ENTRYPOINT ["/app/entrypoint.sh"]
