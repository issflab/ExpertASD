# Architecture

## Topology

```
   caller / smoke test
          │  HTTP :8000
          ▼
    ┌───────────┐      enqueue      ┌──────────┐
    │  gateway  │ ────────────────▶ │  redis   │  (RQ queues + job bookkeeping)
    │ (FastAPI) │ ◀──────────────── │  :6379   │
    └─────┬─────┘   job status      └────┬─────┘
          │ StaticFiles /outputs         │ one named queue per system
          ▼                              ▼
   /data/.../outputs/<job_id>/     worker-<system> × 10 (one container per
     audio.wav + metadata.json     ◀── TTS system, each consuming its own
          ▲                            named queue and holding one GPU
          └── writes results           slot — see the GPU-to-worker table
                                        below for the current system→GPU
                                        assignment. All workers mount
                                        /data/hf_cache (weights, ro-ish).
```

The gateway is the only host-exposed service. Workers are reachable only on the
internal compose network; their `/health` ports (8080) are not published.

## GPU-to-worker assignment

| Worker            | Queue          | GPU index | Notes                              |
|-------------------|----------------|-----------|------------------------------------|
| worker-tortoise   | tortoise-tts   | 0         | Tortoise-TTS, diffusion (slow)     |
| worker-cosyvoice2 | cosyvoice2     | 1         | requires reference_text            |
| worker-metavoice  | metavoice-1b   | 2         | torch.compile warm-up on 1st call  |
| worker-xtts       | xtts           | 0 (shared with worker-tortoise) | weights are CPML, non-commercial only — see `licensing.md` |
| worker-styletts2  | styletts2      | 1 (shared with worker-cosyvoice2) | smallest/fastest system; not pip-installable, cloned repo — see `adding-a-new-tts-system.md` |
| worker-f5tts      | f5-tts         | 2 (shared with worker-metavoice) | weights are CC-BY-NC-4.0, non-commercial only; output duration is sensitive to reference audio/text pairing quality — see `resource-requirements.md` |
| worker-maskgct    | maskgct        | 1 (shared with worker-cosyvoice2 + worker-styletts2, 3-way) | weights are CC-BY-NC-4.0; heaviest system (6 model components + a w2v-bert-2.0 encoder, ~11.1 GB resident); requires reference_text; vendored monorepo like StyleTTS2, plus a broken upstream `LangSegment` PyPI package patched in the Dockerfile — see `adding-a-new-tts-system.md` |
| worker-fishspeech | fish-speech    | 2 (shared with worker-metavoice + worker-f5tts, 3-way) | pinned to v1.5.1, NOT current main (see `licensing.md`); weights are CC-BY-NC-SA-4.0 (share-alike); requires reference_text; only system that outputs 44.1 kHz — see `resource-requirements.md` |
| worker-ssrspeech  | ssrspeech      | 0 (shared with worker-tortoise + worker-xtts, 3-way) | a speech-*editing* model repurposed for zero-shot TTS, not a from-scratch design; always auto-transcribes the reference via WhisperX internally (reference_text ignored); weights are CC-BY-NC-SA-4.0 (share-alike); only system that outputs 16 kHz; shares MetaVoice's torch 2.1.0/xformers stack — see `resource-requirements.md` and `licensing.md` |
| worker-llasa      | llasa          | 2 (shared with worker-metavoice + worker-f5tts + worker-fishspeech, 4-way) | no upstream code repo — a standard HF `transformers.AutoModelForCausalLM` checkpoint + the `xcodec2` PyPI package; weights are CC-BY-NC-4.0; requires reference_text; needed a `repetition_penalty` added to `generate()` (absent upstream) to fix AR degeneration/looping — see `resource-requirements.md` and `licensing.md` |

**Hard constraint: 3 physical A100s.** Past the 3rd concurrent worker, either
share a GPU index (accept contention) or run it non-concurrently. XTTS shares
index 0 with Tortoise: Tortoise's resident footprint is ~13-20 GB and XTTS's
is measured at ~2.2 GB, so both fit a 40 GB card with large headroom.
StyleTTS2 shares index 1 with CosyVoice2 for the same reason (~4.3 GB +
~2.0 GB measured). F5-TTS shares index 2 with MetaVoice (~10.4 GB +
~3.2 GB measured). MaskGCT joins index 1 as a *third* occupant alongside
CosyVoice2 and StyleTTS2 (~11.1 GB more on top of their ~6.4 GB combined);
Fish-Speech joins index 2 as a *third* occupant alongside MetaVoice and
F5-TTS (~2.2 GB more); SSR-Speech joins index 0 as a *third* occupant
alongside Tortoise and XTTS (~6.7 GB more), chosen specifically because GPU 1
and GPU 2 were already 3-way shares — making all three GPUs 3-way shares.
LLASA then joins index 2 as a *fourth* occupant (~9.6 GB more), the first
4-way share in this pipeline, chosen because GPU 2 had marginally more
headroom than GPU 1 at the time (GPU 0 was already nearly full from
SSR-Speech). All still fit with headroom, but concurrent load on any of them
measurably degrades per-worker latency (see `resource-requirements.md`). All
of this is accepted contention rather than non-concurrent scheduling. Update
this table whenever a system is added (see `adding-a-new-tts-system.md`) —
and check current `nvidia-smi` usage before picking a GPU for the next one,
since GPU 0 and GPU 2 are both getting tight.

## Why per-model containers

The upstream repos pin mutually incompatible dependency stacks (Tortoise uses
transformers 4.31 + torch 2.3.1; MetaVoice and SSR-Speech both pin torch 2.1.0
+ xformers 0.0.22(.post7); CosyVoice2 pins torch 2.3.1 with its own submodule;
XTTS and StyleTTS2 both need torch 2.5.1 — both leave transformers unpinned,
and pip resolves that to a 5.x release that silently disables its torch
backend under torch<2.4; F5-TTS needs the same torch 2.5.1 fix and separately
needed its unpinned `torchcodec` dependency force-removed since the version
pip resolves needs torch>=2.11; MaskGCT pins torch==2.0.1+cu118 and
transformers==4.41.2 exactly; Fish-Speech pins torch<=2.4.1, resolved to
exactly 2.4.1 to simultaneously satisfy that ceiling and transformers 5.x's
torch>=2.4 floor; LLASA pins torch==2.5.0 per xcodec2's own setup.py, plus a
downstream torchao re-pin and a transformers==4.46.3 downgrade — see
`resource-requirements.md`/Dockerfile comments for why). They cannot coexist
in one Python process, so each runs in its own image with its own
environment. The gateway never imports model code — it only enqueues jobs
and reads results.

## Portability to Kubernetes (not built here)

Design choices that translate directly later: each worker is a stateless
consumer of a named queue (→ a `Deployment` with `replicas: N`); GPU allocation
uses the long-form device reservation (→ `resources.requests: {nvidia.com/gpu: 1}`);
all config is env-driven; the shared outputs directory is the one piece of
shared state (→ a `PersistentVolumeClaim`); health is a real readiness probe.
