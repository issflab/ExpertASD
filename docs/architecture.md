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
   /data/.../outputs/<job_id>/    ┌──────────────┬──────────────┬──────────────┐
     audio.wav + metadata.json    │ worker-      │ worker-      │ worker-      │
          ▲                       │ tortoise     │ cosyvoice2   │ metavoice    │
          └───────────────────────┤ queue:       │ queue:       │ queue:       │
              writes results      │ tortoise-tts │ cosyvoice2   │ metavoice-1b │
                                  │ GPU 0        │ GPU 1        │ GPU 2        │
                                  └──────────────┴──────────────┴──────────────┘
                                         all mount /data/hf_cache (weights, ro-ish)
```

The gateway is the only host-exposed service. Workers are reachable only on the
internal compose network; their `/health` ports (8080) are not published.

## GPU-to-worker assignment

| Worker            | Queue          | GPU index | Notes                              |
|-------------------|----------------|-----------|------------------------------------|
| worker-tortoise   | tortoise-tts   | 0         | Tortoise-TTS, diffusion (slow)     |
| worker-cosyvoice2 | cosyvoice2     | 1         | requires reference_text            |
| worker-metavoice  | metavoice-1b   | 2         | torch.compile warm-up on 1st call  |

**Hard constraint: 3 physical A100s.** Adding a 4th system means either sharing a
GPU index (accept contention) or running it non-concurrently. Update this table
whenever a system is added (see `adding-a-new-tts-system.md`).

## Why per-model containers

The three upstream repos pin mutually incompatible dependency stacks (Tortoise
uses transformers 4.31 + torch 2.3.1; MetaVoice pins torch 2.1.0 + xformers
0.0.22; CosyVoice2 pins torch 2.3.1 with its own submodule). They cannot coexist
in one Python process, so each runs in its own image with its own environment.
The gateway never imports model code — it only enqueues jobs and reads results.

## Portability to Kubernetes (not built here)

Design choices that translate directly later: each worker is a stateless
consumer of a named queue (→ a `Deployment` with `replicas: N`); GPU allocation
uses the long-form device reservation (→ `resources.requests: {nvidia.com/gpu: 1}`);
all config is env-driven; the shared outputs directory is the one piece of
shared state (→ a `PersistentVolumeClaim`); health is a real readiness probe.
