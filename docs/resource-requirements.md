# Resource requirements

> **STATUS: NOT YET MEASURED.** The numbers below are placeholders and MUST be
> replaced with real measurements from a build+smoke-test run on the target box
> before production handoff. Do not ship estimates. Measure with:
> `nvidia-smi` (VRAM per worker under load), the `latency_sec` field in each
> sample's metadata, `docker system df` / image inspect (image size), and
> `du -sh /data/hf_cache/expertasd_models/*` (weights on disk).

## Per-system (to be filled in)

| System | Image size | Weights on disk | Idle VRAM | VRAM under load | Cold-start (load) | Latency / short sentence |
|--------|-----------|-----------------|-----------|-----------------|-------------------|--------------------------|
| tortoise-tts | TBD | TBD | TBD | TBD | TBD | TBD (diffusion; expect tens of seconds even at `fast`) |
| cosyvoice2 | TBD | TBD | TBD | TBD | TBD | TBD (flow-matching; expect single-digit seconds) |
| metavoice-1b | TBD | TBD | TBD | TBD | TBD (first call includes torch.compile warm-up) | TBD |

## Aggregate

- Combined weights footprint on `/data`: TBD (budget against the ~70GB free on
  the shared volume before starting).
- Build-time peak Docker storage: TBD.
- Concurrent 3-GPU run confirmed: TBD (via `nvidia-smi` during
  `smoke_test.py --concurrent`).

## Measurement checklist

- [ ] Image sizes: `docker images | grep expertasd`
- [ ] Weights: `du -sh /data/hf_cache/expertasd_models/* /data/hf_cache/models--metavoiceio*`
- [ ] Per-worker VRAM under load: `nvidia-smi` during a synthesis
- [ ] Cold-start: time from container start to `/health` = ready
- [ ] Latency: `generation.latency_sec` from metadata for the standard smoke sentence
- [ ] Concurrency: 3 distinct GPU indices busy during `--concurrent`
