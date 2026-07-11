# Resource requirements

> **STATUS: PARTIAL.** tortoise-tts row below is MEASURED on the target box
> (ISSFLab, A100-40GB). cosyvoice2 and metavoice-1b are not yet built/measured.
> Measure with: `nvidia-smi` (VRAM), the `latency_sec` field in metadata,
> `docker images` (image size), `du -sh /data/hf_cache/expertasd_models/*`.

## Per-system

| System | Image size | Weights on disk | VRAM (model resident) | Cold-start (load→ready) | Latency / short sentence |
|--------|-----------|-----------------|-----------------------|-------------------------|--------------------------|
| tortoise-tts | 9.66 GB | 4.0 GB | ~13.3 GB (GPU 0) | ~50 s (incl. first-run weight download) | ~25 s warm, ~29 s first call (`fast` preset, ~4.6 s output) |
| cosyvoice2 | TBD | TBD | TBD | TBD | TBD (flow-matching; expect single-digit seconds) |
| metavoice-1b | TBD | TBD | TBD | TBD (first call includes torch.compile warm-up) | TBD |

Notes: tortoise latency is dominated by diffusion sampling even at the `fast`
preset; ~25 s for a one-sentence clone is expected, not a fault. The 13.3 GB
resident footprint leaves ample headroom on a 40 GB A100.

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
