# Resource requirements

> **STATUS: MEASURED.** All three rows measured on the target box (ISSFLab,
> A100-40GB, one GPU per worker). Measure with: `nvidia-smi` (VRAM), the
> `latency_sec` field in metadata, `docker images` (image size), and `du -sh`
> on the weight cache dirs.

## Per-system

| System | Image size | Weights on disk | VRAM (model resident) | Cold-start (load→ready) | Latency / short sentence |
|--------|-----------|-----------------|-----------------------|-------------------------|--------------------------|
| tortoise-tts | 9.66 GB | 4.0 GB | ~13–20 GB (GPU 0) | ~50 s (incl. first-run weight download) | ~25 s (`fast` preset, ~4.6 s output) |
| cosyvoice2 | 11.1 GB | 4.6 GB | ~4.3 GB (GPU 1) | ~90 s (incl. weight download) | ~4.5 s (~4.3 s output) |
| metavoice-1b | 10.4 GB | 4.8 GB | ~10.4 GB (GPU 2) | ~5.5 min (weight download + ~147 s torch.compile warm-up at load) | ~22 s first call, ~6.5 s warm |

Notes:
- Tortoise latency is dominated by diffusion sampling even at the `fast` preset;
  ~25 s for a one-sentence clone is expected, not a fault.
- CosyVoice2 is by far the fastest (flow-matching) and lightest on VRAM.
- MetaVoice pays a one-time ~147 s `torch.compile` warm-up during model load
  (not per request); warm latency drops to ~6.5 s. It emits 48 kHz output.

## Aggregate

- Combined weights footprint: ~13.4 GB across the three systems in the shared
  `/data/hf_cache`.
- All three fit one-model-per-GPU on 3× A100-40GB with large headroom (max
  resident ~20 GB on a 40 GB card).
- Concurrent 3-GPU run confirmed: all three pass simultaneously via
  `smoke_test.py --concurrent`, one model resident per GPU (indices 0/1/2).

## Reference-audio length constraints (important)

The systems disagree on reference-clip length, which the smoke test accounts for:
- **MetaVoice-1B** requires **≥30 s** of reference audio.
- **CosyVoice2** rejects reference audio **> 30 s**, and requires a transcript
  of the reference (`reference_text`).
- **Tortoise** is flexible (short clips fine).

`smoke_test.py` routes MetaVoice to a ~39 s fixture and the others to a ~20 s
fixture. When supplying your own clips, respect these bounds.
