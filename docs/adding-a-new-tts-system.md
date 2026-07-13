# Adding a 4th TTS system

The pilot pattern is designed so onboarding another system is a repeatable
recipe, not new architecture. Work through these steps.

## 1. Read the actual source (not memory, not just the README)

For the target repo, confirm from the pinned files:
- Python version and key dependency pins (torch especially).
- Whether any dependency compiles from source at install time (would force a
  `devel` CUDA base image instead of `runtime`).
- The exact zero-shot / cloning call signature and what it returns (tensor?
  file path? generator? sample rate source?).
- Where weights come from (HF repo id? ModelScope? a custom cache dir that
  ignores `HF_HOME`, like Tortoise's `TORTOISE_MODELS_DIR`?).
- Whether a transcript of the reference audio is required (like CosyVoice2).
- Code license AND weights license, verbatim, from the LICENSE file + model card.

## 2. Register it

Add an entry to `shared/registry/tts_systems.yaml` with `queue`, `worker_host`,
`ar_nar`, `tts_type`, `vocoder`, `zero_shot`, `requires_reference_text`,
`upstream_repo`, `upstream_commit` (a specific SHA — never a branch), the
`license` block, and an optional `default_params` block (per-system generation
parameters the gateway merges under each request; request params override them
per key). Only keys the system's `model.py` reads have any effect.

## 3. Create the worker

Copy the nearest-analog worker directory under `services/workers/<id>/`:
- `model.py` — implement `TTSModel.load()` and `.synthesize()` (see
  `shared/python/expertasd_common/model_base.py`). `synthesize` must write a WAV
  to `out_path` and return `SynthOutput(sample_rate, duration_sec)`.
- `worker.py` — copy verbatim, change `QUEUE_NAME`. Keep `SimpleWorker`.
- `entrypoint.sh` — copy verbatim.
- `Dockerfile` — pin the commit via `ARG`, install the repo's own deps, point
  weight caching at a `/data/hf_cache/...` subdir, `COPY shared/python` and
  `pip install` it.

## 4. Wire into compose

Add a service block mirroring an existing worker. **Only 3 GPUs exist** — past
the 3rd concurrent worker, either share a `device_ids` index (accept contention)
or run the new system non-concurrently. Update the GPU table in
`architecture.md`.

## 5. Fixtures and smoke test

If the system needs a different reference convention (e.g. a longer minimum
clip, or a transcript), extend `data_fixtures/smoke/`. The smoke test discovers
systems from `/v1/systems` automatically, so no code change is needed there
unless the reference handling differs.

## 6. Verify before merging

```bash
docker compose build worker-<id>
docker compose up -d worker-<id>
docker compose exec worker-<id> curl -s localhost:8080/health   # wait for ready
python3 scripts/smoke_test.py --system <id>
```

## 7. Record

Add measured VRAM/latency/disk to `resource-requirements.md` and verify the
license row in `licensing.md`.
