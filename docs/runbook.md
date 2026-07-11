# Runbook

## Start / stop

```bash
make init          # once: create .env and /data dirs
make build         # build all images (long first time — see resource-requirements.md)
make up            # start detached
make ps            # service status
make logs          # follow logs
make down          # stop
```

## Health and readiness

Each worker serves `/health` on port 8080 (internal only). It reports `loading`
until the model is resident in GPU memory, then `ready`. Compose healthchecks
have a 600s `start_period` to tolerate first-run weight downloads.

```bash
docker compose ps                              # look for "healthy"
docker compose exec worker-cosyvoice2 curl -s localhost:8080/health
curl -s localhost:8000/v1/systems | python3 -m json.tool   # per-system health via gateway
```

## Stuck or failed jobs

Jobs have a 600s timeout (`JOB_TIMEOUT_SEC` in `expertasd_common/queue.py`). A
failed job records the error in its `metadata.json` and in RQ. Inspect:

```bash
curl -s localhost:8000/v1/jobs/<job_id> | python3 -m json.tool
cat /data/expertasd_tts_pipeline/outputs/<job_id>/metadata.json
```

**SimpleWorker caveat:** workers do not fork per job (needed to avoid reloading
the GPU model every job). A job that corrupts CUDA state can degrade later jobs
on that worker until it is recycled:

```bash
docker compose restart worker-<system>
```

The healthcheck will also recycle a worker whose process has exited.

## Disk (shared, near-full volume)

`/data` is a shared lab volume (~70GB free of 5.9TB). Each worker runs
`check_data_disk.sh` at startup and refuses to start if free space is below
`DATA_DISK_MIN_GB` (default 5). Monitor during first-run downloads:

```bash
watch -n 30 df -h /data
```

## Output retention (manual)

There is no automatic deletion — accidental deletion on a shared volume is worse
than manual cleanup. To prune old samples deliberately:

```bash
# Dry run first:
find /data/expertasd_tts_pipeline/outputs -mindepth 1 -maxdepth 1 -type d -mtime +30
# Then delete:
find /data/expertasd_tts_pipeline/outputs -mindepth 1 -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +
```

## Weights cache

Weights live under `/data/hf_cache/expertasd_models/` (Tortoise, CosyVoice2) and
the standard HF cache under `/data/hf_cache` (MetaVoice via `snapshot_download`).
Deleting these forces a re-download on next start.
