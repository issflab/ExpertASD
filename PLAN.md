# Multi-TTS Deepfake-Generation Pipeline — 3-System Pilot

## Context

This repo (`/home/alhashim/ExpertASD`) currently holds only two reference files: `selected_tts_systems.csv` (15 candidate TTS systems with type/vocoder/difficulty/zero-shot metadata) and `tts_systems.txt`. The goal is to build a pipeline that generates synthetic/cloned speech using multiple open-source TTS systems, to feed an expert-in-the-loop audio deepfake detection system — this is defensive anti-spoofing research (the lab already runs `ssl_antispoofing`, `antideepfake`, `spoof_SUPERB` and similar projects on this same machine).

Rather than building all 15 systems and a full production deployment at once, we're validating the entire pattern — container wrapper, shared API contract, local orchestration, and handoff packaging — on **3 representative pilot systems** first: Tortoise-TTS, CosyVoice2, and MetaVoice-1B. All three are "Easy" implementation difficulty (official repo + official weights + permissive license) and support zero-shot voice cloning from a short reference clip, which is the core capability this pipeline needs. Once this pattern is proven end-to-end, adding the remaining 12 systems becomes a repeatable recipe rather than new architecture.

Confirmed decisions from discussion with the user:
1. Pilot with 3 systems, not all 15.
2. Test locally via a **full local stack** (docker-compose: gateway + queue + per-model workers), not just isolated per-container smoke tests.
3. Handoff stays **portable** (docker-compose-based) rather than committing to Kubernetes manifests, since the production team's target platform is unknown — but design choices should translate cleanly to k8s later.
4. **No expert review/labeling UI** in this pass — scope is generation + storage with rich metadata only.

## Verified environment facts

I independently checked the ground truth on this machine and against the three pilot repos rather than trusting claims at face value — one claimed fact (`/data/alhashim/` existing as a per-user root) turned out to be fabricated, and two of three repos' dependency pins were wrong as first reported. Corrected facts below:

- Host: `ISSFLab.engin.umd.umich.edu`. Docker 26.1.3, Compose v2.27.0, `nvidia` container runtime registered and working. 3x A100-PCIE-40GB, all idle.
- `/home` has ~12GB free (99% full) — usable only for source code, not weights or containers.
- `/data` is a **flat, shared, multi-tenant lab volume** (5.9TB, 70GB free, 99% full) — there is no per-user subdirectory convention; existing projects live directly at the top level (e.g. `/data/ssl_anti_spoofing`, `/data/ASD_app`, `/data/hf_cache`). This pipeline's data root should follow that same flat convention: **`/data/expertasd_tts_pipeline/`** (new top-level dir), not a nonexistent `/data/alhashim/...` path.
- `/data/hf_cache` already holds 354GB from other lab projects and is the existing `HF_HOME` — reuse it for model weights (content-addressed by repo id, safe to share) rather than creating a separate cache.
- No job scheduler, no passwordless sudo, `ffmpeg`/`sox`/`espeak-ng` already installed system-wide, internet egress open.
- **Per-repo facts, verified directly against each repo's current `requirements.txt`/`pyproject.toml`/README (not assumed):**
  - **Tortoise-TTS** (`neonbjb/tortoise-tts`, Apache-2.0): `transformers==4.31.0` (not 4.29.2 as first reported). `torch`/`torchaudio` are **unpinned** in requirements.txt — we choose a compatible modern version ourselves rather than matching a nonexistent pin. Zero-shot via `tts.tts_with_preset(text, voice_samples=[...], preset='fast')`.
  - **CosyVoice2** (`FunAudioLLM/CosyVoice`, Apache-2.0): confirmed `git clone --recursive` is required — `.gitmodules` shows exactly one submodule, `third_party/Matcha-TTS` → `shivammehta25/Matcha-TTS`. `requirements.txt` pins `torch==2.3.1`, `torchaudio==2.3.1`, `deepspeed==0.15.1`, `tensorrt-cu12==10.13.3.9` (+ bindings/libs), `gradio==5.4.0` — all Linux-only, inference-irrelevant extras we should trim from the container build. Weights via `snapshot_download('FunAudioLLM/CosyVoice2-0.5B', ...)`. Zero-shot via `cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_wav_path)` — **note this requires a transcript of the reference audio** (`prompt_text`), unlike the other two pilot systems. This is a real API-contract difference the schema must accommodate.
  - **MetaVoice-1B** (`metavoiceio/metavoice-src`, Apache-2.0 confirmed, "no restrictions"): Python `^3.10`, `torch^2.1.0`, GPU VRAM `>=12GB` stated explicitly, no Rust/maturin build step in `pyproject.toml`. Zero-shot via `tts.synthesise(text=..., spk_ref_path=...)`. Exact HF weight repo id not confirmed from the README excerpt — verify at implementation time.
- None of the 3 pilot systems compile custom CUDA kernels, so the host's older `gcc 8.5.0` is a non-issue (containers install their own toolchain).
- CosyVoice2's return shape from `inference_zero_shot` (a generator, exact chunk dict keys) and MetaVoice-1B's exact HF weight repo id are **not yet verified** — flagged as open risks to confirm during implementation, not asserted as fact.

## Repo structure

Initialize git now (`git init` + commit the two existing reference files as the first commit) — a multi-service pipeline with an explicit handoff obligation needs version history and commit-SHA pinning discipline from day one.

```
ExpertASD/
├── .git/  .gitignore  .env.example  README.md  Makefile
├── docker-compose.yml
├── selected_tts_systems.csv          # existing, kept as provenance
├── tts_systems.txt                   # existing, kept as provenance
├── docs/
│   ├── architecture.md               # topology + GPU-to-worker assignment table
│   ├── metadata-schema.md
│   ├── licensing.md                  # per-system license ledger (code vs weights)
│   ├── runbook.md                    # restart/health/disk-check/stuck-job triage
│   ├── adding-a-new-tts-system.md    # the 4th-system recipe
│   └── resource-requirements.md      # populated with MEASURED numbers before handoff
├── shared/
│   ├── registry/tts_systems.yaml     # machine-readable system registry
│   ├── schemas/openapi.yaml          # static API contract snapshot
│   └── python/expertasd_common/      # installable shared package
│       ├── schemas.py                # pydantic: SynthesizeRequest, JobResult, GenerationMetadata
│       ├── model_base.py             # ABC: TTSModel.load() / .synthesize()
│       ├── storage.py                # writes audio+metadata.json under DATA_ROOT/outputs/<job_id>/
│       ├── queue.py                  # RQ enqueue/status helpers
│       └── health.py                 # shared /health endpoint helper
├── services/
│   ├── gateway/{Dockerfile,requirements.txt,main.py}
│   └── workers/
│       ├── tortoise/{Dockerfile,requirements.txt,model.py,worker.py,entrypoint.sh}
│       ├── cosyvoice2/               # same skeleton, queue "cosyvoice2"
│       └── metavoice/                # same skeleton, queue "metavoice-1b"
├── scripts/
│   ├── smoke_test.py
│   ├── check_data_disk.sh            # fails fast if /data free space < threshold
│   └── warm_weights.sh
└── data_fixtures/smoke/
    ├── reference_female_en.wav
    ├── reference_female_en.txt       # transcript, required input for CosyVoice2 only
    └── smoke_texts.json
```

**Not in the repo**: model weights (reuse `/data/hf_cache`, `HF_HOME` already points there); generated audio + Redis persistence (new `/data/expertasd_tts_pipeline/{outputs,redis-data}/`).

## Common API contract

Defined once in `shared/schemas/openapi.yaml`, mirrored as pydantic models in `shared/expertasd_common/schemas.py` so gateway and every worker import the same types.

- **`POST /v1/synthesize`** (gateway only): `{tts_system, text, reference_audio_url?, reference_text?, params?, requested_by}` → `202 {job_id, status: "queued", status_url}`. Gateway validates `tts_system` against the registry and enforces `reference_text` presence when the registry marks `requires_reference_text: true` (true only for `cosyvoice2` among the pilot 3).
- **`GET /v1/jobs/{job_id}`** → `{job_id, tts_system, status, timestamps, error?, result: {audio_url, duration_sec, sample_rate, metadata}}`.
- **`GET /v1/systems`** → registry contents (id, ar_nar, tts_type, vocoder, zero_shot, requires_reference_text, license, worker health) — the machine-readable descendant of the CSV; this is what makes onboarding system #4 a config change, not new code.
- **Per-worker health contract** (port 8080, internal only): `GET /health` → `{"status": "loading"|"ready"|"error", "model": ..., "detail": ...}` — must only report `ready` after the model is actually resident in memory, since cold-start load times vary a lot across the 3 systems.
- **`GenerationMetadata`** written to every `outputs/<job_id>/metadata.json`: includes `tts_system`, pinned `tts_system_commit` SHA, `ar_nar`, `tts_type`, `vocoder`, a `license: {code_license, weights_license, license_verified_at, notes}` block (populated from day one even though all 3 pilot systems are Apache-2.0, since several of the other 12 systems are NC/ND and the schema must already support recording that), `zero_shot`, `reference_audio` provenance, `output` (path/sample_rate/duration/sha256), `generation` (latency, worker_host, gpu_index), and `pipeline_version` (this repo's git SHA).

## Per-model wrapper pattern (worked example: Tortoise-TTS)

- **Dockerfile** base: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (runtime, not devel — no custom kernel compilation needed). Install `python3.9 python3.9-venv python3-pip ffmpeg`, then `pip install transformers==4.31.0` + a modern compatible `torch`/`torchaudio` (choose and pin a specific version at implementation time, since upstream leaves this open), then `pip install "git+https://github.com/neonbjb/tortoise-tts.git@<pinned-sha>"` — **pin every upstream repo to a commit SHA, never a moving branch**. `COPY shared/python /shared_python && pip install /shared_python` for `expertasd_common`.
- **Weight loading**: do **not** bake weights into the image. `ENV HF_HOME=/data/hf_cache`; weights download on first container start into the bind-mounted cache volume, subsequent restarts are cache hits. This follows directly from the disk constraint — `/data` has only 70GB free and is shared lab-wide.
- **Health-check-only-when-loaded**: `entrypoint.sh` starts a background health thread reporting `loading`, then `worker.py` synchronously loads the model (triggers weight download/instantiation) and only flips to `ready` once that succeeds; on failure it reports `error` and exits non-zero so Compose can recycle it.
- **Critical RQ detail**: use **`rq.worker.SimpleWorker`**, not the default `Worker` — the default forks a child process per job, which would reload the GPU model (and reinitialize the CUDA context) on every single job. `SimpleWorker` reuses the once-loaded model across the container's lifetime. Documented tradeoff: a job that corrupts CUDA state degrades subsequent jobs until the container is restarted by its health check — accepted for pilot scope, flagged for the production team to revisit (e.g. replica recycling) later.

**CosyVoice2 and MetaVoice-1B follow the identical skeleton**, with these confirmed deltas:
- CosyVoice2: `git clone --recursive` for the `third_party/Matcha-TTS` submodule. Trim `requirements.txt` to an inference-only subset (drop `deepspeed`, `tensorrt-cu12*`, `gradio`) and verify empirically that `cosyvoice.cli.cosyvoice.CosyVoice2` still imports cleanly — if the package imports those unconditionally at module load, keep them and accept a larger image. `model.py.synthesize()` must reject requests missing `reference_text`. The exact return shape of `inference_zero_shot` (generator, chunk dict keys) needs confirming against live code, not just the call signature.
- MetaVoice-1B: reserve a full A100 given the stated `>=12GB` minimum plus JIT warm-up headroom on first call. Confirm the exact HF weight repo id at implementation time.

## Job queue: Redis + RQ

One **named queue per TTS system** (`tortoise-tts`, `cosyvoice2`, `metavoice-1b`) rather than one shared queue with routing logic — trivial to run in Compose (`redis:7-alpine`, `--appendonly yes` for persistence to `/data/expertasd_tts_pipeline/redis-data/`), self-documenting, naturally isolates GPU usage per system, and maps directly onto a Kubernetes `Deployment` with `replicas: N` later without redesign. Redis holds job/queue bookkeeping only — generated audio is always persisted independently to the output directory, so Redis is not the source of truth for artifacts.

## Gateway service

`services/gateway/main.py`, FastAPI, the **only** service exposed to the host (`localhost:8000`). Validates requests against the registry, writes an initial `metadata.json` skeleton, enqueues onto the system-named RQ queue, serves job status and generated audio (via `StaticFiles` mounted at `/outputs`) back to callers. All config via env vars (`REDIS_URL`, `DATA_ROOT`, `TTS_REGISTRY_PATH`, `GATEWAY_PORT`). Output storage: `/data/expertasd_tts_pipeline/outputs/<job_id>/{audio.wav, metadata.json}`. No automated retention/cleanup in this pilot (given `/data` is shared and 99% full, deleting other people's data by accident is a worse failure mode than manual cleanup) — `docs/runbook.md` documents a manual cleanup command, and `scripts/check_data_disk.sh` fails fast (wired into worker `entrypoint.sh`) if free space drops below a threshold before a weight download or write, protecting the shared volume.

## docker-compose.yml design

Services: `redis`, `gateway`, `worker-tortoise`, `worker-cosyvoice2`, `worker-metavoice`.
- GPU allocation via `deploy.resources.reservations.devices: [{driver: nvidia, device_ids: ['N'], capabilities: [gpu]}]`, one distinct GPU index per worker (0/1/2) — chosen over the `gpus:` shorthand because it maps conceptually onto Kubernetes' `resources.requests: {nvidia.com/gpu: 1}` later.
- Volumes: all workers mount `/data/hf_cache:/data/hf_cache` (shared weight cache) and `/data/expertasd_tts_pipeline/outputs:/data/outputs`; gateway mounts outputs read-write and the registry read-only.
- `.env` (gitignored) supplies `DATA_ROOT`, `HF_HOME`, `GATEWAY_PORT`, `REDIS_URL`, and an empty `HF_TOKEN=` wired to every worker even though none of the 3 pilot systems need it — zero-cost hook for later systems that may be gated.
- Healthchecks: `redis-cli ping` for Redis; `curl -f localhost:8000/v1/health` for gateway; `curl -f localhost:8080/health` per worker, with a generous `start_period` (~300s) to tolerate first-run weight downloads without Compose flapping into restart loops.
- `restart: unless-stopped` on all services; worker health ports not published to the host (debug via `docker compose exec ... curl`).

## Local test plan

1. `cp .env.example .env`, set `DATA_ROOT=/data/expertasd_tts_pipeline`, reuse existing `HF_HOME=/data/hf_cache`.
2. `mkdir -p /data/expertasd_tts_pipeline/{outputs,redis-data}`.
3. `docker compose build` — expect a non-trivial first build (CosyVoice2's torch/tensorrt pulls alone are multi-GB); monitor `/data` free space throughout given the thin 70GB margin on a shared volume.
4. `docker compose up -d`; tail logs per worker until each reports `ready`.
5. `docker compose ps` — confirm all services healthy.
6. `scripts/smoke_test.py`: for each of the 3 systems, POST a fixed sentence + `data_fixtures/smoke/reference_female_en.wav` (+ its transcript, consumed only by CosyVoice2) to `/v1/synthesize`, poll to terminal state, assert a valid non-empty WAV comes back with a complete `metadata.json` (including the license block).
7. Concurrency check: fire all 3 smoke requests together, confirm via `nvidia-smi` that 3 distinct GPU indices are simultaneously active.
8. Restart resilience: `docker compose restart worker-tortoise` mid-idle, confirm health cycles `loading → ready` without re-downloading weights (cache hit).
9. Record actual measured latency/VRAM/disk footprint per system in `docs/resource-requirements.md` — do not ship estimated numbers to the production handoff.

## "Add a 4th system" recipe (`docs/adding-a-new-tts-system.md`)

1. Read the target repo's actual requirements/README (not assumed) for Python version, key deps, exact zero-shot call signature, weight source, gating, and license (code **and** weights, verbatim).
2. Add an entry to `shared/registry/tts_systems.yaml`.
3. Copy the nearest-analog worker skeleton into `services/workers/<id>/`, implement `model.py` against the shared `TTSModel` ABC.
4. Pin the upstream repo to a specific commit SHA.
5. Wire weight caching under `/data/hf_cache/<id>`, download-on-first-start.
6. Add a compose service block — noting the hard constraint of only 3 physical GPUs: past the 3rd concurrent worker, either share a GPU index (accept contention) or run non-concurrently. Update the GPU-to-worker table in `docs/architecture.md`.
7. Extend `data_fixtures/smoke/` and `scripts/smoke_test.py` if the new system has different reference-audio conventions.
8. Build, verify health, smoke-test in isolation before adding to the full suite.
9. Record measured resource footprint and update `docs/licensing.md`.

## Handoff artifacts for the production team

`docker-compose.yml` + `.env.example`; the 4 Dockerfiles; `shared/schemas/openapi.yaml`; `shared/expertasd_common/` (schema/storage/health/queue library + `TTSModel` ABC); `shared/registry/tts_systems.yaml`; `docs/{architecture,metadata-schema,licensing,runbook,adding-a-new-tts-system,resource-requirements}.md` (resource doc populated with measured numbers); `scripts/smoke_test.py` and `scripts/check_data_disk.sh`. **Not** handed off: Kubernetes manifests, review/labeling UI, wrappers for the remaining 12 systems.

## Genuine open risks (flagged, not resolved by planning)

1. CosyVoice2's `inference_zero_shot` return shape (generator chunk dict keys) — confirmed only at call-signature level.
2. Whether CosyVoice2 unconditionally imports `deepspeed`/`tensorrt-cu12` at module load (affects whether the inference-only trim actually works) — untested.
3. MetaVoice-1B's exact HF weight repo id — unconfirmed.
4. No system's GPU memory footprint under concurrent/repeated load has been measured yet.
5. `/data` is shared and 99% full lab-wide — the pilot's estimated combined weight footprint should fit in the 70GB free margin, but it's thin and shared with other researchers; `check_data_disk.sh` mitigates but doesn't guarantee against contention from others' concurrent usage.
6. No commit SHAs chosen yet for the 3 upstream repos — pick and record at implementation time.
7. Tortoise-TTS ships no working torch/torchaudio pin — we must choose one and verify it resolves cleanly with `transformers==4.31.0`.
8. RQ's `SimpleWorker` trades away per-job crash isolation for avoiding GPU-model-reload-per-job — accepted for pilot scope, flagged for the production team to revisit under Kubernetes (e.g. replica recycling).

## Verification

- `docker compose config` validates the compose file without starting anything.
- Full local stack test plan (Section "Local test plan" above) is the end-to-end verification: build → start → health → smoke test all 3 systems individually and concurrently → restart resilience.
- Before declaring the pilot done: `docs/resource-requirements.md` must contain measured (not estimated) latency/VRAM/disk numbers, and `docs/licensing.md` must have the license ledger filled in and double-checked against each repo's actual LICENSE file (not just README claims), given how many claims required correction during this planning pass.

### Critical files
- `/home/alhashim/ExpertASD/docker-compose.yml`
- `/home/alhashim/ExpertASD/shared/expertasd_common/schemas.py`
- `/home/alhashim/ExpertASD/shared/registry/tts_systems.yaml`
- `/home/alhashim/ExpertASD/services/gateway/main.py`
- `/home/alhashim/ExpertASD/services/workers/tortoise/worker.py`
