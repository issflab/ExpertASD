# ExpertASD TTS Generation Pipeline

Generates synthetic/cloned speech across multiple open-source TTS systems to
feed an expert-in-the-loop audio **deepfake detection** system (defensive
anti-spoofing research). Each TTS system runs in its own container behind a
single gateway and a per-system job queue.

**Pilot scope:** 3 systems — Tortoise-TTS, CosyVoice2, MetaVoice-1B (all
zero-shot voice cloning, all Apache-2.0). **7 more onboarded:** XTTS-v2,
StyleTTS2, F5-TTS, MaskGCT, Fish-Speech, SSR-Speech, LLASA — all zero-shot,
all with some form of non-commercial or otherwise restricted weights license
(several different flavors: CPML, CC-BY-NC-4.0, CC-BY-NC-SA-4.0, and an
undocumented disclosure condition). **Read
[docs/licensing.md](docs/licensing.md) before using any generated audio
outside this project's academic research scope** — it has the full
per-system license ledger and the reasoning behind each. The pattern
generalizes to the remaining candidate systems in `selected_tts_systems.csv`;
see [docs/adding-a-new-tts-system.md](docs/adding-a-new-tts-system.md).

## Quick start

```bash
make init          # create .env + /data dirs (review .env after)
make build         # build images (long first run; see docs/resource-requirements.md)
make up            # start the stack
docker compose ps  # wait for all services "healthy"
make smoke         # end-to-end test across all systems
```

## Layout

- `services/gateway/` — FastAPI entrypoint (the only host-exposed service).
- `services/workers/<system>/` — one containerized worker per TTS system.
- `shared/python/expertasd_common/` — shared schemas, storage, queue, health.
- `shared/registry/tts_systems.yaml` — machine-readable system registry, incl.
  per-system `default_params` (server-side defaults the gateway applies).
- `config/client_params.yaml` — client-side per-system params the scripts send
  (override the registry defaults; `--param KEY=VALUE` overrides this per key).
- `shared/schemas/openapi.yaml` — API contract.
- `docs/` — architecture, metadata schema, licensing, runbook, onboarding,
  resource requirements.
- `scripts/` — smoke test, disk guard, weight pre-warm.

## API

`POST /v1/synthesize` → `{job_id, status_url}`; `GET /v1/jobs/{id}` for status
and result; `GET /v1/systems` for the registry + live worker health. Full
contract in [shared/schemas/openapi.yaml](shared/schemas/openapi.yaml).

## Docs

Start with [docs/architecture.md](docs/architecture.md) and
[docs/runbook.md](docs/runbook.md).
