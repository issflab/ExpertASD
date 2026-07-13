# Metadata schema

Every generated sample is stored at
`$DATA_ROOT/outputs/<job_id>/metadata.json` alongside `audio.wav`. The schema is
defined in `shared/python/expertasd_common/schemas.py` (`GenerationMetadata`).
The gateway writes an initial skeleton at enqueue time (status `queued`); the
worker fills in `output` and `generation` and sets status `succeeded`/`failed`.

## Fields

| Field | Meaning |
|-------|---------|
| `job_id` | UUID4, also the output directory name |
| `tts_system` | Registry id (e.g. `cosyvoice2`) |
| `tts_system_commit` | Pinned upstream commit SHA the audio was produced with |
| `ar_nar` | `AR` or `NAR` |
| `tts_type` | List, e.g. `["flow", "llm"]` |
| `vocoder` | Vocoder name |
| `license` | `{code_license, weights_license, license_verified_at, notes}` |
| `zero_shot` | Whether the sample used reference-audio cloning |
| `reference_audio` | `{provided, source, reference_text, sha256}` |
| `text` | Input text synthesized |
| `params` | Effective system-specific knobs actually used (registry `default_params` merged with the request's params, request winning per key) |
| `output` | `{path, sample_rate, duration_sec, sha256}` |
| `generation` | `{requested_at, latency_sec, worker_host, gpu_index}` |
| `pipeline_version` | This repo's version/SHA (from `PIPELINE_VERSION`) |
| `status` | `queued` / `running` / `succeeded` / `failed` |
| `error` | Set on failure |

## Why the license block is always present

All three pilot systems are Apache-2.0, so the block is uniform today. It exists
from day one because several of the other 12 candidate systems (see
`../selected_tts_systems.csv`) carry non-commercial (CC-BY-NC) or no-derivative
(CC-BY-NC-ND) terms. Recording code/weights license per sample keeps generated
audio auditable when those systems are onboarded — a downstream consumer can
filter by license without re-deriving it. See `licensing.md`.
