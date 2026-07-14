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

The 3 pilot systems are all Apache-2.0, but every system onboarded since
carries some non-commercial or restricted weights license — XTTS (CPML),
StyleTTS2 (no formal license, just a disclosure condition), F5-TTS/MaskGCT/
LLASA (CC-BY-NC-4.0), and Fish-Speech/SSR-Speech (CC-BY-NC-SA-4.0, from two
unrelated lineages) — the block exists from day one precisely so this
kind of difference is recorded per sample rather than assumed uniform. See
`licensing.md` for the full ledger and reasoning behind each. Several of
the remaining candidate systems (see
`../selected_tts_systems.csv`) carry non-commercial (CC-BY-NC) or no-derivative
(CC-BY-NC-ND) terms too. Recording code/weights license per sample keeps
generated audio auditable — a downstream consumer can filter by license
without re-deriving it. See `licensing.md`.
