# License ledger

Per-system code and weights licenses. **Verify against each repo's actual
LICENSE file / model card, not just the README**, and record the date. The
registry (`shared/registry/tts_systems.yaml`) carries these same values; the
gateway copies them into every sample's metadata.

| System | Code license | Weights license | Source of truth | Verified |
|--------|--------------|-----------------|-----------------|----------|
| tortoise-tts | Apache-2.0 | Apache-2.0 | `LICENSE` in pinned repo (confirmed on disk = Apache 2.0); weights `Manmay/tortoise-tts` | 2026-07-12 (code LICENSE on disk) |
| cosyvoice2 | Apache-2.0 | Apache-2.0 | `/opt/CosyVoice/LICENSE` on disk = Apache 2.0; weights `FunAudioLLM/CosyVoice2-0.5B` | 2026-07-12 (code LICENSE on disk) |
| metavoice-1b | Apache-2.0 | Apache-2.0 | `/opt/metavoice/LICENSE` on disk = Apache 2.0; weights `metavoiceio/metavoice-1B-v0.1` | 2026-07-12 (code LICENSE on disk) |

## Verification status

Code licenses confirmed on 2026-07-12 by reading the actual `LICENSE` file
inside each built image (all three are Apache 2.0). Weights licenses are per
each HF model card and were not re-fetched from the card UI here — a reviewer
should confirm the HF model-card `license:` field before external distribution,
though all three projects release code and weights under Apache 2.0.

## Why this matters beyond the pilot

The pilot deliberately uses only permissive (Apache-2.0) systems. The remaining
candidate systems are not uniform — e.g. LLaSA is CC-BY-NC-ND (no derivatives,
no commercial), and F5-TTS / MaskGCT / Fish-Speech / XTTS carry non-commercial
weight licenses. When those are onboarded, the `notes` field should capture any
restriction (e.g. "no fine-tuning", "research-only") so it rides along in every
sample's metadata.
