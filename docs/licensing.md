# License ledger

Per-system code and weights licenses. **Verify against each repo's actual
LICENSE file / model card, not just the README**, and record the date. The
registry (`shared/registry/tts_systems.yaml`) carries these same values; the
gateway copies them into every sample's metadata.

| System | Code license | Weights license | Source of truth | Verified |
|--------|--------------|-----------------|-----------------|----------|
| tortoise-tts | Apache-2.0 | Apache-2.0 | github.com/neonbjb/tortoise-tts `LICENSE`; weights `Manmay/tortoise-tts` model card | PENDING on-box verification |
| cosyvoice2 | Apache-2.0 | Apache-2.0 | github.com/FunAudioLLM/CosyVoice `LICENSE`; `FunAudioLLM/CosyVoice2-0.5B` model card | PENDING on-box verification |
| metavoice-1b | Apache-2.0 | Apache-2.0 | github.com/metavoiceio/metavoice-src `LICENSE`; `metavoiceio/metavoice-1B-v0.1` model card | PENDING on-box verification |

## Verification status

Values above come from README/model-card statements gathered during planning.
Before production handoff, each row's "Verified" cell MUST be replaced with a
date and confirmed against the actual `LICENSE` file in the pinned commit and
the HF model card `license:` field. The pinned commit SHAs are in the registry.

## Why this matters beyond the pilot

The pilot deliberately uses only permissive (Apache-2.0) systems. The remaining
candidate systems are not uniform — e.g. LLaSA is CC-BY-NC-ND (no derivatives,
no commercial), and F5-TTS / MaskGCT / Fish-Speech / XTTS carry non-commercial
weight licenses. When those are onboarded, the `notes` field should capture any
restriction (e.g. "no fine-tuning", "research-only") so it rides along in every
sample's metadata.
