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
| xtts | MPL-2.0 | CPML 1.0.0 (**non-commercial only**) | Code: `coqui_tts-0.27.5.dist-info/licenses/LICENSE.txt` on disk in the built image = Mozilla Public License Version 2.0 (confirmed by `head`ing the file directly). Weights: `coqui/XTTS-v2` `LICENSE.txt` on HF — quoted directly: "Non-commercial purposes include... personal use for research and learning [and] testing by commercial entities for non-commercial R&D", "only so far as you do not receive any direct or indirect payment arising from the use of the model or its output" | 2026-07-13 (code LICENSE on disk; weights per HF model-card LICENSE.txt) |
| styletts2 | MIT | Not formally licensed (**disclosure condition**, not a legal restriction) | Code: `/opt/StyleTTS2/LICENSE` on disk in the built image = MIT, copyright Aaron (Yinghao) Li 2023 (confirmed by `head`ing the file directly). Weights: `yl4579/StyleTTS2-LibriTTS` on HF has no license tag/model card; usage condition instead comes from the GitHub README (quoted below) | 2026-07-13 (code LICENSE on disk; weights repo has no license file, README condition only) |
| f5-tts | MIT | CC-BY-NC-4.0 (**non-commercial only**) | Code: `f5_tts-1.1.21.dist-info/licenses/LICENSE` on disk in the built image = MIT, copyright Yushen CHEN 2024 (confirmed by `head`ing the file directly). Weights: `SWivid/F5-TTS` model-card license tag = `cc-by-nc-4.0`; README explains this is because the base model trained on Emilia, an in-the-wild non-commercially-licensed dataset | 2026-07-13 (code LICENSE on disk; weights per HF model-card license tag) |
| maskgct | MIT | CC-BY-NC-4.0 (**non-commercial only**) | Code: `/opt/Amphion/LICENSE` on disk in the built image = MIT, copyright Amphion 2023 (confirmed by `head`ing the file directly). Weights: `amphion/MaskGCT` model-card license tag = `cc-by-nc-4.0`; same Emilia-training rationale as F5-TTS | 2026-07-13 (code LICENSE on disk; weights per HF model-card license tag) |
| fish-speech | Apache-2.0 | CC-BY-NC-SA-4.0 (**non-commercial, share-alike**) | Pinned to the **v1.5.1 tag**, not current main (see note below). Code: `/opt/fish-speech/LICENSE` on disk in the built image = Apache License 2.0 (confirmed by `head`ing the file directly). Weights: `fishaudio/fish-speech-1.5` model-card license tag = `cc-by-nc-sa-4.0` | 2026-07-13 (code LICENSE on disk; weights per HF model-card license tag) |
| ssrspeech | MIT | CC-BY-NC-SA-4.0 (**non-commercial, share-alike**) | Code: `/opt/SSR-Speech/LICENSE` on disk in the built image = MIT, copyright Helin Wang 2024 (confirmed by `head`ing the file directly). Weights: `westbrook/SSR-Speech-English` model-card license tag = `cc-by-nc-sa-4.0` | 2026-07-14 (code LICENSE on disk; weights per HF model-card license tag) |
| llasa | No single repo license (xcodec2 package: MIT) | CC-BY-NC-4.0 (**non-commercial only**) | No cloned repo — see note below. `xcodec2` PyPI package (the only actual installed code artifact) declares MIT via its setup.py classifier. Weights: `HKUSTAudio/Llasa-1B` and `HKUSTAudio/xcodec2` model-card license tags both = `cc-by-nc-4.0` | 2026-07-14 (weights per HF model-card license tag; no on-disk repo LICENSE to check) |

## Verification status

Code licenses for nine of ten systems confirmed by reading the actual
`LICENSE` file inside each built image (tortoise/cosyvoice2/metavoice on
2026-07-12, all Apache-2.0; xtts on 2026-07-13, MPL-2.0; styletts2, f5-tts,
and maskgct on 2026-07-13, all MIT; fish-speech on 2026-07-13, Apache-2.0;
ssrspeech on 2026-07-14, MIT). LLASA has no cloned repo to check a LICENSE
file in — see below. Weights licenses are per each model's HF model card /
LICENSE.txt, except styletts2 whose weights repo has neither — see below.

## XTTS / CPML — read before using outside this project's research scope

XTTS-v2 is the first non-Apache-2.0 system in this pipeline. Its weights are
**Coqui Public Model License (CPML) 1.0.0**, which only permits non-commercial
use — no direct or indirect payment may arise from use of the model or its
output. This project's use (academic anti-spoofing research at UMich) falls
under the license's explicit "personal use for research and learning" and
"non-commercial R&D" carve-outs. It does **not** permit any commercial
deployment or product built on XTTS output without a separate license — there
is no current licensor to obtain one from, since Coqui Inc. shut down in
January 2024. Do not repurpose XTTS-generated audio for anything outside
non-commercial research without re-checking this.

## StyleTTS2 — disclosure condition, not a license restriction

StyleTTS2's code is MIT (confirmed on disk) and its LibriTTS weights carry no
separate license file at all — but the GitHub README imposes an ethical usage
condition on the pretrained models: "you agree to inform the listeners that
the speech samples are synthesized by the pre-trained models, unless you have
the permission to use the voice you synthesize." This is meaningfully
different from XTTS's CPML: it's a disclosure obligation, not a legal
commercial-use restriction. It's still recorded in the registry/metadata
`notes` field so it travels with every generated sample, same as CPML.

## F5-TTS and MaskGCT / CC-BY-NC-4.0 — read before using outside this project's research scope

Both F5-TTS's and MaskGCT's weights are **CC-BY-NC-4.0**, a standard Creative
Commons non-commercial license — simpler to reason about than CPML but the
same practical restriction: no commercial use of the model or its output.
Both were trained on Emilia, an in-the-wild dataset whose own license forces
this downstream restriction onto any model trained on it. This project's
academic research use is fine; do not repurpose F5-TTS- or
MaskGCT-generated audio commercially. Note the Amphion *codebase's* own
top-level notice says it's usable commercially — that claim is about the code
only and does not override this specific checkpoint's own CC-BY-NC-4.0 tag.

## Fish-Speech — pinned to v1.5.1, not current main

`fishaudio/fish-speech` has since rebranded to **"Fish Audio S2"**; its
current `main` branch LICENSE (dated 2026-03-07) is the **Fish Audio Research
License**, which restricts the *codebase itself* — not just weights/output —
to research or non-commercial use, and explicitly excludes "creating,
modifying, or distributing Your product or service" and "internal
operations" for any commercial entity. That's a materially stricter category
than every other system in this pipeline, none of which restrict the code
itself. We deliberately pinned to the **v1.5.1 tag** instead (May 2025) —
the version `selected_tts_systems.csv` actually refers to — which keeps the
familiar split-license shape: Apache-2.0 code (confirmed on disk) and
CC-BY-NC-SA-4.0 weights (still live and downloadable on HF as
`fishaudio/fish-speech-1.5`). This was a deliberate version choice, confirmed
with the user before implementation — see conversation history. Do not
"helpfully" update `upstream_commit` to a newer main-branch commit without
re-checking this license situation first.

## SSR-Speech — CC-BY-NC-SA-4.0, same category as Fish-Speech

SSR-Speech's weights are also CC-BY-NC-SA-4.0 (non-commercial, share-alike) —
same license, different lineage (not traceable to Emilia; this is its own
license choice by the authors). Code is MIT. Also worth noting: some code
under `./models/modules` and `data/tokenizer.py` is separately Apache-2.0
per the upstream README, and the `phonemizer` dependency is GPLv3 — none of
this affects this project's internal research use (GPL copyleft obligations
trigger on distribution, not internal use), but would matter if this
pipeline were ever redistributed. Separately, SSR-Speech supports an
optional audio watermark (`use_watermark` param, a built-in safety feature
from the paper) — left off by default here to match every other system's
unwatermarked output; turn it on deliberately if a specific use case wants
watermark-detectability as a signal.

## LLASA — no cloned repo, and a corrected fact from earlier planning

Earlier planning notes (before implementation) assumed LLaSA's weights were
CC-BY-NC-**ND**-4.0 (no derivatives). Verified directly against the actual
HF model-card YAML frontmatter during implementation: it's **CC-BY-NC-4.0**
(no "ND" clause) — same category as F5-TTS/MaskGCT, not the more restrictive
no-derivatives one. Also unlike every other system here, LLASA has no
upstream code repo to clone and pin: `LLaSA_training`
(github.com/zhenye234/LLaSA_training) is training/finetuning scripts only,
with no inference code at all. The runnable model is a standard HF
`transformers.AutoModelForCausalLM` checkpoint plus a separate small `xcodec2`
PyPI package (MIT) for the codec — `upstream_repo`/`upstream_commit` in the
registry pin the two HF model repos' revisions instead of a git commit.

The model card's own disclaimer is unusually blunt: "prohibits free
commercial use because of ethics and privacy concerns; detected violations
will result in legal consequences."

## Why this matters beyond the pilot

The pilot deliberately used only permissive (Apache-2.0) systems. XTTS was the
first onboarded system with a non-commercial weights license (CPML); StyleTTS2
added a different wrinkle (MIT code, but an undocumented/no-license weights
repo plus an informal disclosure condition); F5-TTS, MaskGCT, and LLASA all
add a third variant (CC-BY-NC-4.0 weights — a standard named license, LLASA's
own code license situation additionally complicated by having no cloned
repo); Fish-Speech and SSR-Speech both add a fourth (CC-BY-NC-SA-4.0
weights — the "SA" share-alike clause, from two unrelated lineages), and
Fish-Speech is also the first case where the *current upstream* moved to
something considerably more restrictive than the pinned version we actually
use. All of this is why the `notes` field exists, so this kind of nuance
travels with every sample rather than being assumed uniform.
