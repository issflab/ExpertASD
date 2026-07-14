# Resource requirements

> **STATUS: MEASURED.** All ten rows measured on the target box (ISSFLab,
> A100-40GB). Measure with: `nvidia-smi` (VRAM), the `latency_sec` field in
> metadata, `docker images` (image size), and `du -sh` on the weight cache dirs.

## Per-system

| System | Image size | Weights on disk | VRAM (model resident) | Cold-start (load→ready) | Latency / short sentence |
|--------|-----------|-----------------|-----------------------|-------------------------|--------------------------|
| tortoise-tts | 9.66 GB | 4.0 GB | ~13–20 GB (GPU 0) | ~50 s (incl. first-run weight download) | ~25 s (`fast` preset, ~4.6 s output) |
| cosyvoice2 | 11.1 GB | 4.6 GB | ~4.3 GB (GPU 1) | ~90 s (incl. weight download) | ~4.5 s (~4.3 s output) |
| metavoice-1b | 10.4 GB | 4.8 GB | ~10.4 GB (GPU 2) | ~5.5 min (weight download + ~147 s torch.compile warm-up at load) | ~22 s first call, ~6.5 s warm |
| xtts | 9.89 GB | 1.8 GB | ~2.2 GB (GPU 0, shared with tortoise-tts) | ~22 min on one measured run (see note below) | ~2–3.5 s warm |
| styletts2 | 10.4 GB | 736 MB | ~2.0 GB (GPU 1, shared with cosyvoice2) | ~14 s (incl. weight download) | ~14.6 s first call, ~0.2–0.6 s warm |
| f5-tts | 10.4 GB | 1.4 GB | ~3.2 GB (GPU 2, shared with metavoice-1b) | ~12 s measured warm-cache restart (cold download not separately re-measured; comparable ~1.4 GB to StyleTTS2's, likely well under a minute on a normal network) | ~0.8–2.2 s warm |
| maskgct | 9.99 GB | 8.4 GB (6.2 GB checkpoints + 2.2 GB `facebook/w2v-bert-2.0` encoder) | ~11.1 GB (GPU 1, shared with cosyvoice2 + styletts2) | ~58 s (incl. weight download across 6 separate HF downloads) | ~23.6 s first call, ~7.6 s warm |
| fish-speech | 10.9 GB | 1.4 GB | ~2.2 GB (GPU 2, shared with metavoice-1b + f5-tts) | ~20 s (incl. weight download) | ~11.4 s first call, ~5.5 s warm |
| ssrspeech | 22 GB | 12 GB (English.pth + wmencodec.th + WhisperX align/VAD models, all in the shared HF hub cache) | ~6.7 GB (GPU 0, shared with tortoise-tts + xtts) | ~31 s (incl. weight download) | ~19.4 s first call, ~4-5 s warm |
| llasa | 9.72 GB | 5.7 GB new (Llasa-1B 2.6 GB + xcodec2 3.1 GB) + 2.2 GB `facebook/w2v-bert-2.0` (reused from MaskGCT's cache) | ~9.6 GB (GPU 2, shared with metavoice-1b + f5-tts + fish-speech) | not cleanly isolated across debug iterations; warm-cache restarts under 30 s | highly variable, ~4-30 s (see note below) |

Notes:
- Tortoise latency is dominated by diffusion sampling even at the `fast` preset;
  ~25 s for a one-sentence clone is expected, not a fault.
- CosyVoice2 is by far the fastest (flow-matching) and lightest on VRAM.
- MetaVoice pays a one-time ~147 s `torch.compile` warm-up during model load
  (not per request); warm latency drops to ~6.5 s. It emits 48 kHz output.
- **XTTS's cold-start download was measured at ~22 minutes**, far slower than
  the other three (~50s-90s for comparable or larger weight sizes). Root
  cause: coqui-tts's own model downloader is a single-threaded, apparently
  non-resumable HTTP GET (observed throughput fluctuating ~150-250 KiB/s for
  most of the transfer, then jumping to ~85 MiB/s in the last ~15%) — unlike
  the other three systems, which use `huggingface_hub`'s optimized
  downloader. This may partly reflect transient shared-network conditions at
  measurement time rather than a fixed cost; re-measure on a second cold
  pull before treating 22 min as representative. Because of this, `docker-
  compose.yml` gives `worker-xtts` its own healthcheck with a 1800s
  `start_period` (vs. the other workers' 600s) so Compose doesn't mark it
  unhealthy mid-download. Warm latency (2-3.5s) is unaffected and is
  competitive with CosyVoice2/MetaVoice.
- XTTS is small enough on VRAM (~2.2 GB) that it shares GPU 0 with Tortoise
  rather than getting a dedicated index — see `architecture.md`.
- StyleTTS2 is the smallest and fastest system so far — a compact diffusion
  model (~736 MB checkpoint) with a single-digit-second cold start and
  sub-second warm latency. First-call latency (~14.6s) is dominated by
  one-time phonemizer/espeak-ng backend initialization, not the model itself.
  It shares GPU 1 with CosyVoice2 rather than getting a dedicated index.
- **F5-TTS's output duration is a function of the reference audio/text
  pairing**, not just the generated text: it estimates duration as
  `ref_audio_len + ref_audio_len / ref_text_len * gen_text_len / speed` (a
  speaking-rate ratio extrapolated from the reference). With the default
  smoke fixture (`reference_female_en.wav`, an espeak-synthesized clip whose
  transcript is unusually dense for its length) this measurably under-predicts
  duration — the fixed smoke sentence comes out ~2.8s, vs. 4-7s on every other
  system for the same text. Confirmed as a fixture-pairing artifact, not an
  integration bug: the same sentence against a real-speech reference (a Trump
  clip) produced a normal ~4.0s. Prefer natural-speech reference clips with
  accurate transcripts for this system; treat pacing from synthetic/espeak
  references with suspicion.
- **F5-TTS's `torchcodec` dependency is unpinned and incompatible with our
  torch pin** (resolves to 0.14.0, which needs torch>=2.11 and a CUDA-13
  `libnvrtc`) — it crashed loading some reference WAVs with "Could not load
  libtorchcodec" while others worked, since torchaudio only reaches for it as
  an optional backend for certain files. Fixed by uninstalling it in the
  Dockerfile; f5-tts's own code never calls it directly, so torchaudio falls
  back to its standard backend with no loss of function.
- MaskGCT is the heaviest system architecturally — six separate model
  components (semantic model, semantic codec, acoustic codec encoder/decoder,
  t2s, two s2a stages) plus a `facebook/w2v-bert-2.0` speech encoder loaded
  simultaneously, hence the largest resident VRAM (~11.1 GB) and slowest warm
  latency (~7.6s) of the non-Tortoise systems. It shares GPU 1 as a *third*
  occupant alongside CosyVoice2 and StyleTTS2 (a 3-way share) rather than a
  pair; under `smoke_test.py --concurrent` this visibly degrades latency for
  all three GPU-1 occupants (e.g. CosyVoice2 ~3.8s→~12.6s) — expected GPU
  compute contention under genuine concurrent load on one card, not a bug,
  and all still complete well within the job timeout.
- **Fish-Speech is the only system that outputs 44.1 kHz** (every other
  system emits 24 kHz or MetaVoice's 48 kHz) — its Firefly VQGAN decoder's
  native sample rate. `SynthOutput.sample_rate` carries this through
  correctly; no special handling needed downstream, but worth knowing if
  comparing output files across systems. It shares GPU 2 as a *third*
  occupant alongside MetaVoice and F5-TTS; the same 3-way contention pattern
  as GPU 1 applies — under `smoke_test.py --concurrent`, MetaVoice's latency
  on that GPU jumped from its usual ~6-7s to ~23.6s. Expected, not a bug.
- **SSR-Speech is architecturally a speech-*editing* model repurposed for
  zero-shot TTS**, not a from-scratch TTS design — it appends a mask right
  after a forced-aligned reference clip and lets an AR transformer fill it
  in, then trims the prompt back off the generated audio. This means every
  request runs a multi-pass pipeline internally (transcribe → align → cut
  reference at a word boundary → re-transcribe → generate → re-transcribe →
  trim), via WhisperX, regardless of any caller-supplied `reference_text`
  (which is accepted but ignored — see the reference-length table below).
  **Output is 16 kHz**, the lowest of any system in this pipeline (everything
  else is 24/44.1/48 kHz) — SSR-Speech's native/only `codec_audio_sr`. It
  shares GPU 0 as a *third* occupant alongside Tortoise and XTTS (the first
  system placed on GPU 0 past a pair), balancing load since GPU 1 and GPU 2
  were already 3-way shares before this — all three GPUs are now 3-way
  shares. It also supports an optional audio watermark (`use_watermark`
  param); left off by default to match every other system's output — see
  `licensing.md`.
- **LLASA originally had no repetition control on its `generate()` call at
  all — upstream's own example doesn't set one either.** This let generation
  degenerate: audible as a held/stuttered final sound (e.g. "immediately" ->
  "...eeeeeee") and/or needlessly long output. Confirmed via real usage with
  longer reference clips (`trump_ff_v2`, ~6-9s) that this — not reference
  length — was the actual cause of the wild duration variance (5.16s to
  35.52s for the identical smoke sentence, and near-zero output in one
  isolated case). **Fixed by adding `repetition_penalty` (default 1.2) to
  `model.generate()`** — not present upstream. After the fix, 3 repeated runs
  of the same sentence against a `trump_ff_v2` reference landed at 3.1s,
  4.0s, and 4.4s — a normal, tight range. The earlier claim that LLASA
  strictly requires a short (~4s) reference clip was **wrong** — retracted;
  see the corrected note in the reference-length section below. Also:
  **upstream's own default example doesn't trim the echoed prompt from
  output at all** (the trim line is commented out in their README) — this
  worker's `model.py` trims it by default (`trim_prompt: true`) for
  consistency with every other system's "output is just the target text"
  convention; the 4th occupant on GPU 2, chosen over GPU 0/1 for having
  (marginally) more headroom at implementation time — GPU 2 is now the
  tightest of the three (~14 GB free) after this addition.

## Aggregate

- Combined weights footprint: ~44.7 GB across the ten systems in the shared
  `/data/hf_cache`.
- All three GPUs have multiple occupants: GPU 0 (Tortoise + XTTS + SSR-Speech,
  ~28.4 GB combined resident, 3-way), GPU 1 (CosyVoice2 + StyleTTS2 + MaskGCT,
  ~17.5 GB combined resident, 3-way), GPU 2 (MetaVoice + F5-TTS + Fish-Speech
  + LLASA, ~26.6 GB combined resident, **4-way** — the first 4-way share in
  this pipeline). All still leave some headroom, but check current
  `nvidia-smi` usage before adding another occupant to any GPU — GPU 0 has
  ~8 GB free and GPU 2 has ~14 GB free, the two tightest.
- Concurrent run confirmed: all ten pass simultaneously via
  `smoke_test.py --concurrent`, including the 4-way GPU-2 share (with the
  expected latency degradation under concurrent load noted throughout this
  document).

## Reference-audio length constraints (important)

**Quick reference — which systems require `reference_text`:**

| System | `reference_text` required? | `--reference-metadata` needed for the scripts? |
|--------|------------------------------|-------------------------------------------------|
| tortoise-tts | No | No |
| cosyvoice2 | **Yes** | **Yes** |
| metavoice-1b | No | No |
| xtts | No (optional, recommended) | No, but improves quality if passed |
| styletts2 | No | No |
| f5-tts | No (optional; auto-transcribes via Whisper if omitted) | No, but strongly recommended — see pacing note below |
| maskgct | **Yes** | **Yes** |
| fish-speech | **Yes** | **Yes** |
| ssrspeech | No (always auto-transcribed internally; caller-supplied value is ignored) | No |
| llasa | **Yes** | **Yes** |

This is the same `requires_reference_text` flag in
`shared/registry/tts_systems.yaml` and served live at `GET /v1/systems`; the
gateway enforces it and returns `400 Bad Request` with an explicit message if
you submit a request to a "Yes" system without `reference_text`. For
`scripts/generate_from_dir.py` and `scripts/generate_from_metadata.py`, that
means you must pass `--reference-metadata <csv>` (a filename→transcript CSV
covering your reference pool) when targeting one of the "Yes" systems — see
each script's own `--help`/docstring for the exact flag usage and examples.

The systems disagree on reference-clip length, which the smoke test accounts for:
- **MetaVoice-1B** requires **≥30 s** of reference audio.
- **CosyVoice2** rejects reference audio **> 30 s**, and requires a transcript
  of the reference (`reference_text`).
- **Tortoise** is flexible (short clips fine).
- **XTTS** is flexible too — upstream docs recommend as little as 3-6 s;
  tested successfully against the same ~20 s fixture as Tortoise/CosyVoice2.
- **StyleTTS2** is flexible too (upstream demo uses clips as short as a few
  seconds); tested successfully against the same ~20 s fixture. No transcript
  needed.
- **F5-TTS** recommends ~5-15 s; no hard min/max enforced. Transcript optional
  (auto-transcribes via Whisper if omitted — extra ~1.6 GB model, lazily
  downloaded on first use of that path) but strongly preferred: output pacing
  is derived from the ref audio/text pair's implied speaking rate (see
  Aggregate notes above), so an accurate transcript matters more here than for
  the other systems.
- **MaskGCT** requires `reference_text` (a hard-required positional argument
  upstream, no ASR fallback like F5-TTS). Tested successfully against the
  same ~10 s Trump fixture used elsewhere. `target_len` is left at upstream's
  auto-predict (`None`) rather than fixed.
- **Fish-Speech** requires `reference_text` (`ServeReferenceAudio.text` has
  no default upstream, no ASR fallback). Tested successfully against the
  same ~10 s Trump fixture used elsewhere; upstream recommends 10-30 s
  reference clips but no hard bound is enforced.
- **SSR-Speech** only actually uses the first `prompt_length` seconds of the
  reference (default 3s, cut at the nearest word boundary via forced
  alignment) regardless of how long the supplied clip is; no hard minimum
  enforced but very short clips (<1-2s) may not leave enough audio for
  alignment to find a good cut point. Tested successfully against the same
  ~10 s Trump fixture used elsewhere.
- **LLASA — no confirmed length constraint.** Earlier testing suggested it
  needed a short (~4s) reference after a long clip produced near-zero output,
  but real usage since with `trump_ff_v2` clips (~6-9s) worked well once
  `repetition_penalty` was added to `model.generate()` (see the note above) —
  the original near-zero-output case was very likely the same generation
  degeneracy, not a reference-length issue. `smoke_test.py` still uses a
  short dedicated fixture (`trump_short/Donald_Trump_104_short.wav`) out of
  caution, but there's no evidence anymore that longer references are
  actually a problem — treat this as unconfirmed rather than a real
  constraint.

`smoke_test.py` routes MetaVoice to a ~39 s fixture and the others to a ~20 s
fixture. When supplying your own clips, respect these bounds.
