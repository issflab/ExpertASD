# Design review: human-in-the-loop audio deepfake detection

A critique of the pipeline in `Human in the Loop Audio Deepfake Detection.png`, written
from the position of someone who would have to defend a report this system produces.
It covers what is sound, what will break, and what to build next.

Status: review document, 2026-08-13. Not a spec — the concrete work items it implies
are listed in [Work items](#work-items) at the end.

---

## 1. The verdict on the concept

The architecture is aimed at the correct weakness in the field. Generic audio deepfake
detectors collapse out of domain: systems that reach ~1% EER on ASVspoof routinely
land at 20–40% EER on in-the-wild audio. Conditioning detection on a *specific* speaker
for whom we hold genuine reference material is the single largest available lever, and
pairing it with expert review and a written report is where forensic value actually lives.

So the concept works. The risk is not conceptual. It is that the design as drawn has
three places where it will produce a confident, well-formatted, wrong answer, and the
human loop as currently wired will not catch any of them.

---

## 2. Three structural problems

### 2.1 UMAP cannot be the evidence

The plan asks a human to "see where the suspected audio lies in the feature space".
As drawn, that is not a sound inference for three separate reasons.

**A single query point cannot be honestly placed on a UMAP that was fitted with it.**
Refitting with the query changes the entire map, so two runs of the same case produce
different pictures and neither is reproducible. This is disqualifying in a report.

**UMAP distance is not a quantity.** Local neighbourhoods are approximately preserved;
global geometry is not. "It sits far from the real cluster" carries no calibrated meaning
and cannot be compared across cases, speakers, or feature streams.

**The clusters will mostly be channel, not spoofing.** SSL and speaker embeddings separate
microphone, codec, and session far more strongly than they separate genuine from synthetic.
A clean visual separation *feels* like proof and is very often an artefact of the reference
material coming from YouTube while the query came from a phone.

**Fix.** Fit the projection once over the reference corpus, persist it, and project queries
through the frozen model. Compute the decision in the **original high-dimensional space** —
kNN distance to the genuine manifold, Mahalanobis distance, subspace residual — and report
it as a percentile against held-out genuine clips from the same speaker. Report the
reliability of the 2D placement (high-dim vs 2D neighbour agreement) and flag the plot when
that agreement is low. The plot is for the human's intuition. The number is the evidence.

*Implemented: see [features/manifold.py](../features/manifold.py) and §5.*

### 2.2 Fake generation creates a closed world

If the speaker-specific detector is trained on the ten TTS systems in `services/workers/`,
it learns "is this one of my ten vocoders", not "is this not the real speaker". An attacker
using system eleven — or simply running the output through Opus at 24 kbps — defeats it.

Subtler and more dangerous: the fakes are synthesised *from* the reference audio, so fakes
and genuine references share a channel. The model can learn a channel-matched-vs-mismatched
shortcut, score near-zero EER internally, and be useless in the field.

**Fixes, in priority order.**

1. **Leave-one-system-out is a mandatory reported metric**, not an ablation. Seen-attack
   numbers must never be reported alone.
2. **Prefer a one-class / speaker-anchored formulation.** Model the genuine speaker's
   distribution tightly and score deviation from it, rather than training a binary
   discriminator over the attacks you happen to own. It degrades far more gracefully
   on unseen synthesisers.
3. **Apply the same augmentation chain to genuine references, generated fakes, and query**:
   MP3/AAC/Opus at several bitrates, telephone band, resampling, reverberation, background
   music, loudness normalisation. Codec laundering is the most common real-world evasion
   and defending against it costs nothing.
4. **Add explicit bandwidth/rolloff features.** A vocoder running at 22.05 kHz and upsampled
   to 44.1 kHz leaves a hard spectral cliff. It is among the most reliable cues available
   *and* it is directly visible in the spectrogram panel the human is already reading —
   machine evidence and human evidence pointing at the same pixels is exactly what a report
   needs.

### 2.3 The loop is not closed, and the human will agree with the model

Two distinct problems on the human-facing side of the diagram.

**Automation bias.** If the expert sees detector scores and the LLM's draft before forming
a view, they anchor on it. What you collect is agreement, not verification, and the
human-in-the-loop claim does not survive scrutiny.

*Fix — blind-first protocol.* Show the expert the audio and spectrogram, capture their
judgement and rationale, *then* reveal model scores and the feature view, then allow
revision. Log both verdicts. This costs nothing, and it yields a free dataset on expert
calibration plus a defensible claim about what the loop contributes.

**No return arrow.** In the diagram the human feeds the report engine and nothing else.
That is human-in-the-*line*. Expert verdicts should flow back: corrected labels into the
reference set, disputed cases into an active-learning queue, per-speaker threshold updates.

---

## 3. Missing components

**Segment-level scoring with a timeline.** Utterance-level real/fake is the 2021 problem.
The realistic current attack is partial — a few words spliced into genuine audio. Score in
~1 s windows with overlap and produce a per-second track overlaid on the spectrogram. This
is both a detection gain and the thing that makes the human's job concrete ("the model flags
4.2–5.8 s; listen there").

**A calibration and fusion module.** The generic detector, the speaker-specific detector,
and the feature-space distance currently produce three incomparable numbers. Fusion needs
per-speaker score normalisation (borrow s-norm from speaker verification — per-speaker score
distributions vary enormously) and calibration to a likelihood ratio. Forensic reporting
standards expect an LR, not a verdict.

**"Inconclusive" as a first-class output.** Three-way: synthetic / genuine / insufficient
evidence. Most of the real-world value lies in the inconclusive band being honest. Falsely
calling a genuine recording of a public figure a deepfake is a catastrophic error class, so
the operating point belongs at very low false-positive rate with abstention in the middle.

**An evidence-sufficiency gate, before the run.** If reference material falls below a
minimum duration across a minimum number of distinct sessions/channels, the speaker-specific
path should refuse rather than emit a weak score.

**A reference-set poisoning check.** Search-and-collect scraping a public figure will
eventually ingest audio that is itself synthetic — and then it is enrolment data. Run the
generic detector over the reference set itself, prefer high-trust sources (official channels,
broadcast archives, C2PA-signed media), and flag outliers before enrolment.

---

## 4. The LLM report engine

This is the component that will demo best and deploy worst. Constraints, all of them hard:

- **The LLM does not make the determination.** The verdict comes from the calibrated fusion
  score via a fixed rule. The LLM writes prose *around* a decision it has no authority to
  change.
- **Every quantitative claim cites a field.** Pass structured fields with IDs; validate the
  generated text against those fields before any human sees it. A model handed six scores and
  asked to "write a report" will produce a fluent, confident narrative whether or not the
  evidence supports one.
- **The counter-hypothesis is mandatory.** Channel mismatch, limited reference material,
  unseen synthesiser, out-of-domain content — a limitations section generated from flags the
  pipeline actually raised, not from the model's imagination.
- **Report a likelihood ratio, not a verdict sentence.**
- **Human input is stored verbatim and kept separate from machine evidence.**
- **Version-stamp everything and keep the audit trail:** model checkpoints and hashes, config,
  reference-set manifest, projection fit ID, prompt, raw output, expert verdict. If a report
  is contested, exact reproducibility is the whole case.

---

## 5. One expectation to reset

The ECAPA and pyannote speaker streams will probably **not** separate genuine from synthetic,
and that is not a bug. A good zero-shot cloner is explicitly optimised to land on top of the
target in speaker-embedding space. Treat those streams as an identity gate ("is this even the
claimed speaker?") and as a channel probe — not as detection features.

The separation signal lives in the mid SSL layers and in artefact features. Choose the WavLM /
XLS-R layer by leave-one-system-out AUC, not by which UMAP looks best; the per-layer stream
design in [features/pipeline.py](../features/pipeline.py) already makes that sweep cheap.

---

## 6. What makes this publishable rather than a demo

The claim "expert + LLM + speaker-specific detection beats the alternatives" requires an
actual study: N experts, counterbalanced conditions (human alone / model alone / full loop),
measuring accuracy **and** calibration **and** time-to-decision.

Alongside it:

- Cross-dataset generalisation: In-the-Wild, ASVspoof 2021 DF, ASVspoof 5, Deepfake-Eval-2024.
- Leave-one-TTS-out over the ten workers.
- **False-positive rate on a large volume of genuine audio of the target speakers from channels
  never seen in training.** This is the metric everyone skips and the one that kills deployments.
- Laundering robustness: re-encode, resample, add noise and background music, measure the
  degradation curve.

---

## Work items

Ordered by leverage, not by effort.

| # | Item | Status |
|---|---|---|
| 1 | Frozen projection + high-dimensional scoring with calibrated percentiles | **done** — `features/manifold.py` |
| 2 | Codec/channel augmentation chain applied to references, fakes, and query | todo |
| 3 | Leave-one-system-out evaluation harness | todo |
| 4 | Segment-level (windowed) scoring and spectrogram timeline | todo |
| 5 | Blind-first expert elicitation protocol in the review UI | todo |
| 6 | Calibration + fusion module with per-speaker normalisation, LR output | todo |
| 7 | Evidence-sufficiency gate and three-way verdict incl. inconclusive | partial — support check in `manifold.py` |
| 8 | Reference-set poisoning check at enrolment | todo |
| 9 | Bandwidth/rolloff and other explainable artefact features | todo |
| 10 | Report engine with field-grounded generation and audit trail | todo |
| 11 | Human verdicts flowing back into the reference set / active learning | todo |
