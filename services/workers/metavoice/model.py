"""MetaVoice-1B wrapper.

fam.llm.fast_inference.TTS.synthesise(text, spk_ref_path) writes a WAV under
output_dir and returns its path; weights come from HF metavoiceio/metavoice-1B-v0.1
via snapshot_download (respects HF_HOME). Upstream recommends >=30s of
reference audio. First call includes a torch.compile warm-up.

Long-text handling: MetaVoice shares a fixed 2048-token budget between the text
prompt and generated audio tokens, and emits an end-of-audio token
stochastically, so a single call truncates on long input. We split the target
text into sentence-level chunks, synthesize each, and concatenate — the standard
long-form approach. Chunk size is capped by word count to stay well inside the
per-generation limit. Set params.max_chunk_words=0 to disable chunking.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

DEFAULT_MAX_CHUNK_WORDS = 20
GAP_SECONDS = 0.15


def split_into_chunks(text: str, max_words: int) -> List[str]:
    """Split text into sentence-grouped chunks of <= max_words words each."""
    text = text.strip()
    if max_words <= 0 or not text:
        return [text] if text else []
    # sentence boundaries, keeping the terminal punctuation
    sentences = re.findall(r"[^.!?]+[.!?]?", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_n = 0
    for sent in sentences:
        n = len(sent.split())
        if n > max_words:
            # a single long sentence: hard-split on commas / whitespace
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_n = [], 0
            words = sent.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if cur_n + n > max_words and cur:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(sent)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class MetaVoiceModel(TTSModel):
    def load(self) -> None:
        from fam.llm.fast_inference import TTS

        self.tts = TTS(output_dir="/tmp/metavoice_outputs")

    def _synth_one(self, text: str, ref: str, params: Dict[str, Any]) -> str:
        return self.tts.synthesise(
            text=text,
            spk_ref_path=ref,
            top_p=float(params.get("top_p", 0.95)),
            guidance_scale=float(params.get("guidance_scale", 3.0)),
            temperature=float(params.get("temperature", 1.0)),
        )

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import numpy as np
        import soundfile as sf

        if reference_audio_path is None:
            raise ValueError("metavoice-1b requires reference_audio_url")
        params = params or {}
        ref = str(reference_audio_path)
        max_chunk_words = int(params.get("max_chunk_words", DEFAULT_MAX_CHUNK_WORDS))
        chunks = split_into_chunks(text, max_chunk_words)
        if not chunks:
            raise ValueError("empty text after normalization")

        if len(chunks) == 1:
            wav_path = self._synth_one(chunks[0], ref, params)
            data, sr = sf.read(wav_path)
            shutil.move(wav_path, out_path)
            return SynthOutput(sample_rate=sr, duration_sec=len(data) / sr)

        segments = []
        sr = None
        for chunk in chunks:
            wav_path = self._synth_one(chunk, ref, params)
            data, sr = sf.read(wav_path)
            segments.append(data)
            os.remove(wav_path)
        gap = np.zeros(int(GAP_SECONDS * sr), dtype=segments[0].dtype)
        pieces = []
        for i, seg in enumerate(segments):
            if i:
                pieces.append(gap)
            pieces.append(seg)
        full = np.concatenate(pieces)
        sf.write(str(out_path), full, sr)
        return SynthOutput(sample_rate=sr, duration_sec=len(full) / sr)
