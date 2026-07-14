"""F5-TTS wrapper. Zero-shot voice cloning via flow matching (DiT + Vocos)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

MODEL_NAME = "F5TTS_v1_Base"
CACHE_DIR = os.environ.get("F5TTS_MODELS_DIR", "/data/hf_cache/expertasd_models/f5tts")


class F5TTSModel(TTSModel):
    def load(self) -> None:
        from f5_tts.api import F5TTS

        # hf_cache_dir is threaded through cached_path() for both the main
        # checkpoint and the Vocos vocoder, keeping everything under our
        # shared/controlled cache rather than cached_path's own default
        # (~/.cache/cached_path, outside the persisted /data volume).
        self.tts = F5TTS(model=MODEL_NAME, device="cuda", hf_cache_dir=CACHE_DIR)

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        if reference_audio_path is None:
            raise ValueError("f5-tts requires reference_audio_url")
        params = params or {}

        # Empty ref_text triggers upstream's internal Whisper ASR fallback to
        # transcribe the reference audio (extra ~1.6 GB model, downloaded
        # lazily on first use of that path) — pass through what the caller
        # gave us rather than always supplying something.
        wav, sr, _spec = self.tts.infer(
            ref_file=str(reference_audio_path),
            ref_text=reference_text or "",
            gen_text=text,
            nfe_step=int(params.get("nfe_step", 32)),
            cfg_strength=float(params.get("cfg_strength", 2)),
            speed=float(params.get("speed", 1.0)),
            file_wave=str(out_path),
        )
        return SynthOutput(sample_rate=sr, duration_sec=len(wav) / sr)
