"""Tortoise-TTS wrapper. Reference clips at 22.05kHz in, 24kHz WAV out."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

OUTPUT_SAMPLE_RATE = 24000  # per tortoise.api.TextToSpeech.tts docstring
REFERENCE_SAMPLE_RATE = 22050


class TortoiseModel(TTSModel):
    def load(self) -> None:
        from tortoise.api import TextToSpeech

        # models_dir defaults to $TORTOISE_MODELS_DIR (set in Dockerfile to a
        # /data/hf_cache subdir); init triggers weight download on first run.
        self.tts = TextToSpeech(kv_cache=True)

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import torchaudio
        from tortoise.utils.audio import load_audio

        if reference_audio_path is None:
            raise ValueError("tortoise-tts requires reference_audio_url")
        params = params or {}
        sample = load_audio(str(reference_audio_path), REFERENCE_SAMPLE_RATE)
        gen = self.tts.tts_with_preset(
            text,
            voice_samples=[sample],
            preset=params.get("preset", "fast"),
        )
        if gen.dim() == 3:  # (k, 1, S) when k > 1
            gen = gen[0]
        gen = gen.cpu()
        torchaudio.save(str(out_path), gen, OUTPUT_SAMPLE_RATE)
        return SynthOutput(
            sample_rate=OUTPUT_SAMPLE_RATE,
            duration_sec=gen.shape[-1] / OUTPUT_SAMPLE_RATE,
        )
