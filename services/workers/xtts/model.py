"""XTTS-v2 (Coqui) wrapper. Zero-shot voice cloning via TTS.api.TTS."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class XTTSModel(TTSModel):
    def load(self) -> None:
        from TTS.api import TTS

        # COQUI_TOS_AGREED=1 (set in the Dockerfile) skips the interactive
        # CPML prompt that would otherwise block this first-run download.
        self.tts = TTS(MODEL_NAME).to("cuda")

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import torchaudio

        if reference_audio_path is None:
            raise ValueError("xtts requires reference_audio_url")
        params = params or {}
        self.tts.tts_to_file(
            text=text,
            file_path=str(out_path),
            speaker_wav=[str(reference_audio_path)],
            language=params.get("language", "en"),
            temperature=params.get("temperature", 0.65),
            speed=params.get("speed", 1.0),
        )
        info = torchaudio.info(str(out_path))
        return SynthOutput(
            sample_rate=info.sample_rate,
            duration_sec=info.num_frames / info.sample_rate,
        )
