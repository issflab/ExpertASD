"""MetaVoice-1B wrapper.

fam.llm.fast_inference.TTS.synthesise(text, spk_ref_path) writes a WAV under
output_dir and returns its path; weights come from HF metavoiceio/metavoice-1B-v0.1
via snapshot_download (respects HF_HOME). Upstream recommends >=30s of
reference audio. First call includes a torch.compile warm-up.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel


class MetaVoiceModel(TTSModel):
    def load(self) -> None:
        from fam.llm.fast_inference import TTS

        self.tts = TTS(output_dir="/tmp/metavoice_outputs")

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import soundfile as sf

        if reference_audio_path is None:
            raise ValueError("metavoice-1b requires reference_audio_url")
        params = params or {}
        wav_path = self.tts.synthesise(
            text=text,
            spk_ref_path=str(reference_audio_path),
            top_p=float(params.get("top_p", 0.95)),
            guidance_scale=float(params.get("guidance_scale", 3.0)),
            temperature=float(params.get("temperature", 1.0)),
        )
        data, sample_rate = sf.read(wav_path)
        shutil.move(wav_path, out_path)
        return SynthOutput(
            sample_rate=sample_rate,
            duration_sec=len(data) / sample_rate,
        )
