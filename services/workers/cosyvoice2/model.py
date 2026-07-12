"""CosyVoice2 wrapper.

inference_zero_shot(tts_text, prompt_text, prompt_wav) is a generator
yielding {'tts_speech': tensor[1, S]} chunks. In this pinned commit prompt_wav
is a FILE PATH — the frontend calls load_wav() on it internally at 16k and 24k
— so we pass the reference path, not a pre-loaded tensor. Output sample rate
comes from the model config (24000 for CosyVoice2-0.5B). Requires a transcript
of the reference audio.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

HF_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"


class CosyVoice2Model(TTSModel):
    def load(self) -> None:
        model_dir = os.environ.get(
            "COSYVOICE_MODEL_DIR", "/data/hf_cache/expertasd_models/CosyVoice2-0.5B"
        )
        # Pre-download from HF ourselves: CosyVoice2's own fallback for a
        # missing model_dir goes to ModelScope, not HuggingFace.
        if not (Path(model_dir) / "cosyvoice2.yaml").exists():
            from huggingface_hub import snapshot_download

            snapshot_download(HF_REPO_ID, local_dir=model_dir)
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model = CosyVoice2(model_dir)
        self.sample_rate = self.model.sample_rate  # config-driven (24000)

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import torch
        import torchaudio

        if reference_audio_path is None:
            raise ValueError("cosyvoice2 requires reference_audio_url")
        if not reference_text:
            raise ValueError("cosyvoice2 requires reference_text (transcript of the reference audio)")
        params = params or {}
        chunks = [
            out["tts_speech"]
            for out in self.model.inference_zero_shot(
                text,
                reference_text,
                str(reference_audio_path),
                stream=False,
                speed=float(params.get("speed", 1.0)),
            )
        ]
        audio = torch.cat(chunks, dim=1)
        torchaudio.save(str(out_path), audio, self.sample_rate)
        return SynthOutput(
            sample_rate=self.sample_rate,
            duration_sec=audio.shape[-1] / self.sample_rate,
        )
