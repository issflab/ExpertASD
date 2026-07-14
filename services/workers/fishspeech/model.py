"""Fish-Speech (v1.5.1) wrapper. Zero-shot voice cloning via a LLaMA-style
text2semantic model + a Firefly VQGAN decoder, driven through upstream's
TTSInferenceEngine (the same class its own API/WebUI servers use).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

HF_REPO_ID = "fishaudio/fish-speech-1.5"
CACHE_DIR = Path(
    os.environ.get("FISHSPEECH_MODELS_DIR", "/data/hf_cache/expertasd_models/fishspeech")
)
CHECKPOINT_FILES = [
    "model.pth",
    "special_tokens.json",
    "tokenizer.tiktoken",
    "config.json",
    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
]
DECODER_FILENAME = "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"


class FishSpeechModel(TTSModel):
    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as load_decoder_model

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for filename in CHECKPOINT_FILES:
            hf_hub_download(HF_REPO_ID, filename, local_dir=str(CACHE_DIR))

        self.precision = torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(CACHE_DIR),
            device="cuda",
            precision=self.precision,
            compile=False,
        )
        decoder_model = load_decoder_model(
            "firefly_gan_vq", str(CACHE_DIR / DECODER_FILENAME), device="cuda"
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=False,
        )

    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        import soundfile as sf
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        if reference_audio_path is None:
            raise ValueError("fish-speech requires reference_audio_url")
        if not reference_text:
            raise ValueError(
                "fish-speech requires reference_text (transcript of the reference audio)"
            )
        params = params or {}

        req = ServeTTSRequest(
            text=text,
            references=[
                ServeReferenceAudio(audio=reference_audio_path.read_bytes(), text=reference_text)
            ],
            max_new_tokens=int(params.get("max_new_tokens", 1024)),
            top_p=float(params.get("top_p", 0.7)),
            repetition_penalty=float(params.get("repetition_penalty", 1.2)),
            temperature=float(params.get("temperature", 0.7)),
        )

        final_result = None
        for result in self.engine.inference(req):
            if result.code == "error":
                raise RuntimeError(f"fish-speech inference failed: {result.error}")
            if result.code == "final":
                final_result = result
        if final_result is None:
            raise RuntimeError("fish-speech produced no final audio segment")

        sample_rate, audio = final_result.audio
        sf.write(str(out_path), audio, sample_rate)
        return SynthOutput(sample_rate=sample_rate, duration_sec=len(audio) / sample_rate)
