"""MaskGCT (Amphion) wrapper. Zero-shot voice cloning via masked generative
codec transformers (semantic + acoustic token prediction).

Ported to call directly into Amphion's models.tts.maskgct.maskgct_utils
module — the upstream repo isn't pip-installable, so it's imported from a
cloned checkout on PYTHONPATH (see Dockerfile), same approach as StyleTTS2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from expertasd_common.model_base import SynthOutput, TTSModel

REPO_DIR = "/opt/Amphion"
HF_REPO_ID = "amphion/MaskGCT"
OUTPUT_SAMPLE_RATE = 24000


class MaskGCTModel(TTSModel):
    def load(self) -> None:
        sys.path.insert(0, REPO_DIR)
        # maskgct_utils.build_semantic_model() and the model configs load a
        # few paths as plain relative strings (e.g.
        # "./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"), resolved
        # relative to cwd.
        os.chdir(REPO_DIR)

        import safetensors.torch
        import torch
        from huggingface_hub import hf_hub_download
        from models.tts.maskgct.maskgct_utils import (
            MaskGCT_Inference_Pipeline,
            build_acoustic_codec,
            build_s2a_model,
            build_semantic_codec,
            build_semantic_model,
            build_t2s_model,
        )
        from utils.util import load_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = load_config("./models/tts/maskgct/config/maskgct.json")
        semantic_model, semantic_mean, semantic_std = build_semantic_model(self.device)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, self.device)
        codec_encoder, codec_decoder = build_acoustic_codec(
            cfg.model.acoustic_codec, self.device
        )
        t2s_model = build_t2s_model(cfg.model.t2s_model, self.device)
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, self.device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, self.device)

        safetensors.torch.load_model(
            semantic_codec, hf_hub_download(HF_REPO_ID, "semantic_codec/model.safetensors")
        )
        safetensors.torch.load_model(
            codec_encoder, hf_hub_download(HF_REPO_ID, "acoustic_codec/model.safetensors")
        )
        safetensors.torch.load_model(
            codec_decoder, hf_hub_download(HF_REPO_ID, "acoustic_codec/model_1.safetensors")
        )
        safetensors.torch.load_model(
            t2s_model, hf_hub_download(HF_REPO_ID, "t2s_model/model.safetensors")
        )
        safetensors.torch.load_model(
            s2a_model_1layer,
            hf_hub_download(HF_REPO_ID, "s2a_model/s2a_model_1layer/model.safetensors"),
        )
        safetensors.torch.load_model(
            s2a_model_full,
            hf_hub_download(HF_REPO_ID, "s2a_model/s2a_model_full/model.safetensors"),
        )

        self.pipeline = MaskGCT_Inference_Pipeline(
            semantic_model,
            semantic_codec,
            codec_encoder,
            codec_decoder,
            t2s_model,
            s2a_model_1layer,
            s2a_model_full,
            semantic_mean,
            semantic_std,
            self.device,
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

        if reference_audio_path is None:
            raise ValueError("maskgct requires reference_audio_url")
        if not reference_text:
            raise ValueError("maskgct requires reference_text (transcript of the reference audio)")
        params = params or {}
        language = params.get("language", "en")

        wav = self.pipeline.maskgct_inference(
            str(reference_audio_path),
            reference_text,
            text,
            language,
            params.get("target_language", language),
            target_len=None,  # upstream auto-predicts from text/prompt ratio
            n_timesteps=int(params.get("n_timesteps", 25)),
            cfg=float(params.get("cfg", 2.5)),
            rescale_cfg=float(params.get("rescale_cfg", 0.75)),
        )
        sf.write(str(out_path), wav, OUTPUT_SAMPLE_RATE)
        return SynthOutput(
            sample_rate=OUTPUT_SAMPLE_RATE,
            duration_sec=len(wav) / OUTPUT_SAMPLE_RATE,
        )
