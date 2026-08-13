"""Speaker embeddings via SpeechBrain ECAPA-TDNN."""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeakerExtractor:
    """SpeechBrain `EncoderClassifier` wrapper. Requires 16 kHz mono input."""

    def __init__(self, cfg, device: str) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._classifier = None

    def _load(self) -> None:
        if self._classifier is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info("loading speaker encoder '%s' on %s", self.cfg.model, self.device)
        kwargs = {"source": self.cfg.model, "run_opts": {"device": str(self.device)}}
        if self.cfg.savedir:
            kwargs["savedir"] = self.cfg.savedir
        self._classifier = EncoderClassifier.from_hparams(**kwargs)

    def warmup(self) -> None:
        self._load()

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        self._load()
        wav = waveform.reshape(1, -1).to(self.device)
        with torch.no_grad():
            embedding = self._classifier.encode_batch(wav)
        # encode_batch returns [B, 1, D]
        return embedding.reshape(-1).float().cpu().numpy()
