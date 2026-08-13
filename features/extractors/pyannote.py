"""Speaker embeddings via pyannote.audio.

`pyannote/embedding` is a gated model: the HF account must have accepted its
terms and the token is read from the HF_TOKEN environment variable. The token is
never written to disk or into any config.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

TOKEN_ENV = "HF_TOKEN"


class PyannoteExtractor:
    """`Inference(..., window="whole")` wrapper. Requires 16 kHz mono input."""

    def __init__(self, cfg, device: str) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._inference = None

    def _load(self) -> None:
        if self._inference is not None:
            return
        from pyannote.audio import Inference, Model

        token = os.environ.get(TOKEN_ENV)
        if not token:
            logger.warning(
                "%s is not set; loading '%s' will fail if the model is gated",
                TOKEN_ENV, self.cfg.model,
            )
        logger.info("loading pyannote model '%s' on %s", self.cfg.model, self.device)
        # `token=` is the modern huggingface_hub kwarg; `use_auth_token` was
        # removed in hub >=1.0. pyannote.audio must be new enough to forward it
        # (>=3.3 accepts `token`); older lines need an older hub.
        try:
            model = Model.from_pretrained(self.cfg.model, token=token)
        except TypeError:
            model = Model.from_pretrained(self.cfg.model, use_auth_token=token)
        if model is None:
            raise RuntimeError(
                f"pyannote returned no model for '{self.cfg.model}'. Accept the model's "
                f"terms on huggingface.co and export a valid {TOKEN_ENV}."
            )
        self._inference = Inference(model, window="whole", device=self.device)

    def warmup(self) -> None:
        self._load()

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        self._load()
        wav = waveform.reshape(1, -1)  # pyannote wants [channel, sample]
        embedding = self._inference({"waveform": wav, "sample_rate": sample_rate})
        if torch.is_tensor(embedding):
            embedding = embedding.detach().cpu().numpy()
        return np.asarray(embedding, dtype=np.float32).reshape(-1)
