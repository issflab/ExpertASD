"""SSL embeddings via an s3prl upstream (WavLM, wav2vec2, HuBERT, ...).

Hidden states come back as a list over layers; each requested layer is pooled
over time independently. `extract_layers` returns those per-layer vectors keyed
by the layer index as configured (the pipeline writes one file per layer);
`extract` concatenates them for the uniform single-vector interface.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SSLExtractor:
    """s3prl upstream + time pooling. Requires 16 kHz mono input."""

    def __init__(self, cfg, device: str) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from s3prl.nn import S3PRLUpstream

        logger.info("loading s3prl upstream '%s' on %s", self.cfg.model, self.device)
        self._model = S3PRLUpstream(self.cfg.model).to(self.device).eval()

    def warmup(self) -> None:
        self._load()

    @staticmethod
    def _to_btd(hidden: object, batch_size: int, layer_idx: int) -> torch.Tensor:
        """Coerce one layer's output to [B, T, D].

        Upstreams differ: some yield tensors, some (tensor, lengths) tuples, and
        the batch axis is not always first.
        """
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        if not torch.is_tensor(hidden):
            raise TypeError(f"layer {layer_idx} output is not a tensor: {type(hidden)}")
        if hidden.dim() != 3:
            raise ValueError(
                f"expected a 3-D hidden state at layer {layer_idx}, got {tuple(hidden.shape)}"
            )
        if hidden.shape[0] == batch_size:
            return hidden
        if hidden.shape[1] == batch_size:
            return hidden.transpose(0, 1)
        raise ValueError(
            f"cannot locate the batch axis in layer {layer_idx} output "
            f"{tuple(hidden.shape)} for batch_size={batch_size}"
        )

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """[1, T, D] -> [D] or [2D] depending on the pooling mode."""
        mean = hidden.mean(dim=1).squeeze(0)
        if self.cfg.pooling == "mean":
            return mean
        std = hidden.std(dim=1, unbiased=False).squeeze(0)
        return torch.cat([mean, std], dim=-1)

    def extract_layers(self, waveform: torch.Tensor, sample_rate: int) -> Dict[int, np.ndarray]:
        """Pooled embedding per configured layer, keyed by the layer as given.

        One forward pass serves every requested layer. Keys are the raw config
        values (so `-1` stays `-1`), which is what the pipeline uses to name the
        per-layer output files.
        """
        self._load()
        wav = waveform.reshape(1, -1).to(self.device)
        lengths = torch.LongTensor([wav.shape[1]]).to(self.device)
        with torch.no_grad():
            all_hs, _ = self._model(wav, lengths)

        n_layers = len(all_hs)
        out: Dict[int, np.ndarray] = {}
        for layer_idx in self.cfg.layers:
            resolved = layer_idx if layer_idx >= 0 else n_layers + layer_idx
            if not 0 <= resolved < n_layers:
                raise IndexError(
                    f"ssl.layers requested layer {layer_idx} but '{self.cfg.model}' "
                    f"exposes {n_layers} layers (valid: -{n_layers}..{n_layers - 1})"
                )
            pooled = self._pool(self._to_btd(all_hs[resolved], 1, resolved))
            out[layer_idx] = pooled.float().cpu().numpy()
        return out

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """All configured layers concatenated into one vector (uniform interface)."""
        per_layer = self.extract_layers(waveform, sample_rate)
        return np.concatenate([per_layer[layer] for layer in self.cfg.layers], axis=-1)
