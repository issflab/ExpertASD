"""Embedding extractors.

Every extractor exposes the same two-method surface:

    Extractor(cfg, device)
    Extractor.warmup()                                                # load weights
    Extractor.extract(waveform: torch.Tensor, sample_rate: int) -> np.ndarray  # (D,)

Weights load on `warmup`/first `extract`, never in `__init__`, and only enabled
extractors are ever built -- so a run that asks for SSL features alone never
downloads the ECAPA or pyannote checkpoints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch/s3prl just to name the classes
    from .pyannote import PyannoteExtractor
    from .speaker import SpeakerExtractor
    from .ssl import SSLExtractor


def build_extractor(feature: str, cfg, device: str):
    """Instantiate the extractor for a feature type, importing it lazily."""
    if feature == "ssl":
        from .ssl import SSLExtractor

        return SSLExtractor(cfg, device)
    if feature == "speaker":
        from .speaker import SpeakerExtractor

        return SpeakerExtractor(cfg, device)
    if feature == "pyannote":
        from .pyannote import PyannoteExtractor

        return PyannoteExtractor(cfg, device)
    raise ValueError(f"unknown feature type '{feature}'")


__all__ = ["build_extractor", "PyannoteExtractor", "SpeakerExtractor", "SSLExtractor"]
