"""Feature extraction over an anti-spoofing protocol: SSL, speaker and
pyannote embeddings, plus UMAP visualisation.

Deliberately free of any transport concerns -- no queue, no web framework, no
dependency on this repo's shared service library -- so the same code runs as a
local CLI today and inside a `worker-features` container later, with the worker
as a thin wrapper around `run_extraction`.
"""
from __future__ import annotations

from .config import ExtractionConfig
from .pipeline import Artifact, ExtractionResult, FileFailure, run_extraction

__all__ = [
    "Artifact",
    "ExtractionConfig",
    "ExtractionResult",
    "FileFailure",
    "run_extraction",
]
