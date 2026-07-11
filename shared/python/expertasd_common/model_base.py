"""Abstract base class every worker's model wrapper must implement."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SynthOutput:
    sample_rate: int
    duration_sec: float


class TTSModel(abc.ABC):
    """Contract between a worker's model wrapper and the shared job runner.

    load() is called once at container startup, before the worker starts
    consuming jobs; synthesize() is called once per job and must write a
    WAV file to out_path.
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights onto the GPU. Must block until fully ready."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        out_path: Path,
        reference_audio_path: Optional[Path] = None,
        reference_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthOutput:
        """Generate speech for text and write a WAV file to out_path."""
