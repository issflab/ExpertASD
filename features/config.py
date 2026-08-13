"""Configuration objects for the feature-extraction module.

A plain object graph with no I/O and no global state: `run_extraction` can be
driven identically from the CLI, a notebook, or a future RQ worker that builds
an ExtractionConfig straight from a request payload. Validation happens at
construction so a bad request fails in milliseconds rather than 40 minutes into
a GPU run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

#: Feature types this module knows how to extract.
FEATURE_TYPES = ("ssl", "speaker", "pyannote")
POOLING_MODES = ("mean", "mean+std")
PROTOCOL_FORMATS = ("auto", "csv", "asvspoof")


class ConfigError(ValueError):
    """Raised for any malformed or contradictory configuration."""


def _check_keys(cls: type, data: Dict[str, Any], where: str) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in `{where}`: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )


def _sub(cls: type, data: Optional[Dict[str, Any]], where: str):
    """Build a nested config dataclass from a mapping, rejecting stray keys."""
    data = dict(data or {})
    if not isinstance(data, dict):
        raise ConfigError(f"`{where}` must be a mapping, got {type(data).__name__}")
    _check_keys(cls, data, where)
    return cls(**data)


@dataclass
class ProtocolConfig:
    """How to read the protocol file.

    `columns` maps a role (filename/label/system/speaker) to a column name for
    CSV protocols or a 0-based token index for ASVspoof-style ones. Anything
    left unset is auto-detected.
    """

    path: Optional[str] = None
    format: str = "auto"  # auto | csv | asvspoof
    delimiter: Optional[str] = None  # CSV only; auto-detected when None
    columns: Dict[str, Union[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in PROTOCOL_FORMATS:
            raise ConfigError(
                f"protocol.format must be one of {', '.join(PROTOCOL_FORMATS)}, "
                f"got '{self.format}'"
            )
        bad_roles = sorted(set(self.columns) - {"filename", "label", "system", "speaker"})
        if bad_roles:
            raise ConfigError(
                f"protocol.columns has unknown role(s): {', '.join(bad_roles)}. "
                "Allowed: filename, label, system, speaker"
            )


PAD_MODES = ("none", "tile")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    max_duration: Optional[float] = 10.0  # seconds; None keeps the full file
    min_duration: Optional[float] = None  # seconds; clips shorter are skipped, not failed
    pad_mode: str = "none"  # none = leave short clips as-is; tile = repeat to max_duration
    extensions: List[str] = field(default_factory=lambda: [".wav", ".flac"])

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigError("audio.sample_rate must be positive")
        if self.max_duration is not None and self.max_duration <= 0:
            raise ConfigError("audio.max_duration must be positive or null")
        if self.min_duration is not None and self.min_duration <= 0:
            raise ConfigError("audio.min_duration must be positive or null")
        if (self.min_duration is not None and self.max_duration is not None
                and self.min_duration >= self.max_duration):
            raise ConfigError(
                f"audio.min_duration ({self.min_duration}s) must be less than "
                f"audio.max_duration ({self.max_duration}s)"
            )
        if self.pad_mode not in PAD_MODES:
            raise ConfigError(
                f"audio.pad_mode must be one of {', '.join(PAD_MODES)}, got '{self.pad_mode}'"
            )
        if self.pad_mode == "tile" and self.max_duration is None:
            raise ConfigError(
                "audio.pad_mode 'tile' needs a fixed length: set audio.max_duration"
            )
        self.extensions = [e if e.startswith(".") else f".{e}" for e in self.extensions]


@dataclass
class SSLConfig:
    """s3prl upstream settings.

    `layers` are indices into the upstream's hidden states (negative counts from
    the last layer). When several are given, each is pooled independently and
    the results are concatenated in the order listed.
    """

    model: str = "wavlm_large"
    layers: List[int] = field(default_factory=lambda: [-1])
    pooling: str = "mean"  # mean | mean+std

    def __post_init__(self) -> None:
        if not self.layers:
            raise ConfigError("ssl.layers must list at least one layer index")
        if self.pooling not in POOLING_MODES:
            raise ConfigError(
                f"ssl.pooling must be one of {', '.join(POOLING_MODES)}, got '{self.pooling}'"
            )


@dataclass
class SpeakerConfig:
    model: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: Optional[str] = None  # where SpeechBrain caches the checkpoint


@dataclass
class PyannoteConfig:
    model: str = "pyannote/embedding"


@dataclass
class UmapConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    n_components: int = 2
    random_state: int = 42


@dataclass
class PlotConfig:
    dpi: int = 200
    point_size: float = 6.0
    alpha: float = 0.6
    figsize: List[float] = field(default_factory=lambda: [9.0, 7.0])
    real_color: str = "#1f77b4"  # blue is reserved for real, always
    fake_color: str = "#ff7f0e"  # used only when the protocol has no system column


@dataclass
class ExtractionConfig:
    """Everything `run_extraction` needs. Build with `from_dict` or `from_yaml`."""

    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    audio_dir: Optional[str] = None
    output_dir: str = "features_out"
    features: List[str] = field(default_factory=lambda: ["ssl"])
    ssl: SSLConfig = field(default_factory=SSLConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    pyannote: PyannoteConfig = field(default_factory=PyannoteConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    umap: UmapConfig = field(default_factory=UmapConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    resume: bool = True
    device: str = "auto"  # auto | cpu | cuda | cuda:N

    def __post_init__(self) -> None:
        if not self.features:
            raise ConfigError(
                f"`features` is empty; list at least one of: {', '.join(FEATURE_TYPES)}"
            )
        unknown = [f for f in self.features if f not in FEATURE_TYPES]
        if unknown:
            raise ConfigError(
                f"unknown feature type(s): {', '.join(unknown)}. "
                f"Known types: {', '.join(FEATURE_TYPES)}"
            )
        if len(set(self.features)) != len(self.features):
            raise ConfigError(f"`features` contains duplicates: {self.features}")
        if not self.protocol.path:
            raise ConfigError("`protocol.path` is required (path to the protocol file)")
        if not self.audio_dir:
            raise ConfigError("`audio_dir` is required (directory holding the audio files)")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionConfig":
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a mapping, got {type(data).__name__}")
        data = dict(data)
        _check_keys(cls, data, "config")
        return cls(
            protocol=_sub(ProtocolConfig, data.pop("protocol", None), "protocol"),
            ssl=_sub(SSLConfig, data.pop("ssl", None), "ssl"),
            speaker=_sub(SpeakerConfig, data.pop("speaker", None), "speaker"),
            pyannote=_sub(PyannoteConfig, data.pop("pyannote", None), "pyannote"),
            audio=_sub(AudioConfig, data.pop("audio", None), "audio"),
            umap=_sub(UmapConfig, data.pop("umap", None), "umap"),
            plot=_sub(PlotConfig, data.pop("plot", None), "plot"),
            **data,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExtractionConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        return cls.from_dict(yaml.safe_load(path.read_text()) or {})

    # -- resolved views ----------------------------------------------------

    @property
    def protocol_path(self) -> Path:
        return Path(str(self.protocol.path)).expanduser()

    @property
    def audio_root(self) -> Path:
        return Path(str(self.audio_dir)).expanduser()

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).expanduser()

    def resolve_device(self) -> str:
        """Turn `auto` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def extractor_config(self, feature: str):
        return {"ssl": self.ssl, "speaker": self.speaker, "pyannote": self.pyannote}[feature]
