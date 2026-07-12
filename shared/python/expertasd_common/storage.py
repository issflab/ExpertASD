"""Filesystem layout and metadata persistence for generated samples.

Layout: $DATA_ROOT/outputs/<job_id>/{audio.wav, metadata.json}
The gateway writes the initial metadata skeleton; the worker updates it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from .schemas import GenerationMetadata

OUTPUTS_ENV = "OUTPUTS_DIR"


def outputs_dir() -> Path:
    return Path(os.environ.get(OUTPUTS_ENV, "/data/outputs"))


def job_dir(job_id: str, tts_system: Optional[str] = None, create: bool = False) -> Path:
    """Resolve a job's output directory: outputs/<tts_system>/<job_id>/.

    Creating requires tts_system (the gateway knows it). Lookups by job_id
    alone glob across systems, since job_id is globally unique.
    """
    if create:
        if tts_system is None:
            raise ValueError("tts_system is required to create a job directory")
        d = outputs_dir() / tts_system / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d
    matches = sorted(outputs_dir().glob(f"*/{job_id}"))
    if matches:
        return matches[0]
    return outputs_dir() / "_unresolved" / job_id


def metadata_path(job_id: str) -> Path:
    return job_dir(job_id) / "metadata.json"


def audio_path(job_id: str) -> Path:
    return job_dir(job_id) / "audio.wav"


def write_metadata(meta: GenerationMetadata) -> None:
    path = metadata_path(meta.job_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(meta.model_dump_json(indent=2))
    tmp.replace(path)


def read_metadata(job_id: str) -> Optional[GenerationMetadata]:
    path = metadata_path(job_id)
    if not path.exists():
        return None
    return GenerationMetadata.model_validate_json(path.read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_reference_audio(url: str, dest_dir: Path) -> Path:
    """Materialize a reference_audio_url as a local file in dest_dir.

    Supports file:// paths (bind-mounted fixtures) and http(s) URLs.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "reference_audio"
    if url.startswith("file://"):
        src = Path(url[len("file://"):])
        if not src.exists():
            raise FileNotFoundError(f"reference audio not found: {src}")
        dest = dest.with_suffix(src.suffix or ".wav")
        shutil.copyfile(src, dest)
    elif url.startswith(("http://", "https://")):
        dest = dest.with_suffix(".wav")
        with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    else:
        raise ValueError(f"unsupported reference_audio_url scheme: {url}")
    return dest
