"""The RQ job function executed inside every worker.

The worker process (SimpleWorker, no fork-per-job) loads its model once at
startup and registers it here via set_model(); run_synthesis then reuses
that resident model for every job on the queue.
"""
from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import storage
from .model_base import TTSModel
from .schemas import GenerationInfo, OutputInfo, utcnow_iso

_MODEL: Optional[TTSModel] = None


def set_model(model: TTSModel) -> None:
    global _MODEL
    _MODEL = model


def run_synthesis(payload: Dict[str, Any]) -> Dict[str, Any]:
    if _MODEL is None:
        raise RuntimeError("worker has no model registered; set_model() not called")

    job_id = payload["job_id"]
    meta = storage.read_metadata(job_id)
    if meta is None:
        raise RuntimeError(f"metadata skeleton missing for job {job_id}")

    meta.status = "running"
    storage.write_metadata(meta)

    out_dir = storage.job_dir(job_id)  # gateway already created it
    ref_path: Optional[Path] = None
    if payload.get("reference_audio_url"):
        ref_path = storage.resolve_reference_audio(payload["reference_audio_url"], out_dir)
        meta.reference_audio.sha256 = storage.sha256_file(ref_path)

    out_path = storage.audio_path(job_id)
    started = time.monotonic()
    try:
        synth = _MODEL.synthesize(
            text=payload["text"],
            out_path=out_path,
            reference_audio_path=ref_path,
            reference_text=payload.get("reference_text"),
            params=payload.get("params") or {},
        )
    except Exception as exc:
        meta.status = "failed"
        meta.error = f"{type(exc).__name__}: {exc}"
        storage.write_metadata(meta)
        raise

    latency = time.monotonic() - started
    # Workers run as root; some models (e.g. MetaVoice) write output with 0600.
    # Force world-readable so the file is usable outside the container.
    try:
        out_path.chmod(0o644)
    except OSError:
        pass
    gpu_index_raw = os.environ.get("GPU_INDEX")
    meta.output = OutputInfo(
        path=str(out_path),
        sample_rate=synth.sample_rate,
        duration_sec=round(synth.duration_sec, 3),
        sha256=storage.sha256_file(out_path),
    )
    meta.generation = GenerationInfo(
        requested_at=utcnow_iso(),
        latency_sec=round(latency, 3),
        worker_host=socket.gethostname(),
        gpu_index=int(gpu_index_raw) if gpu_index_raw is not None else None,
    )
    meta.status = "succeeded"
    storage.write_metadata(meta)
    return {
        "audio_path": str(out_path),
        "sample_rate": synth.sample_rate,
        "duration_sec": round(synth.duration_sec, 3),
        "latency_sec": round(latency, 3),
    }
