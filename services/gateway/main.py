"""ExpertASD TTS gateway — the single user-facing entrypoint.

Validates requests against the system registry, enqueues jobs onto the
per-system RQ queue, and serves job status + generated audio.
"""
from __future__ import annotations

import os
import urllib.request
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from expertasd_common import queue as q
from expertasd_common import storage
from expertasd_common.schemas import (
    GenerationMetadata,
    JobAccepted,
    JobResult,
    LicenseInfo,
    ReferenceAudioInfo,
    SynthesizeRequest,
)

REGISTRY_PATH = Path(os.environ.get("TTS_REGISTRY_PATH", "/registry/tts_systems.yaml"))
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "dev")

app = FastAPI(title="ExpertASD TTS Gateway", version="0.1.0")


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())["systems"]


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/systems")
def systems() -> dict:
    registry = load_registry()
    out = {}
    for system_id, entry in registry.items():
        health_status = "unknown"
        worker_host = entry.get("worker_host", f"worker-{system_id}")
        try:
            with urllib.request.urlopen(f"http://{worker_host}:8080/health", timeout=2) as resp:
                health_status = "ready" if resp.status == 200 else "loading"
        except Exception:
            # 503 while loading raises HTTPError; anything else means unreachable
            health_status = "loading"
        out[system_id] = {**entry, "worker_health": health_status}
    return {"systems": out}


@app.post("/v1/synthesize", response_model=JobAccepted, status_code=202)
def synthesize(req: SynthesizeRequest) -> JobAccepted:
    registry = load_registry()
    entry = registry.get(req.tts_system)
    if entry is None:
        raise HTTPException(400, f"unknown tts_system '{req.tts_system}'; see /v1/systems")
    if entry.get("requires_reference_text") and not req.reference_text:
        raise HTTPException(
            400,
            f"{req.tts_system} requires reference_text (a transcript of the reference audio)",
        )
    if entry.get("zero_shot") and not req.reference_audio_url:
        raise HTTPException(400, f"{req.tts_system} requires reference_audio_url for voice cloning")

    job_id = str(uuid.uuid4())
    storage.job_dir(job_id, create=True)
    meta = GenerationMetadata(
        job_id=job_id,
        tts_system=req.tts_system,
        tts_system_commit=entry["upstream_commit"],
        ar_nar=entry["ar_nar"],
        tts_type=entry["tts_type"],
        vocoder=entry["vocoder"],
        license=LicenseInfo(**entry["license"]),
        zero_shot=entry["zero_shot"],
        reference_audio=ReferenceAudioInfo(
            provided=req.reference_audio_url is not None,
            source=req.reference_audio_url,
            reference_text=req.reference_text,
        ),
        text=req.text,
        params=req.params,
        pipeline_version=PIPELINE_VERSION,
        status="queued",
    )
    storage.write_metadata(meta)

    q.enqueue_synthesis(
        entry["queue"],
        job_id,
        {
            "job_id": job_id,
            "text": req.text,
            "reference_audio_url": req.reference_audio_url,
            "reference_text": req.reference_text,
            "params": req.params,
        },
    )
    return JobAccepted(job_id=job_id, status_url=f"/v1/jobs/{job_id}")


@app.get("/v1/jobs/{job_id}", response_model=JobResult)
def job_status(job_id: str) -> JobResult:
    meta = storage.read_metadata(job_id)
    if meta is None:
        raise HTTPException(404, f"unknown job id {job_id}")

    job = q.fetch_job(job_id)
    status = q.job_status(job) if job else meta.status
    result = None
    error = meta.error
    if status == "succeeded" and meta.output is not None:
        result = {
            "audio_url": f"/outputs/{job_id}/audio.wav",
            "duration_sec": meta.output.duration_sec,
            "sample_rate": meta.output.sample_rate,
            "metadata": meta.model_dump(),
        }
    if status == "failed" and error is None and job is not None:
        error = (job.exc_info or "").strip().splitlines()[-1] if job.exc_info else "job failed"

    return JobResult(
        job_id=job_id,
        tts_system=meta.tts_system,
        status=status,
        created_at=str(job.created_at) if job and job.created_at else None,
        started_at=str(job.started_at) if job and job.started_at else None,
        finished_at=str(job.ended_at) if job and job.ended_at else None,
        error=error,
        result=result,
    )


app.mount("/outputs", StaticFiles(directory=str(storage.outputs_dir())), name="outputs")
