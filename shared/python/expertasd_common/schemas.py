"""Pydantic models mirroring shared/schemas/openapi.yaml.

The gateway and every worker import these types; they are the single
source of truth for the request/response and metadata contracts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SynthesizeRequest(BaseModel):
    tts_system: str
    text: str = Field(min_length=1)
    reference_audio_url: Optional[str] = None
    # Transcript of the reference audio. Required by systems whose registry
    # entry sets requires_reference_text (CosyVoice2 in the pilot).
    reference_text: Optional[str] = None
    # Optional human-readable stem for the output dir / job id. When set it is
    # used instead of the reference-audio basename (e.g. to name outputs after
    # the text row rather than the reference clip).
    label: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    requested_by: str = "unknown"
    callback_url: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    status_url: str


class LicenseInfo(BaseModel):
    code_license: str
    weights_license: str
    license_verified_at: str
    notes: str = ""


class ReferenceAudioInfo(BaseModel):
    provided: bool
    source: Optional[str] = None
    reference_text: Optional[str] = None
    sha256: Optional[str] = None


class OutputInfo(BaseModel):
    path: str
    sample_rate: int
    duration_sec: float
    sha256: str


class GenerationInfo(BaseModel):
    requested_at: str
    latency_sec: float
    worker_host: str
    gpu_index: Optional[int] = None


class GenerationMetadata(BaseModel):
    job_id: str
    tts_system: str
    tts_system_commit: str
    ar_nar: str
    tts_type: List[str]
    vocoder: str
    license: LicenseInfo
    zero_shot: bool
    reference_audio: ReferenceAudioInfo
    text: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[OutputInfo] = None
    generation: Optional[GenerationInfo] = None
    pipeline_version: str = "dev"
    status: str = "queued"
    error: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    tts_system: str
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
