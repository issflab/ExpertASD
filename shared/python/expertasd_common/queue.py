"""RQ enqueue/status helpers. Queue-name-per-system convention."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import redis
from rq import Queue
from rq.job import Job

JOB_TIMEOUT_SEC = 600
RESULT_TTL_SEC = 7 * 24 * 3600  # keep RQ bookkeeping a week; audio persists on disk


def redis_conn() -> "redis.Redis":
    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"))


def enqueue_synthesis(queue_name: str, job_id: str, payload: Dict[str, Any]) -> None:
    q = Queue(queue_name, connection=redis_conn())
    q.enqueue(
        "expertasd_common.jobs.run_synthesis",
        payload,
        job_id=job_id,
        job_timeout=JOB_TIMEOUT_SEC,
        result_ttl=RESULT_TTL_SEC,
        failure_ttl=RESULT_TTL_SEC,
    )


def fetch_job(job_id: str) -> Optional[Job]:
    try:
        return Job.fetch(job_id, connection=redis_conn())
    except Exception:
        return None


def job_status(job: Job) -> str:
    status = job.get_status(refresh=False)
    mapping = {
        "queued": "queued",
        "started": "running",
        "deferred": "queued",
        "scheduled": "queued",
        "finished": "succeeded",
        "failed": "failed",
        "stopped": "failed",
        "canceled": "failed",
    }
    return mapping.get(str(status), str(status))
