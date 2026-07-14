"""F5-TTS worker: load model once, then consume the f5-tts queue.

SimpleWorker (not the default forking Worker) is required: the default forks
a child per job, which would reload the model and CUDA context every job.
Tradeoff: a CUDA-corrupting job can degrade this worker until the container
healthcheck recycles it. See docs/runbook.md.
"""
import os
import sys

from rq import Queue, SimpleWorker

from expertasd_common import jobs
from expertasd_common.health import HealthState, start_health_server
from expertasd_common.queue import redis_conn

from model import F5TTSModel

QUEUE_NAME = os.environ.get("QUEUE_NAME", "f5-tts")


def main() -> None:
    state = HealthState(model=QUEUE_NAME)
    start_health_server(state)
    try:
        model = F5TTSModel()
        model.load()
    except Exception as exc:
        state.set("error", f"{type(exc).__name__}: {exc}")
        raise
    jobs.set_model(model)
    state.set("ready")
    conn = redis_conn()
    SimpleWorker([Queue(QUEUE_NAME, connection=conn)], connection=conn).work()


if __name__ == "__main__":
    sys.exit(main())
