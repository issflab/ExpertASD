"""MetaVoice-1B worker: load model once, then consume the metavoice-1b queue.

SimpleWorker (not the default forking Worker) is required: the default forks
a child per job, which would reload the model and CUDA context every job.
"""
import os
import sys

from rq import Queue, SimpleWorker

from expertasd_common import jobs
from expertasd_common.health import HealthState, start_health_server
from expertasd_common.queue import redis_conn

from model import MetaVoiceModel

QUEUE_NAME = os.environ.get("QUEUE_NAME", "metavoice-1b")


def main() -> None:
    state = HealthState(model=QUEUE_NAME)
    start_health_server(state)
    try:
        model = MetaVoiceModel()
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
