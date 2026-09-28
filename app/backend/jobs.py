"""In-process background jobs for long analyses (hotspot scan, mutation design) that exceed an HTTP timeout on CPU.

A job runs on a single worker thread (ESM passes are serialised by ESMFeatureExtractor anyway) and reports
progress through a callback; clients poll GET /analysis/jobs/{job_id}. Finished jobs are kept for JOB_TTL seconds.
"""
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

JOB_TTL = 3600
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-job")
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _prune():
    cutoff = time.time() - JOB_TTL
    for job_id in [j for j, job in _jobs.items() if job["finished_at"] and job["finished_at"] < cutoff]:
        del _jobs[job_id]


def submit(kind: str, fn: Callable[[Callable[[int, int, str], None]], Any]) -> str:
    """Queues fn(progress) and returns the job id. fn's return value becomes the job result."""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _prune()
        _jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "queued", "progress": {"done": 0, "total": 0, "stage": "Queued"},
                         "result": None, "error": None, "created_at": time.time(), "finished_at": None}

    def progress(done: int, total: int, stage: str):
        with _lock:
            _jobs[job_id]["progress"] = {"done": int(done), "total": int(total), "stage": stage}

    def run():
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"]["stage"] = "Starting"
        try:
            result = fn(progress)
            with _lock:
                _jobs[job_id].update(status="done", result=result, finished_at=time.time())
        except Exception as e:
            traceback.print_exc()
            with _lock:
                _jobs[job_id].update(status="error", error=str(e) or type(e).__name__, finished_at=time.time())

    _executor.submit(run)
    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        out = {k: v for k, v in job.items() if k not in ("created_at", "finished_at")}
        out["elapsed_seconds"] = round((job["finished_at"] or time.time()) - job["created_at"], 1)
        return out
