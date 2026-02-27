# Module: store
# License: MIT (ARVTON project)
# Description: Thread-safe job state management with asyncio.Lock.
# Platform: Cloud GPU
# Dependencies: asyncio, uuid, dataclasses

"""
Job Store — Thread-safe in-memory job state.
Uses asyncio.Lock for concurrency safety.
"""

import asyncio
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: Optional[str] = None
    glb_url: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    person_path: Optional[str] = None
    garment_path: Optional[str] = None
    quality: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "glb_url": self.glb_url,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
        }


class JobStore:
    """
    Thread-safe in-memory job store.
    Keeps the last 1000 jobs in an OrderedDict (LRU eviction).
    """

    def __init__(self, max_jobs: int = 1000):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_jobs = max_jobs

    async def create(
        self,
        person_path: str,
        garment_path: str,
        quality: str = "auto",
    ) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            person_path=person_path,
            garment_path=garment_path,
            quality=quality,
        )

        async with self._lock:
            # Evict oldest if at capacity
            while len(self._jobs) >= self._max_jobs:
                self._jobs.popitem(last=False)
            self._jobs[job_id] = job

        return job

    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[str] = None,
        glb_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update job fields."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            if status is not None:
                job.status = status
                if status == JobStatus.PROCESSING and job.started_at is None:
                    job.started_at = time.time()
                elif status in (JobStatus.DONE, JobStatus.FAILED):
                    job.completed_at = time.time()
                    if job.started_at:
                        job.duration_ms = int(
                            (job.completed_at - job.started_at) * 1000
                        )

            if progress is not None:
                job.progress = progress
            if glb_url is not None:
                job.glb_url = glb_url
            if error is not None:
                job.error = error

            return job

    async def delete(self, job_id: str) -> bool:
        """Delete a job. Returns True if deleted."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    async def queue_length(self) -> int:
        """Number of queued jobs."""
        async with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status == JobStatus.QUEUED
            )

    async def last_n_jobs(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get summaries of the last N jobs."""
        async with self._lock:
            jobs = list(self._jobs.values())[-n:]
            return [j.summary() for j in reversed(jobs)]
