"""Bounded in-memory LLM jobs; stale browser work can be cancelled explicitly."""

import asyncio
from dataclasses import dataclass, field
import time
import uuid

from .editor_controller import ApplicationError, document_hash


@dataclass
class CorrectionJob:
    id: str
    text_hash: str
    status: str = "queued"
    result: dict | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    created_at: float = 0
    warnings: list[str] = field(default_factory=list)
    retrieval: dict | None = None

    def public(self):
        return dict(id=self.id, text_hash=self.text_hash, status=self.status, result=self.result, error=self.error,
                    warnings=self.warnings, retrieval=self.retrieval)


class CorrectionJobs:
    def __init__(self, controller, *, timeout_seconds=240, retention_seconds=1800, capacity=20, operation="correct"):
        self.controller = controller
        self.operation = operation
        self.timeout_seconds, self.retention_seconds, self.capacity = timeout_seconds, retention_seconds, capacity
        self.jobs = {}

    def create(self, text, topic):
        if any(j.status in ("queued", "running") for j in self.jobs.values()):
            raise ApplicationError("Запрос уже выполняется. Дождитесь результата или отмените его.", 409)
        for identifier, job in list(self.jobs.items()):
            if time.monotonic()-job.created_at > self.retention_seconds:
                del self.jobs[identifier]
        while len(self.jobs) >= self.capacity:
            del self.jobs[next(iter(self.jobs))]
        job = CorrectionJob(uuid.uuid4().hex, document_hash(text), created_at=time.monotonic())
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job, text, topic))
        return job.public()

    async def _run(self, job, text, topic):
        job.status = "running"
        def prepared(metadata):
            job.warnings = list(metadata.get("warnings", []))
            job.retrieval = metadata.get("retrieval")
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await getattr(self.controller, self.operation)(text, topic, on_prepared=prepared)
            if job.status != "cancelled":
                job.result, job.status = result, "succeeded"
                job.warnings = result.get("warnings", job.warnings)
                job.retrieval = result.get("retrieval", job.retrieval)
        except asyncio.CancelledError:
            job.status = "cancelled"
        except TimeoutError:
            job.error, job.status = "Сервис не ответил вовремя. Повторите запрос позже.", "failed"
        except ApplicationError as error:
            job.error, job.status = str(error), "failed"
        except Exception:
            job.error, job.status = "Не удалось обработать текст. Повторите запрос.", "failed"

    def get(self, identifier):
        job = self.jobs.get(identifier)
        if job is None:
            raise ApplicationError("Результат поиска больше недоступен. Запустите поиск заново.", 404)
        return job.public()

    def cancel(self, identifier):
        job = self.jobs.get(identifier)
        if job is None:
            raise ApplicationError("Поиск не найден.", 404)
        if job.status in ("queued", "running"):
            job.status = "cancelled"
            job.task.cancel()
        return job.public()

    async def aclose(self):
        tasks = []
        for job in self.jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
