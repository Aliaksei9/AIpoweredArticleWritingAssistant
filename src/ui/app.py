"""Same-origin local web UI; credentials stay on the Python server."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from src.controller import ApplicationError, EditorController
from src.controller.correction_jobs import CorrectionJobs

STATIC = Path(__file__).with_name("static")


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=50000)
    topic: str | None = Field(default=None, max_length=100)

    @field_validator("text")
    @classmethod
    def has_content(cls, value):
        if not value.strip():
            raise ValueError("Введите текст публикации")
        return value

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value):
        return value.strip() or None if value is not None else None


class PredictionInput(TextInput):
    resource_id: str = Field(min_length=1, max_length=200)
    source: str = Field(default="TG", min_length=1, max_length=30)


def create_app(controller=None):
    controller = controller or EditorController()
    jobs = CorrectionJobs(controller)
    language_jobs = CorrectionJobs(controller, operation="check_language")

    @asynccontextmanager
    async def lifespan(app):
        yield
        await jobs.aclose()
        await language_jobs.aclose()
        await controller.aclose()

    app = FastAPI(title="Редактор публикаций", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.controller, app.state.jobs = controller, jobs
    app.state.language_jobs = language_jobs
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_headers(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "DELETE", "PUT", "PATCH") and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse(dict(error="Запрос должен быть отправлен из локального редактора."), status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApplicationError)
    async def application_error(request, error):
        return JSONResponse(dict(error=str(error)), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse(dict(error="Проверьте текст и параметры. Максимальная длина — 50 000 символов."), status_code=422)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/workspace")
    async def workspace(refresh: bool = False):
        return await controller.workspace(refresh)

    @app.post("/api/predictions")
    async def predictions(data: PredictionInput):
        try:
            return await controller.evaluate(data.text, data.topic, data.resource_id, data.source)
        except ApplicationError:
            raise
        except Exception:
            raise ApplicationError("Не удалось рассчитать прогноз. Проверьте рабочие модели и выбранный канал.") from None

    @app.post("/api/corrections", status_code=202)
    async def start_corrections(data: TextInput):
        return jobs.create(data.text, data.topic)

    @app.get("/api/corrections/{identifier}")
    async def correction_status(identifier: str):
        return jobs.get(identifier)

    @app.delete("/api/corrections/{identifier}")
    async def cancel_correction(identifier: str):
        return jobs.cancel(identifier)

    @app.post("/api/language-checks", status_code=202)
    async def start_language_check(data: TextInput):
        return language_jobs.create(data.text, None)

    @app.get("/api/language-checks/{identifier}")
    async def language_check_status(identifier: str):
        return language_jobs.get(identifier)

    @app.delete("/api/language-checks/{identifier}")
    async def cancel_language_check(identifier: str):
        return language_jobs.cancel(identifier)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
