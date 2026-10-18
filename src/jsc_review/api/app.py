"""FastAPI application assembly."""

from fastapi import FastAPI

from jsc_review.api.routes.health import router as health_router
from jsc_review.core.logging import configure_logging
from jsc_review.core.settings import get_settings


def create_app() -> FastAPI:
    """Create the API without embedding domain logic in transport code."""

    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title="JSC AI Review PoC",
        version="0.1.0",
        description="Local recommendations only; employees make final decisions.",
    )
    application.include_router(health_router)
    return application


app = create_app()
