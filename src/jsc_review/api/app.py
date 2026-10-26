"""FastAPI application assembly."""

from fastapi import FastAPI

from jsc_review.api.routes.companies import router as companies_router
from jsc_review.api.routes.health import router as health_router
from jsc_review.api.routes.reviewer import router as reviewer_router
from jsc_review.api.routes.uploads import router as uploads_router
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
    application.include_router(reviewer_router)
    application.include_router(uploads_router)
    application.include_router(companies_router)
    return application


app = create_app()
