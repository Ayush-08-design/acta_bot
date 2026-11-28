from fastapi import APIRouter, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.app.api.deps import engine

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def healthz() -> dict[str, str]:
    """Liveness check: confirms the process is running"""
    return {"message": {"status": "running"}}


@router.get("/ready")
async def readyz(response: Response) -> dict[str, str]:
    """Readiness check: verifies critical dependencies [db, cache]"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "database": "unavailable"}
    return {"status": "ok", "database": "ok"}
