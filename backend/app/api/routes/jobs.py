from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_repository
from app.schemas.jobs import JobRecord
from app.storage.repository import Repository
from app.utils.errors import JobNotFound

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRecord)
async def get_job(job_id: str, repository: Repository = Depends(get_repository)) -> JobRecord:
    job = repository.get_job(job_id)
    if not job:
        raise JobNotFound()
    return job
